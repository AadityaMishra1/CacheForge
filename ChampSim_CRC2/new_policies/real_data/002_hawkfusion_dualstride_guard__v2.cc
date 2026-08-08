#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 sets) -------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for sampled

// ---------------- Tunables (DualStride Guard) ---------------------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit SRRIP domain
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU for alive/hot
static constexpr uint8_t  INSERT_COOL_DEPTH   = 5;   // moderate tail
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail for cold
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;   // hard tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // two forward steps (+1/+2)
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: 2nd demand hit promotes
static constexpr uint8_t  HITS_PROMOTE_STR    = 3;   // stream-tagged: 3rd demand hit to clear tag+promote
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;   // demote streams on any touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 8;   // TinyLFU use threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH   = 2;   // PC coldness (0..3)
static constexpr uint32_t LFU_DECAY_PERIOD    = 4096;// decay/aging cadence
static constexpr uint32_t BANDIT_EPOCH        = 8192;// selector epoch in accesses
static constexpr int8_t   MODEB_ENABLE_MARGIN = 2;   // B must beat A by >=2
static constexpr int8_t   BANDIT_MIN          = -8;
static constexpr int8_t   BANDIT_MAX          = 7;

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (demand hits), stream_tag:1b -> total 6 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set/global selectors -----------------------
static int8_t   bandit_score[LLC_SETS]; // [-8..7]
static uint8_t  use_modeB_global = 0;   // 0=A, 1=B
static uint64_t access_count = 0;
static int32_t  leaderA_score = 0;      // hits - misses in leader A
static int32_t  leaderB_score = 0;      // hits - misses in leader B

// ---------------- Mode A (Hawkeye-like via tiny SHCT) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static constexpr uint8_t SHCT_THRES_ALIVE = 16; // >= alive; else dead

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader A per-line buffers for training (only 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

// ---------------- Mode B (DualStride + TinyLFU) ------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B -> 10b

// Per-PC state: 10b last_line, 2b stream_conf, 4b use, 2b coldness
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE];
static uint8_t  pc_use4[PC_TBL_SIZE];
static uint8_t  pc_cold2[PC_TBL_SIZE];

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < BANDIT_MAX) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > BANDIT_MIN) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// DualStride: +1/+2 forward increases confidence; -1/-2 backward exempts/decays
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];

    // Compute small strides in the 10-bit space
    bool fwd1 = (ln == (uint16_t)(last + 1));
    bool fwd2 = (ln == (uint16_t)(last + 2));
    bool bwd1 = (ln == (uint16_t)(last - 1));
    bool bwd2 = (ln == (uint16_t)(last - 2));

    if (fwd1 || fwd2) {
        sat_inc_u2(pc_stream_conf[idx]);
    } else if (bwd1 || bwd2) {
        // backward small loop exemption: decay confidence
        sat_dec_u2(pc_stream_conf[idx]);
    } else {
        // unrelated access/path change resets confidence slowly
        if (pc_stream_conf[idx] > 0) pc_stream_conf[idx]--;
    }

    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

void InitReplacementState() {
    std::memset(rrpv,        0, sizeof(rrpv));
    std::memset(hitcnt,      0, sizeof(hitcnt));
    std::memset(stream_tag,  0, sizeof(stream_tag));
    std::memset(bandit_score,0, sizeof(bandit_score));
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch,0,sizeof(shct_prefetch));
    std::memset(hawk_sig,    0, sizeof(hawk_sig));
    std::memset(hawk_used,   0, sizeof(hawk_used));
    std::memset(hawk_is_pref,0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10,0,sizeof(pc_last_line10));
    std::memset(pc_stream_conf,0,sizeof(pc_stream_conf));
    std::memset(pc_use4,     0, sizeof(pc_use4));
    std::memset(pc_cold2,    0, sizeof(pc_cold2));

    // Initialize RRIP to "long re-ref" state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
        }
    }
    use_modeB_global = 0;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP victim selection with aging; guaranteed to terminate
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Update replacement state
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    (void)cpu; (void)victim_addr;

    access_count++;

    // Leader accounting (who "owns" this set)
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    if (leaderA) { leaderA_score += (hit ? 1 : -1); }
    if (leaderB) { leaderB_score += (hit ? 1 : -1); }

    // Periodic global selector update
    if ((access_count % BANDIT_EPOCH) == 0) {
        use_modeB_global = (leaderB_score >= (leaderA_score + MODEB_ENABLE_MARGIN)) ? 1 : 0;
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Determine mode used for this access
    bool use_modeB = leaderB || (!leaderA && (use_modeB_global && (bandit_score[set] >= 0)));

    // Always update stream detector history (cheap)
    bool stream_now = detect_and_update_stream(PC, paddr);

    // Update TinyLFU-ish PC stats
    uint32_t pcidx = pc_index(PC);
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pcidx]);
        if (hit) {
            if (pc_cold2[pcidx] > 0) pc_cold2[pcidx]--;
        } else {
            if (pc_cold2[pcidx] < 3) pc_cold2[pcidx]++;
        }
    }

    // Per-set bandit updates
    if (!leaderA && !leaderB) {
        if (use_modeB) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else     sat_dec_i8(bandit_score[set]);
        } else {
            // slow decay toward neutral when using Mode A
            if (bandit_score[set] > 0) bandit_score[set]--;
            else if (bandit_score[set] < 0) bandit_score[set]++;
        }
    }

    // LFU decay and stream confidence leakage
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i]--;
            if (pc_cold2[i] > 0) pc_cold2[i]--; // prevent runaway coldness
            if (pc_stream_conf[i] > 0) pc_stream_conf[i]--; // decay runs between phases
        }
    }

    // Hit path
    if (hit) {
        // Hawkeye training: mark reuse on leader A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Stream-aware touch behavior
        if (stream_tag[set][way]) {
            // scan resistance: demote on any touch
            for (uint8_t k = 0; k < STREAM_DEMOTE_TOUCH; k++) rrpv_demote(set, way);
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    // proven short-reuse despite stream appearance: clear tag and promote
                    stream_tag[set][way] = 0;
                    rrpv_set(set, way, 1); // near-MRU
                }
            }
        } else {
            // non-stream: gated promotion
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 1); // promote near MRU
                } else {
                    // gentle promotion
                    rrpv_promote(set, way);
                }
            } else {
                // prefetch hits do not promote on first touch
                rrpv_promote(set, way);
            }
        }
        return;
    }

    // Miss / Fill path
    // Hawkeye training on eviction for leader A: update SHCT using victim's recorded reuse
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train with prior occupant of 'way'
        uint16_t vsig = hawk_sig[slot][way];
        uint32_t vidx = shct_idx(vsig);
        if (hawk_is_pref[slot][way]) {
            if (hawk_used[slot][way]) shct_inc(shct_prefetch[vidx]);
            else                      shct_dec(shct_prefetch[vidx]);
        } else {
            if (hawk_used[slot][way]) shct_inc(shct_demand[vidx]);
            else                      shct_dec(shct_demand[vidx]);
        }
        // Prepare new occupant's training slots
        hawk_sig[slot][way]      = pc_sig12(PC);
        hawk_used[slot][way]     = 0;
        hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
    }

    // Choose insertion depth and tags
    uint8_t ins_rrpv = INSERT_COOL_DEPTH;
    uint8_t new_stream_tag = 0;

    if (is_writeback(type)) {
        // never bypass WB; keep them harmless
        ins_rrpv = STREAM_TAIL_DEPTH;
        new_stream_tag = 0;
    } else if (use_modeB) {
        // Mode B: DualStride + TinyLFU coldness/use
        bool is_stream = stream_now;
        bool pc_cold   = (pc_cold2[pcidx] >= PC_COLD_STRICT_TH);
        bool pc_hot    = (pc_use4[pcidx]  >= PC_USE_HOT_THRESH);

        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH; // quarantine
            new_stream_tag = is_stream ? 1 : 0; // remember if it looked like a stream
        } else if (is_stream) {
            // aggressive stream control: hard tail, tagged, no early promotion
            ins_rrpv = STREAM_TAIL_DEPTH;
            new_stream_tag = 1;
        } else if (pc_hot) {
            ins_rrpv = INSERT_WARM_DEPTH;
            new_stream_tag = 0;
        } else if (pc_cold) {
            ins_rrpv = INSERT_COLD_DEPTH;
            new_stream_tag = 0;
        } else {
            ins_rrpv = INSERT_COOL_DEPTH;
            new_stream_tag = 0;
        }
    } else {
        // Mode A: Hawkeye-like SHCT-guided insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_prefetch(type)) {
            // aggressive tail for prefetch
            ins_rrpv = STREAM_TAIL_DEPTH;
        } else {
            bool alive = (shct_demand[idx] >= SHCT_THRES_ALIVE);
            ins_rrpv = alive ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
        new_stream_tag = 0;
    }

    // Install metadata
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = new_stream_tag;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}