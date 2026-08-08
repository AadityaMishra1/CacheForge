#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// -------------- RRIP and per-line state (bit-packed conceptually) --------------
static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP
static uint8_t rrpv[LLC_SETS][LLC_WAYS];     // 3b effective
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];   // 2b effective (0..3)
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1b: quarantined stream/prefetch

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
}

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables ----------------
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for hot streams
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // +1/+2 run to arm
static constexpr uint8_t STRONG_STREAM_CONF  = 3;  // strong at 3 steps
static constexpr uint8_t HITS_PROMOTE_BASE   = 2;  // non-stream base
static constexpr uint8_t HITS_PROMOTE_COLD   = 3;  // cold PC
static constexpr uint8_t HITS_PROMOTE_HOTSTR = 1;  // hot stream escape
static constexpr uint8_t PREFETCH_QUARANTINE = 2;  // touches to escape
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote-on-touch while quarantined

// ---------------- Selector (bandit) ----------------
static constexpr uint32_t BANDIT_EPOCH = 4096;
static constexpr int32_t  GATE_MARGIN  = 4; // stronger bias to Mode A

static bool     prefer_B = false;
static int32_t  leaderA_good = 0;
static int32_t  leaderB_good = 0;
static uint64_t access_count = 0;

// ---------------- Hawkeye-like SHCT (Mode A bias + gating) ----------------
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static constexpr uint8_t SHCT_WARM_THRESH = 16; // >= warm

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)
static uint8_t  hawk_valid[64][LLC_WAYS];   // valid (0/1)

// ---------------- Mode B: Stream detector + TinyLFU ----------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B->line

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 0..15

static constexpr uint8_t  PC_USE_HOT_THRESH = 8;
static constexpr uint32_t LFU_DECAY_PERIOD  = 1024;

// Helpers
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }

// Returns updated stream confidence
static inline uint8_t stream_update(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) {
        if (pc_stream_conf[idx] < STRONG_STREAM_CONF) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return pc_stream_conf[idx];
}

static inline bool pc_hot(uint64_t PC) {
    return pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH;
}

static inline void lfu_decay_if_needed() {
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    // Per-line
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    // SHCT
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    // Leader training buffers
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
            hawk_valid[ls][w] = 0;
        }
    }
    // PC tables
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    // Selector
    prefer_B = false;
    leaderA_good = leaderB_good = 0;
    access_count = 1; // avoid immediate decay
}

// ---------------- Victim selection ----------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim search with bounded aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
}

// ---------------- Update state (hit or fill) ----------------
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
    // Update TinyLFU and stream detector on all accesses
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
        access_count++;
        lfu_decay_if_needed();
    }
    uint8_t sconf = stream_update(PC, paddr);

    // Bandit accounting on leader sets (demand only)
    if (SEL_SAMPLED(set) && is_demand(type)) {
        if (LEADER_A(set)) {
            leaderA_good += (hit ? 1 : -1);
        } else if (LEADER_B(set)) {
            leaderB_good += (hit ? 1 : -1);
        }
        // Epoch gate
        if ((access_count % BANDIT_EPOCH) == 0) {
            prefer_B = (leaderB_good > (leaderA_good + GATE_MARGIN));
            leaderA_good = leaderB_good = 0;
        }
    }

    // Leader Hawkeye reuse marking on demand hits
    if (hit && SEL_SAMPLED(set) && is_demand(type)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_used[slot][way] = 1;
    }

    // Promotion on hit
    if (hit) {
        bool hot = pc_hot(PC);
        uint16_t sig = pc_sig12(PC);
        uint32_t sidx = shct_idx(sig);
        bool pc_warm = (shct_demand[sidx] >= SHCT_WARM_THRESH);

        uint8_t thr = HITS_PROMOTE_BASE;
        if (stream_tag[set][way]) {
            thr = hot ? HITS_PROMOTE_HOTSTR : PREFETCH_QUARANTINE;
        } else if (!pc_warm) {
            thr = HITS_PROMOTE_COLD;
        }

        // Quarantine handling: demote-on-touch until threshold met
        if (stream_tag[set][way] && (hitcnt[set][way] < thr)) {
            if (STREAM_DEMOTE_TOUCH && !hot && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }

        // Increment hit count (saturate at 3) and decide promotion
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        if (hitcnt[set][way] >= thr) {
            // Escape quarantine and promote to MRU
            stream_tag[set][way] = 0;
            rrpv_set(set, way, 0);
        }
        return;
    }

    // Miss path: train Hawkeye on eviction for leader sets
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hawk_valid[slot][way]) {
            uint16_t old_sig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(old_sig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                else                      shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                else                      shct_dec(shct_demand[idx]);
            }
        }
        // Initialize new training entry
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]    = 0;
        hawk_valid[slot][way]   = 1;
    }

    // Decide mode for this set
    bool forceA = LEADER_A(set);
    bool forceB = LEADER_B(set);
    bool useB = forceB || (!forceA && prefer_B);

    // Compute insertion depth and quarantine tag
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;
    uint8_t new_hitcnt = 0;

    // SHCT warm/cold for non-streams
    uint16_t sig = pc_sig12(PC);
    uint32_t sidx = shct_idx(sig);
    bool pc_warm = (shct_demand[sidx] >= SHCT_WARM_THRESH);

    if (is_writeback(type)) {
        ins_depth = STREAM_TAIL_DEPTH; // never bypass writebacks
        new_stream_tag = 0;
        new_hitcnt = 0;
    } else if (useB) {
        if (is_prefetch(type)) {
            ins_depth = STREAM_TAIL_DEPTH;
            new_stream_tag = 1; // quarantine
            new_hitcnt = 0;
        } else if (sconf >= STRONG_STREAM_CONF) {
            ins_depth = STREAM_TAIL_DEPTH;
            new_stream_tag = 1;
            new_hitcnt = 0;
        } else if (sconf >= STREAM_ARM_THRESH) {
            ins_depth = pc_hot(PC) ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            new_stream_tag = 1;
            new_hitcnt = 0;
        } else {
            ins_depth = pc_warm ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_tag = 0;
            new_hitcnt = 0;
        }
    } else { // Mode A (Hawkeye-like)
        if (is_prefetch(type)) {
            ins_depth = STREAM_TAIL_DEPTH;
            new_stream_tag = 1; // quarantine prefetch
            new_hitcnt = 0;
        } else {
            ins_depth = pc_warm ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_tag = 0;
            new_hitcnt = 0;
        }
    }

    rrpv_set(set, way, ins_depth);
    stream_tag[set][way] = new_stream_tag;
    hitcnt[set][way] = new_hitcnt;
}

// Print end-of-simulation statistics
void PrintStats() {
    // keep blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // keep blank
}