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

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // Stream-Guardian
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (R5T) ---------------------------------
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail insertion
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for confident streams
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;   // shallower tail for TinyLFU-hot streams
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // +1/+2 forward steps needed
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 3;   // stream promote on 3rd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;   // hot-PC stream early escape
static constexpr uint8_t  PC_USE_HOT_THRESH     = 8;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH     = 2;   // cold if >=2 (0..3)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024;// fast decay
static constexpr uint32_t BANDIT_EPOCH          = 8192;// calmer selector epoch
static constexpr uint8_t  GSEL_MAX              = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES            = 20;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;   // per-set bias threshold

// ---------------- Per-line metadata (concept: 3b rrpv, 2b hitcnt, 1b stream) ----
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set and global selectors -------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B
static uint8_t GSEL = 0;               // global gate (0..31)
static uint64_t access_count = 0;

// Leader performance accumulators (hysteretic global gate)
static int32_t leaderA_good = 0; // hits - misses
static int32_t leaderB_good = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHiP) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b PC signature hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A per-line training buffers (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (Stream-Guardian PC tables) --------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

// Per-PC state (information-theoretic: 10b, 2b, 4b, 2b)
static uint16_t pc_last_line10[PC_TBL_SIZE]; // last line (10b in 16b storage)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // coldness 2b (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_u5(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// +1/+2 forward-run detector with 2-step confidence; reset on non-forward
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr, bool do_update) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];

    if (do_update) {
        if ((uint16_t)(last + 1) == ln || (uint16_t)(last + 2) == ln) {
            if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
        } else {
            pc_stream_conf[idx] = 0; // break the run
        }
        pc_last_line10[idx] = ln;
    }
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Policy core ------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    GSEL = 0;
    leaderA_good = leaderB_good = 0;
    access_count = 0;
}

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
    // SRRIP victim selection with bounded aging
    for (uint32_t attempt = 0; attempt <= maxRRPV; attempt++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    return 0; // fallback
}

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

    // Periodic TinyLFU decay (global, light)
    access_count++;
    if ((access_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
            sat_dec_u2(pc_cold2[i]);
        }
        // Selector epoch end: compare leaders and adjust gate
        if (leaderB_good > leaderA_good) sat_inc_u5(GSEL, GSEL_MAX);
        else if (leaderB_good < leaderA_good) sat_dec_u5(GSEL);
        leaderA_good = leaderB_good = 0;
    }

    bool demand = is_demand(type);
    bool pref   = is_prefetch(type);
    bool wb     = is_writeback(type);

    // PC-based updates
    uint32_t pc_idx = pc_index(PC);
    if (demand) sat_inc_u4(pc_use4[pc_idx]); // TinyLFU on demand only

    // Stream detector: update on demand accesses
    bool stream_now = detect_and_update_stream(PC, paddr, demand);
    bool pc_hot  = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);
    bool pc_cold = (pc_cold2[pc_idx] >= PC_COLD_STRICT_TH);

    // Choose mode (leaders are pinned; followers default to Mode A unless B proven)
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    bool preferB = (!leaderA && !leaderB) && (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
    enum { MODE_A = 0, MODE_B = 1 };
    uint8_t mode = leaderA ? MODE_A : (leaderB ? MODE_B : (preferB ? MODE_B : MODE_A));

    if (hit) {
        // Train leader metrics (hit = good)
        if (leaderA) leaderA_good++;
        else if (leaderB) leaderB_good++;

        // Leader-A reuse mark for SHiP training (demand hits only)
        if (leaderA && demand) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Multi-hit gating and stream demotion
        if (demand) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (stream_tag[set][way]) {
                uint8_t need = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
                if (hitcnt[set][way] >= need) {
                    rrpv_set(set, way, 0); // escape quarantine
                    stream_tag[set][way] = 0; // treat as regular after proving reuse
                } else {
                    rrpv_demote(set, way); // demote-on-touch for stream-tagged until proven
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // normal promotion on 2nd+ demand hit
                } else {
                    rrpv_promote(set, way); // light benefit
                }
            }
            // Demand hit => PC likely not cold
            sat_dec_u2(pc_cold2[pc_idx]);
        } else if (pref) {
            // Prefetch hits do not promote
            if (stream_tag[set][way]) rrpv_demote(set, way);
        }

        // Per-set bandit adjustment for followers (calm)
        if (!leaderA && !leaderB) {
            if (mode == MODE_B) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
        }
        return;
    }

    // Miss path (fill/insertion)
    // Train leader metrics (miss = bad)
    if (leaderA) leaderA_good--;
    else if (leaderB) leaderB_good--;

    // Follower bandit adjustment on miss
    if (!leaderA && !leaderB) {
        if (mode == MODE_B) sat_dec_i8(bandit_score[set]); else sat_inc_i8(bandit_score[set]);
    }

    // SHiP training on eviction for Leader-A (before overwrite)
    if (leaderA) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t osig = hawk_sig[slot][way];
        uint8_t  ouse = hawk_used[slot][way];
        uint8_t  opf  = hawk_is_pref[slot][way];
        if (osig) {
            uint32_t hx = shct_idx(osig);
            if (ouse) {
                if (opf) shct_inc(shct_prefetch[hx]);
                else     shct_inc(shct_demand[hx]);
            } else {
                if (opf) shct_dec(shct_prefetch[hx]);
                else     shct_dec(shct_demand[hx]);
            }
        }
    }

    // Decide insertion depth and tags
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t tag_stream = 0;

    if (wb) {
        ins_rrpv = INSERT_WARM_DEPTH; // never bypass writebacks
        tag_stream = 0;
    } else if (pref) {
        // Quarantine prefetches deeper; stream prefetches at hard tail
        ins_rrpv = stream_now ? STREAM_TAIL_DEPTH : STREAM_TAIL_DEPTH;
        tag_stream = stream_now ? 1 : 0;
    } else {
        // demand miss
        if (mode == MODE_A) {
            // Tiny SHiP: warm if predicted reuse-friendly PC
            uint16_t sig = pc_sig12(PC);
            uint32_t hx  = shct_idx(sig);
            bool warm = (shct_demand[hx] > 0);
            ins_rrpv = warm ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

            // Leader-A bookkeeping for new line
            if (leaderA) {
                uint32_t slot = LEADER_SLOT(set);
                hawk_sig[slot][way] = sig;
                hawk_used[slot][way] = 0;
                hawk_is_pref[slot][way] = 0;
            }
            tag_stream = 0; // Mode A stays conservative
        } else { // MODE_B
            if (stream_now && !pc_hot) {
                ins_rrpv = STREAM_TAIL_DEPTH; // strong bypass for scans
                tag_stream = 1;
            } else if (stream_now && pc_hot) {
                ins_rrpv = STREAM_SHALLOW_TAIL; // allow short-reuse streams to survive briefly
                tag_stream = 1;
            } else if (pc_cold) {
                ins_rrpv = INSERT_COLD_DEPTH; // suppress noisy PCs
                tag_stream = 0;
            } else {
                ins_rrpv = INSERT_WARM_DEPTH; // default warm insertion
                tag_stream = 0;
            }
        }
        // Demand miss => tentatively colder PC
        sat_inc_u2(pc_cold2[pc_idx]);
    }

    // Install
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = tag_stream;

    // If leaderA and this was a prefetch fill, record it for SHiP training
    if (leaderA && pref) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig = pc_sig12(PC);
        hawk_sig[slot][way] = sig;
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = 1;
    }
}

void PrintStats() {
    // keep blank
}

void PrintStats_Heartbeat() {
    // keep blank
}