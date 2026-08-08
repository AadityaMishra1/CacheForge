#include <vector>
#include <cstdint>
#include <cstring>
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

// RRIP params and tunables
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // need 2 forward (+1/+2) steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU threshold for hot PCs
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// decay TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch (ops)
static constexpr uint8_t  GSEL_MAX            = 31;  // global selector range
static constexpr uint8_t  GSEL_THRES          = 20;  // followers enable B only if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set bandit threshold

// Leader-set sampling (64 leaders total: 32 A, 32 B)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Per-line metadata (conceptual packing: rrpv:3, hitcnt:2, stream_lock:1)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// Per-set local selector (bandit), global selector
static int8_t  bandit_score[LLC_SETS]; // [-8..7] conceptual 4b
static uint8_t GSEL = 0;

// ------------- Mode A: Hawkeye-like SHiP fallback (safe) -------------
// Tiny SHCTs (1024 entries, 5-bit counters) for demand & prefetch
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Training state only for 64 leader sets (A side only)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective, stored in 16b
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse observed
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // was prefetched at fill

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ------------- Mode B: Prism (stream-safe DBP path) -------------------
// Compact per-PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 10b line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // last 10b line
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // PC deadness (0..3)

// Helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void gsel_inc() { if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec() { if (GSEL > 0) GSEL--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;            // always test Mode B on B-leaders
    if (LEADER_A(set)) return false;           // always test Mode A on A-leaders
    // Followers: enable Mode B only if both global and local gates say so
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Periodic decay and selector update
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 1, sizeof(shct_demand));
    std::memset(shct_prefetch, 1, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
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
    // Return any invalid first
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection: find maxRRPV, else age and retry
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
    op_count++;

    // Periodic decays
    if ((op_count & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;           // TinyLFU decay
            if (pc_cold2[i] > 0) pc_cold2[i]--; // ease coldness slowly
        }
    }
    if ((op_count & (BANDIT_EPOCH - 1)) == 0) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) gsel_inc();
        else gsel_dec();
        leaderA_hits_epoch = leaderB_hits_epoch = 0;
    }

    const bool leaderA = LEADER_A(set);
    const bool leaderB = LEADER_B(set);

    if (hit) {
        // Bandit training on hits
        if (leaderA) { leaderA_hits_epoch++; sat_dec_i8(bandit_score[set]); }
        else if (leaderB) { leaderB_hits_epoch++; sat_inc_i8(bandit_score[set]); }

        // Hawkeye reuse mark for A-leaders
        if (leaderA) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // PC accounting
        if (is_demand(type)) {
            uint32_t idx = pc_index(PC);
            sat_inc_u4(pc_use4[idx]);
            sat_dec_u2(pc_cold2[idx]);
        }

        // Promotion/Quarantine logic
        if (is_demand(type)) {
            if (stream_lock[set][way]) {
                // stream-locked: demote on touch until escape on 2nd demand hit
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    hitcnt[set][way] = 1;
                    rrpv_set(set, way, 0); // escape to MRU
                } else {
                    if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                    rrpv_demote(set, way); // keep near tail
                }
            } else {
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_NS) {
                    hitcnt[set][way] = 1;
                    rrpv_set(set, way, 0); // MRU on confirmed reuse
                } else {
                    if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                    rrpv_promote(set, way); // gentle nudge
                }
            }
        } else if (is_prefetch(type)) {
            // keep prefetches quarantined near tail
            rrpv_demote(set, way);
        }
        return;
    }

    // Miss path: bandit training on misses
    if (leaderA) sat_inc_i8(bandit_score[set]);
    else if (leaderB) sat_dec_i8(bandit_score[set]);

    // Hawkeye training on eviction (A-leaders only)
    if (leaderA) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig = hawk_sig[slot][way];
        uint8_t used = hawk_used[slot][way];
        uint8_t was_pref = hawk_is_pref[slot][way];
        if (sig || was_pref || used) {
            uint32_t idx = shct_idx(sig);
            if (used) {
                if (was_pref) shct_inc(shct_prefetch[idx]);
                else          shct_inc(shct_demand[idx]);
            } else {
                if (was_pref) shct_dec(shct_prefetch[idx]);
                else          shct_dec(shct_demand[idx]);
            }
        }
        // Install new signature bookkeeping for incoming line
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }

    // Decide mode for followers
    bool useB = modeB_enabled(set);
    if (leaderA) useB = false;
    else if (leaderB) useB = true;

    // Update PC stream detector (+1/+2 forward run), skip WB
    bool stream_conf = false;
    uint32_t pidx = pc_index(PC);
    if (!is_writeback(type)) {
        uint16_t ln = line10(paddr);
        uint16_t last = pc_last_line10[pidx];
        bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
        if (forward) { if (pc_stream_conf[pidx] < 3) pc_stream_conf[pidx]++; }
        else pc_stream_conf[pidx] = 0;
        pc_last_line10[pidx] = ln;
        stream_conf = (pc_stream_conf[pidx] >= STREAM_CONF_THRESH);
    }

    // PC-level accounting
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pidx]); // bump LFU on activity
        sat_inc_u2(pc_cold2[pidx]); // assume cold until a hit proves otherwise
    }

    // Choose insertion depth and stream lock
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t lock_stream = 0;

    if (useB) {
        if (!is_writeback(type) && stream_conf) {
            // Strong stream guard: hard tail (bypass-ish), with escape-on-2
            lock_stream = 1;
            if (is_prefetch(type)) {
                ins_rrpv = maxRRPV; // quarantine
            } else {
                // Hot PCs get a slightly shallower tail to allow short reuse (zeusmp)
                ins_rrpv = (pc_use4[pidx] >= PC_USE_HOT_THRESH) ? 5 : maxRRPV;
            }
        } else {
            // Non-stream: PC-guided usefulness
            if (is_prefetch(type)) {
                ins_rrpv = maxRRPV; // always quarantine prefetches
            } else {
                ins_rrpv = (pc_cold2[pidx] >= 2) ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            }
        }
    } else {
        // Mode A: Hawkeye-like SHiP insertion
        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV;
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t ctr = shct_demand[idx];
            if (ctr < 8)      ins_rrpv = maxRRPV;         // predicted dead
            else if (ctr < 24) ins_rrpv = INSERT_COLD_DEPTH; // moderate confidence
            else               ins_rrpv = INSERT_WARM_DEPTH; // hot
        }
    }

    // Apply insertion
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock_stream;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}