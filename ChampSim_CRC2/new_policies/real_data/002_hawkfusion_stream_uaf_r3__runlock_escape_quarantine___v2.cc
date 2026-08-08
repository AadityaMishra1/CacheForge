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

// Leader-set sampling (64 leaders total: 32 A, 32 B)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Tunables (R3 retuned)
static constexpr uint8_t  maxRRPV             = 7;   // 3b RRIP window
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU for reused
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail for cold
static constexpr uint8_t  INSERT_HOT_SHORT4   = 4;   // for short-reuse hot PCs
static constexpr uint8_t  INSERT_HOT_SHORT5   = 5;   // slightly deeper
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream/quarantine escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot PCs
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// periodic decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set enable

// Per-line metadata (conceptually packed bitfields: rrpv:3, hitcnt:2, runlock:1)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t runlock[LLC_SETS][LLC_WAYS]; // 1=promotion locked (stream/quarantine), 0=normal

// Per-set and global selectors
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B (conceptually 4b)
static uint8_t GSEL = 0;               // global gate (0..31)

// Mode A (Hawkeye-like SHiP) components
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
// Per-line training only in leader-A sets (64 x 16 = 1024)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Mode B PC predictors (stride runs + TinyLFU + coldness)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b id
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b deadness

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

// Forward-run detector (+1/+2 stride) — update on demand only
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr, bool demand) {
    if (!demand) return false;
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;             // force Mode B on B leaders
    if (LEADER_A(set)) return false;            // force Mode A on A leaders
    // Followers: enable Mode B only if both global and local scores suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Periodic decay and selector update
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;
    // Decay TinyLFU and PC coldness (lazy, periodic)
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1; // halve
            if (pc_cold2[i] > 0) pc_cold2[i]--;   // warm slightly
        }
    }
    // Bandit epoch update
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) gsel_inc();
        else if (leaderB_hits_epoch < leaderA_hits_epoch) gsel_dec();
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start old
            hitcnt[s][w] = 0;
            runlock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    // Initialize SHCT slightly optimistic to avoid over-bypass at cold start
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 1;
        shct_prefetch[i] = 0;
    }
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
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim selection with safe aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // unreachable
    // return 0;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    periodic_housekeeping();

    bool demand = is_demand(type);
    bool pref   = is_prefetch(type);

    // Update per-PC stream detector on demand refs
    bool is_stream = detect_and_update_stream(PC, paddr, demand);

    // Mode selection
    bool use_modeB = modeB_enabled(set);

    // Leader-set hit accounting and per-set bandit nudging
    if (hit) {
        if (LEADER_A(set)) {
            leaderA_hits_epoch++;
            sat_dec_i8(bandit_score[set]);
        } else if (LEADER_B(set)) {
            leaderB_hits_epoch++;
            sat_inc_i8(bandit_score[set]);
        }
    }

    // SHiP-style training for Mode A: only on leader-A sets, when a line is evicted (i.e., on fill)
    if (!hit && LEADER_A(set)) {
        uint32_t ls = LEADER_SLOT(set);
        // Train previous occupant in the victim way
        uint16_t old_sig = hawk_sig[ls][way];
        uint8_t  old_used = hawk_used[ls][way];
        uint8_t  old_pref = hawk_is_pref[ls][way];
        if (old_sig != 0) {
            uint32_t idx = shct_idx(old_sig);
            if (old_pref) {
                if (old_used) shct_inc(shct_prefetch[idx]);
                else          shct_dec(shct_prefetch[idx]);
            } else {
                if (old_used) shct_inc(shct_demand[idx]);
                else          shct_dec(shct_demand[idx]);
            }
        }
        // Initialize new occupant metadata
        hawk_sig[ls][way]     = pc_sig12(PC);
        hawk_is_pref[ls][way] = pref ? 1 : 0;
        hawk_used[ls][way]    = 0;
    }

    // On hit: apply promotion policy
    if (hit) {
        if (demand) {
            // Demand hit updates for PC predictors
            uint32_t pidx = pc_index(PC);
            sat_inc_u4(pc_use4[pidx]);
            sat_dec_u2(pc_cold2[pidx]);
        }

        // Multi-hit gating
        if (runlock[set][way]) {
            // Promotion-locked line: stream/quarantine
            if (demand) {
                // Demote-on-first-touch if it was a hard stream lock (rrpv==7)
                if (hitcnt[set][way] == 0 && rrpv[set][way] == maxRRPV) {
                    rrpv_demote(set, way); // stay old on first hit to resist scans
                }
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    runlock[set][way] = 0; // escape
                    rrpv_set(set, way, 0); // MRU
                }
            }
            // Prefetch hits do not unlock/promote
        } else {
            // Normal line: promote only on 2nd demand hit
            if (demand) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU on 2nd+
                }
            } else {
                // No promotion on prefetch/writeback
            }
        }

        // Mark reuse for Hawkeye training in leader-A sets
        if (LEADER_A(set)) {
            uint32_t ls = LEADER_SLOT(set);
            hawk_used[ls][way] = 1;
        }

        return;
    }

    // Miss path: decide initial insertion for the incoming line (victim selected = 'way')
    // Reset per-line counters
    hitcnt[set][way] = 0;
    runlock[set][way] = 0;

    // Update PC deadness on miss (coarse, without per-line PC tag)
    if (demand) {
        uint32_t pidx = pc_index(PC);
        if (pc_cold2[pidx] < 3) pc_cold2[pidx]++; // nudge colder on miss
    }

    // Mode A insertion (Hawkeye-like SHiP)
    auto modeA_insert = [&](void) {
        uint8_t ins_rrpv = INSERT_WARM_DEPTH;
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);

        if (pref) {
            // Prefetches always quarantine near tail in Mode A
            ins_rrpv = INSERT_COLD_DEPTH;
        } else {
            uint8_t c = shct_demand[idx];
            if (c == 0) {
                ins_rrpv = maxRRPV; // predicted dead
            } else if (c <= 2) {
                ins_rrpv = INSERT_COLD_DEPTH; // cold
            } else {
                ins_rrpv = INSERT_WARM_DEPTH; // warm
            }
        }

        rrpv_set(set, way, ins_rrpv);
        // Quarantine prefetched lines (no first-hit promotion)
        if (pref) runlock[set][way] = 1;
    };

    // Mode B insertion (stream+deadness boost)
    auto modeB_insert = [&](void) {
        uint32_t pidx = pc_index(PC);
        uint8_t ins_rrpv = INSERT_WARM_DEPTH;

        if (pref) {
            // Prefetch quarantine slightly shallower than hard stream
            ins_rrpv = INSERT_COLD_DEPTH; // 6
            runlock[set][way] = 1;
        } else if (is_stream) {
            // Hard tail insertion with 7/8 "bypass" probability; stay locked
            uint8_t hash = (uint8_t)(((PC >> 2) ^ (paddr >> 6)) & 7u);
            ins_rrpv = (hash != 0) ? maxRRPV : INSERT_COLD_DEPTH; // mostly RRIP=7, sometimes 6
            runlock[set][way] = 1; // RunLock until proven with 2 demand hits
        } else {
            // Non-stream: PC-guided insertion depth
            uint8_t use = pc_use4[pidx];
            uint8_t cold = pc_cold2[pidx];
            if (cold >= 2) {
                ins_rrpv = INSERT_COLD_DEPTH; // cold PCs: 6
            } else if (use >= PC_USE_HOT_THRESH) {
                ins_rrpv = INSERT_HOT_SHORT4; // hot short-reuse: 4
            } else if (use >= (PC_USE_HOT_THRESH - 2)) {
                ins_rrpv = INSERT_HOT_SHORT5; // modestly hot: 5
            } else {
                ins_rrpv = INSERT_WARM_DEPTH; // default warm: 2
            }
        }

        rrpv_set(set, way, ins_rrpv);
    };

    if (use_modeB) modeB_insert();
    else           modeA_insert();
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}