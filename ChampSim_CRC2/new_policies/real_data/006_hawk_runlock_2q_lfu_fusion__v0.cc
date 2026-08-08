#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

// CRC2 constants
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (aligned to ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables ----------------
static constexpr uint8_t  maxRRPV             = 7;     // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;     // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;     // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;     // +1/+2 forward steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;     // non-stream MRU at 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;     // stream escape at 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;     // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;  // LFU decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;  // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;    // global gate (0..31)
static constexpr uint8_t  GSEL_THRES          = 16;    // followers need GSEL >= this
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;     // per-set enable

// ---------------- Per-line metadata (conceptual packing) ---------
// rrpv:3b, hitcnt:2b (0..3), runlock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t runlock_[LLC_SETS][LLC_WAYS];

// ---------------- Selector state ---------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], followers default 0 (Mode A)
static uint8_t GSEL = 0;               // global gate

// ---------------- Mode A (Hawkeye-like SHiP) ---------------------
// Tiny SHCTs (demand & prefetch), trained on sampled leader sets (A)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }

// Per-line signature and tags only in 64 leader sets (to train SHCT)
static uint16_t hawk_sig[64][LLC_WAYS];    // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];   // saw demand reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS];// was filled by prefetch

// ---------------- Mode B (RunLock + 2Q-LFU) ----------------------
// 512-entry PC tables: last line id, stream conf, LFU4, coldness2
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b in 16b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
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
    if (LEADER_B(set)) return true;   // force Mode B in B leaders
    if (LEADER_A(set)) return false;  // force Mode A in A leaders
    // Followers: enable B only if both global and local agree
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Epoch bookkeeping
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;

    // Periodic LFU/coldness decay
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1; // halve LFU
            if (pc_cold2[i] > 0) pc_cold2[i]--;   // slowly un-cold
        }
    }

    // Selector epoch update
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch >= leaderA_hits_epoch) {
            if (GSEL < GSEL_MAX) GSEL++;
        } else {
            if (GSEL > 0) GSEL--;
        }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            runlock_[s][w] = 0;
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
    op_count = leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_find_victim(uint32_t set, const BLOCK* current_set, uint64_t PC) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV, age bounded to ensure termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                // Negative training for Mode A on eviction (A leaders only)
                if (LEADER_A(set)) {
                    uint32_t slot = LEADER_SLOT(set);
                    // Only train on lines that did not see demand reuse
                    uint16_t sig = hawk_sig[slot][w];
                    uint32_t idx = shct_idx(sig);
                    if (hawk_is_pref[slot][w]) shct_dec(shct_prefetch[idx]);
                    else if (!hawk_used[slot][w]) shct_dec(shct_demand[idx]);
                }
                // Penalize current PC if it just evicted a single-use line (dead), approximate
                uint32_t pcidx = pc_index(PC);
                if (hitcnt[set][w] == 0) { // dead line
                    if (pc_cold2[pcidx] < 3) pc_cold2[pcidx]++;
                }
                return w;
            }
        }
        // Age all lines by one
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: return the way with largest RRPV (ties -> highest index)
    uint32_t victim = 0, best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)type; (void)paddr;
    return rrip_find_victim(set, current_set, PC);
}

// ---------------- Update state on hit/fill -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    // Demand touches contribute to LFU hotness
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    bool stream = detect_and_update_stream(PC, paddr);
    bool useB = modeB_enabled(set);

    // Hits: multi-hit promotion with RunLock demote-on-touch
    if (hit) {
        if (runlock_[set][way]) {
            // quarantine: demote on any touch
            rrpv_demote(set, way);
        }
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            uint8_t need = runlock_[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                // Escape quarantine (if any) and promote to MRU
                runlock_[set][way] = 0;
                rrpv_set(set, way, 0);
            }
            // Train Mode A on reuse (A leaders only)
            if (LEADER_A(set)) {
                uint32_t slot = LEADER_SLOT(set);
                hawk_used[slot][way] = 1;
                leaderA_hits_epoch++;
                sat_dec_i8(bandit_score[set]); // bias toward A when A leaders hit
            } else if (LEADER_B(set)) {
                leaderB_hits_epoch++;
                sat_inc_i8(bandit_score[set]); // bias toward B when B leaders hit
            }
        }
        periodic_housekeeping();
        return;
    }

    // Miss/fill path
    hitcnt[set][way] = 0;

    // Never bypass writebacks; insert conservatively
    if (is_writeback(type)) {
        runlock_[set][way] = 0;
        rrpv_set(set, way, INSERT_COLD_DEPTH);
        periodic_housekeeping();
        return;
    }

    // Prefetch fills are quarantined at tail
    if (is_prefetch(type)) {
        runlock_[set][way] = 1;
        rrpv_set(set, way, maxRRPV);
        // Train SHCT in leader-A sets for prefetch
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            uint16_t sig = pc_sig12(PC);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = 1;
        }
        periodic_housekeeping();
        return;
    }

    // Demand fills: choose insertion via Mode B or Mode A
    if (useB) {
        uint32_t pidx = pc_index(PC);
        // Stream quarantine with optional shallow tail if PC is hot and not cold
        if (stream) {
            runlock_[set][way] = 1;
            uint8_t depth = maxRRPV;
            if ((pc_use4[pidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pidx] < 2))
                depth = 5; // shallow tail for short-reuse streams (e.g., zeusmp)
            rrpv_set(set, way, depth);
        } else {
            runlock_[set][way] = 0;
            // 2Q-LFU admission: hot PCs warm, cold PCs deep
            if (pc_cold2[pidx] >= 2) {
                rrpv_set(set, way, INSERT_COLD_DEPTH);
            } else if (pc_use4[pidx] >= PC_USE_HOT_THRESH) {
                rrpv_set(set, way, INSERT_WARM_DEPTH);
            } else {
                rrpv_set(set, way, INSERT_COLD_DEPTH);
            }
        }
    } else {
        // Mode A: Hawkeye-like SHiP insertion using tiny SHCT
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t ctr = shct_demand[idx];
        runlock_[set][way] = 0;
        if (ctr >= (SHCT_MAX / 2)) rrpv_set(set, way, INSERT_WARM_DEPTH);
        else                      rrpv_set(set, way, INSERT_COLD_DEPTH);

        // Train per-line signature only for leader-A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = 0;
        }
    }

    // Positive training for Mode A on fills that later see reuse happens on hit above.
    periodic_housekeeping();
}

// ---------------- Stats (silent) ---------------------------------
void PrintStats_Heartbeat() {}
void PrintStats() {}