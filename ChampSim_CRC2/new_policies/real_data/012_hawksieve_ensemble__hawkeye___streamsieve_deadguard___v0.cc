#include <cstdint>
#include <cstring>
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

// ---------------- Leader-set sampling (64 leader sets) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample if low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // force Mode A
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // force Mode B
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 among leaders

// ---------------- Tunables -------------------------------------------------
// RRIP window (3b)
static constexpr uint8_t maxRRPV             = 7;
// Mode A insertion depths (Hawkeye-like)
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t FRIENDLY_THRESH     = 16; // SHCT threshold (0..31)
// Mode B stream detection and handling
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // 2 forward steps (+1/+2)
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined lines on touches
// Multi-hit promotion gates
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream promote on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape on 2nd demand hit
// TinyLFU and coldness thresholds
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot if >=
static constexpr uint8_t PC_COLD_HIGH        = 2;  // cold if >=
// Selector tuning
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048; // PC tables decay
static constexpr uint32_t BANDIT_EPOCH       = 4096; // selector epoch
static constexpr uint8_t  GSEL_MAX           = 31;
static constexpr uint8_t  GSEL_THRES         = 16;   // followers prefer Mode B if GSEL >= THRES
static constexpr int8_t   MODEB_ENABLE_THRESH= 2;    // local bias

// ---------------- Per-line metadata (conceptual packing: 3+2+1 bits) -------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];        // 3b
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 2b (0..3)
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1b run quarantine

// ---------------- Per-set + global selectors --------------------------------
static int8_t  bandit_score[LLC_SETS]; // local bias toward Mode B (saturating -8..7)
static uint8_t GSEL = 0;               // global gate (0..31)

// ---------------- Mode A: Hawkeye-like (tiny SHCTs) -------------------------
// Two SHCTs (demand and prefetch), 2K entries, 5-bit counters
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1u << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Per-line training only in leader-A sets (64 x 16 = 1024 lines)
static uint16_t hawk_sig[64][LLC_WAYS];   // 12b effective stored in 16b
static uint8_t  hawk_used[64][LLC_WAYS];  // 1b reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1b filled-by-prefetch

// ---------------- Mode B: StreamSieve + DeadGuard ---------------------------
// 512-entry per-PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines id (10b)

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// ---------------- Helpers ---------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline uint16_t pc_sig12(uint64_t pc) {
    // low-discrepancy 12b signature from PC
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

// Forward-run detector for +1/+2 strides with confidence
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
    if (LEADER_B(set)) return true;   // force Mode B on B leaders
    if (LEADER_A(set)) return false;  // force Mode A on A leaders
    // Followers: use Mode B only if both global and local selectors agree
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Housekeeping ---------------------------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;

    // Decay TinyLFU and coldness periodically
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1; // halve
            if (pc_cold2[i] > 0) pc_cold2[i] >>= 1;
        }
        // Lightly decay per-set bandit scores toward zero
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (bandit_score[s] > 0) sat_dec_i8(bandit_score[s]);
            else if (bandit_score[s] < 0) sat_inc_i8(bandit_score[s]); // toward zero
        }
    }

    // Update global gate each epoch using leader-set hit totals
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) {
            if (GSEL < GSEL_MAX) GSEL++;
        } else if (leaderB_hits_epoch < leaderA_hits_epoch) {
            if (GSEL > 0) GSEL--;
        }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// ---------------- Initialization -------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;   // start cold
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig,    0, sizeof(hawk_sig));
    std::memset(hawk_used,   0, sizeof(hawk_used));
    std::memset(hawk_is_pref,0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4,        0, sizeof(pc_use4));
    std::memset(pc_cold2,       0, sizeof(pc_cold2));
    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
    GSEL = 0;
}

// ---------------- Victim selection (bounded RRIP) ---------------------------
static inline uint32_t rrip_pick_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Any at maxRRPV?
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Age with bounded passes (ensures termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: pick the highest RRPV
    uint32_t victim = 0, best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    uint32_t victim = rrip_pick_victim(set, current_set);

    // Mode A negative training on eviction in leader-A sets
    if (SEL_SAMPLED(set) && current_set[victim].valid) {
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            // If the evicted line did not see reuse, penalize its fill PC
            if (hawk_used[slot][victim] == 0) {
                uint16_t sig = hawk_sig[slot][victim];
                uint32_t idx = shct_idx(sig);
                if (hawk_is_pref[slot][victim]) shct_dec(shct_prefetch[idx]);
                else                            shct_dec(shct_demand[idx]);
            }
            // Clear for safety (not strictly required)
            hawk_used[slot][victim] = 0;
            hawk_is_pref[slot][victim] = 0;
            hawk_sig[slot][victim] = 0;
        }
    }
    return victim;
}

// ---------------- Update replacement state ----------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    periodic_housekeeping();

    // Ignore writebacks for replacement policy (never bypass on WB)
    if (is_writeback(type)) return;

    // Demand-side PC stats (TinyLFU and coldness)
    uint32_t pcidx = pc_index(PC);
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pcidx]);
        // coldness: increase on misses below, decay in housekeeping, decrease on hits
        if (hit) { if (pc_cold2[pcidx] > 0) pc_cold2[pcidx]--; }
    }

    // Mode decision
    bool use_modeB = modeB_enabled(set);

    // ---------------- On hit: promotion with multi-hit gates ----------------
    if (hit) {
        // Leader hit accounting for selector
        if (LEADER_A(set)) leaderA_hits_epoch++;
        else if (LEADER_B(set)) leaderB_hits_epoch++;

        // Mode A training: increment SHCT for lines that showed reuse in leader-A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
            uint16_t sig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
            else                         shct_inc(shct_demand[idx]);
        }

        // Stream quarantine handling
        if (stream_lock[set][way]) {
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                // Escape quarantine on 2nd demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, 0); // MRU
                } else {
                    // Demote-on-touch to hasten eviction if it stays single-use
                    for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);
                }
            }
            // Prefetch hits do not promote
            return;
        }

        // Non-stream: multi-hit MRU promotion only on 2nd+ demand hit
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                rrpv_set(set, way, 0); // MRU
            } else {
                // gentle bump for exploration
                rrpv_promote(set, way);
            }
        }
        return;
    }

    // ---------------- On miss: decide insertion (no WB here) ---------------
    hitcnt[set][way] = 0;

    // Prefetches always quarantine at tail
    if (is_prefetch(type)) {
        rrpv_set(set, way, maxRRPV);
        stream_lock[set][way] = 1; // quarantine until 2nd demand hit
        // Train Mode A metadata in leader-A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_is_pref[slot][way] = 1;
            hawk_used[slot][way] = 0;
        }
        return;
    }

    // Demand miss insertion
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    bool mark_stream = false;

    if (use_modeB) {
        // StreamSIEVE
        bool is_stream = detect_and_update_stream(PC, paddr);
        if (is_stream) {
            ins_depth = maxRRPV;     // hard tail (bypass via cold insert)
            mark_stream = true;
            if (!LEADER_A(set)) sat_inc_i8(bandit_score[set]); // stream-heavy => Mode B useful
        } else {
            // DeadGuard: TinyLFU + coldness
            bool hot_pc  = (pc_use4[pcidx] >= PC_USE_HOT_THRESH);
            bool cold_pc = (pc_cold2[pcidx] >= PC_COLD_HIGH);
            if (hot_pc && !cold_pc) ins_depth = INSERT_WARM_DEPTH;
            else                    ins_depth = INSERT_COLD_DEPTH;
            if (!LEADER_A(set)) sat_dec_i8(bandit_score[set]); // locally quieter => reduce bias
        }
    } else {
        // Mode A (Hawkeye-like): PC-friendly vs averse
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t conf = shct_demand[idx];
        ins_depth = (conf >= FRIENDLY_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

        // Only leaders carry per-line training metadata
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_is_pref[slot][way] = 0;
            hawk_used[slot][way] = 0;
        }
    }

    rrpv_set(set, way, ins_depth);
    stream_lock[set][way] = mark_stream ? 1 : 0;

    // Demand miss increases PC coldness
    if (is_demand(type)) sat_inc_u2(pc_cold2[pcidx]);
}

// ---------------- Minimal stats hooks ---------------------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}