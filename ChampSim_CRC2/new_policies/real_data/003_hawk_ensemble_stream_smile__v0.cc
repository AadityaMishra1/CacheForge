#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 leaders total) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for storage

// ---------------- Tunables (balanced for lbm/mcf/gcc/omnetpp/astar) -------
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP window (0..7)
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 steps before stream quarantine
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream: escape quarantine on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote quarantined line on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // decay period for TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 4096; // epoch for selector update
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers enable Mode B only if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set enable threshold

// ---------------- Per-line metadata (packed conceptually) -----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b  => 6 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Selector state -----------------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] local bias toward Mode B for followers
static uint8_t GSEL = 0;               // global gate (0..31)

// ---------------- Mode A (Hawkeye-like via tiny SHCT) ---------------------
// Two 2K-entry x 5-bit SHCTs: demand and prefetch
#define SHCT_BITS 11
#define SHCT_SIZE (1u << SHCT_BITS) // 2048
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Per-line signature/flags stored only for 64 leader-A sets (64x16 entries)
static uint16_t hawk_sig[64][LLC_WAYS];    // 12-bit effective in 16-bit cell
static uint8_t  hawk_used[64][LLC_WAYS];   // 0/1 seen reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS];// 0/1 filled by prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    // Simple xorshift-based 12-bit signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B (Stream-SMILE PC tables) -------------------------
// 512-entry PC tables (TinyLFU + coldness + last lineID + stream conf)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3)

// ---------------- Helpers -------------------------------------------------
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
    if (forward) { if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++; }
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;              // Mode B on leader-B sets
    if (LEADER_A(set)) return false;             // Mode A on leader-A sets
    // Followers: enable Mode B only if both global and local scores suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Periodic housekeeping -----------------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;

    // Decay TinyLFU and PC coldness lazily (power-of-two period)
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i] >>= 1;
        }
    }

    // Selector epoch update
    if ((op_count % BANDIT_EPOCH) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) { if (GSEL < GSEL_MAX) GSEL++; }
        else if (leaderA_hits_epoch > leaderB_hits_epoch) { if (GSEL > 0) GSEL--; }
        leaderA_hits_epoch = leaderB_hits_epoch = 0;
    }
}

// ---------------- Initialization ------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
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
    GSEL = 0; op_count = 0; leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Victim selection (RRIP with invalid preference) ----------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Select any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes to ensure termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback (should be unreachable)
    return 0;
}

// ---------------- API: GetVictimInSet -------------------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

// ---------------- API: UpdateReplacementState ------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    periodic_housekeeping();

    // Never bypass/act on writebacks
    if (is_writeback(type)) return;

    // Demand/prefetch activity updates (PC tables)
    uint32_t pcidx = pc_index(PC);
    bool stream_now = detect_and_update_stream(PC, paddr);

    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pcidx]);
        if (pc_cold2[pcidx] > 0) sat_dec_u2(pc_cold2[pcidx]);
    } else if (is_prefetch(type)) {
        // prefetch references do not raise hotness aggressively
        if (pc_use4[pcidx] > 0) sat_dec_u4(pc_use4[pcidx]); // slight downweight
    }

    // ---------------- HIT path ----------------
    if (hit) {
        // Train leader sets and selectors on hits
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
            // Slight positive reinforcement for SHCT on hit
            uint16_t sig = hawk_sig[slot][way] & 0x0FFFu;
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
            else                         shct_inc(shct_demand[idx]);
            if (is_demand(type)) leaderA_hits_epoch++;
            sat_dec_i8(bandit_score[set]); // bias toward Mode A
        } else if (LEADER_B(set)) {
            if (is_demand(type)) leaderB_hits_epoch++;
            sat_inc_i8(bandit_score[set]); // bias toward Mode B
        }

        // Stream quarantine: demote on touch; escape at configured hit count
        if (stream_lock[set][way]) {
            if (STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= (HITS_PROMOTE_STR - 1)) {
                    stream_lock[set][way] = 0;         // escape quarantine
                    rrpv_set(set, way, 0);             // MRU
                }
            }
            return; // no further promotion while quarantined
        }

        // Non-stream: multi-hit gating
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= (HITS_PROMOTE_NS - 1)) {
                rrpv_set(set, way, 0); // MRU on 2nd demand hit
            } else {
                rrpv_promote(set, way); // gentle promotion on 1st hit
            }
        } else {
            // Prefetch hit: do not promote aggressively
            if (rrpv[set][way] > 0) rrpv[set][way]--; // slight nudge
        }
        return;
    }

    // ---------------- MISS/FILL path ----------------
    // Train SHCT using the evicted line (only for leader-A sets)
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t vsig = hawk_sig[slot][way] & 0x0FFFu;
        uint32_t idx = shct_idx(vsig);
        if (hawk_is_pref[slot][way]) {
            if (hawk_used[slot][way] == 0) shct_dec(shct_prefetch[idx]);
            else                            shct_inc(shct_prefetch[idx]);
        } else {
            if (hawk_used[slot][way] == 0) shct_dec(shct_demand[idx]);
            else                            shct_inc(shct_demand[idx]);
        }
    }

    // Decide mode for followers
    bool use_modeB = modeB_enabled(set);

    // Default insertion decisions
    uint8_t ins_rrip = INSERT_COLD_DEPTH;
    uint8_t s_lock   = 0;

    if (use_modeB) {
        // Mode B: Stream-SMILE
        if (is_prefetch(type)) {
            ins_rrip = maxRRPV; // quarantine all prefetches
            s_lock   = 1;
        } else {
            // Demand
            bool hot_pc  = (pc_use4[pcidx] >= PC_USE_HOT_THRESH);
            bool cold_pc = (pc_cold2[pcidx] >= 2);

            if (stream_now) {
                ins_rrip = hot_pc ? 5 : maxRRPV; // hot streams get slightly shallower tail
                s_lock   = 1;
            } else {
                // Non-stream: adaptive insertion by PC hot/cold
                if (hot_pc && !cold_pc) ins_rrip = INSERT_WARM_DEPTH;
                else                    ins_rrip = INSERT_COLD_DEPTH;
                s_lock = 0;
            }
            // Penalize PCs on misses if they keep missing
            sat_inc_u2(pc_cold2[pcidx]);
        }
    } else {
        // Mode A: Hawkeye-like SHCT-based insertion
        if (is_prefetch(type)) {
            ins_rrip = maxRRPV; // prefetch quarantine
            s_lock   = 1;
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t  ctr = shct_demand[idx];
            if (ctr >= 16) ins_rrip = INSERT_WARM_DEPTH; // friendly PC
            else           ins_rrip = INSERT_COLD_DEPTH; // averse PC
            s_lock = 0;
        }
    }

    // Perform insertion
    rrpv_set(set, way, ins_rrip);
    stream_lock[set][way] = s_lock;
    hitcnt[set][way] = 0;

    // Record leader-A per-line signature for training
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

// ---------------- Stats hooks (kept minimal) -------------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}