#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types (aligned to common configs)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (carefully balanced for HM-IPC) --------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU for warm lines
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // deep insertion for cold lines
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 line steps (2 consecutive)
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream MRU at 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape at 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;   // demote quarantined stream on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// periodic TinyLFU/coldness decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch length
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;  // followers allow Mode B if gate >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set bandit enable threshold
static constexpr uint8_t  SHCT_FRIENDLY_THRES = 16;  // Mode A friendly cutoff (0..31)

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_lock(runlock/quarantine):1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set and global selectors -------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B
static uint8_t GSEL = 0;               // global gate (0..31)

// ---------------- Mode A (Hawkeye-like SHCT) ---------------------
// Single 2K-entry, 5-bit counter SHCT trained in leader-A sets
#define SHCT_BITS 11
#define SHCT_SIZE (1u << SHCT_BITS) // 2048
#define SHCT_MAX 31
static uint8_t shct[SHCT_SIZE];

// Per-line signature and reuse flag stored only for leaders (64x16)
static uint16_t hawk_sig[64][LLC_WAYS]; // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];// 0/1: saw reuse in set before eviction

static inline uint16_t pc_sig12(uint64_t pc) {
    // light hash for 12b PC signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx_from_sig(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B PC predictors (stride + TinyLFU) --------
// 256-entry tables (tight storage)
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU usefulness (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b deadness bias (0..3)

// For PC deadness training on leaders: store creator pc_idx per line
static uint8_t leader_pcidx[64][LLC_WAYS];   // 8b index into PC_TBL

// ---------------- Helpers ----------------------------------------
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

// Forward-run detector (+1/+2 stride)
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
    // Followers: enable Mode B only if both global and local scores suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Periodic housekeeping --------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;

    // Decay TinyLFU and PC coldness (lazy, periodic)
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
        }
    }

    // Selector epoch updates
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        // Global gate moves toward the better leader
        if (leaderB_hits_epoch > leaderA_hits_epoch) {
            gsel_inc();
            // bias all sets toward Mode B slightly
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_inc_i8(bandit_score[s]);
        } else if (leaderA_hits_epoch > leaderB_hits_epoch) {
            gsel_dec();
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_dec_i8(bandit_score[s]);
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
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct, 0, sizeof(shct));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(leader_pcidx, 0, sizeof(leader_pcidx));
    op_count = 0;
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
    GSEL = 0;
}

// ---------------- Victim selection (RRIP with safe training) -----
static inline uint32_t rrip_pick_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV; age boundedly to ensure termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age once
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: return the way with largest RRPV (ties arbitrary)
    uint32_t best = 0, best_rrpv = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best_rrpv) { best_rrpv = rrpv[set][w]; best = w; }
    }
    return best;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    uint32_t way = rrip_pick_victim(set, current_set);

    // Train predictors only when evicting a valid line from leaders
    if (current_set[way].valid && SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);

        // Mode A SHCT: decrement if line died without reuse
        uint16_t sig = hawk_sig[slot][way];
        if (LEADER_A(set) && sig) {
            if (hawk_used[slot][way] == 0) {
                uint32_t idx = shct_idx_from_sig(sig);
                shct_dec(shct[idx]);
            }
            hawk_sig[slot][way]  = 0;
            hawk_used[slot][way] = 0;
        }

        // Mode B PC deadness: penalize creator if no reuse, otherwise reward slightly
        uint8_t pcidx = leader_pcidx[slot][way];
        if (hawk_used[slot][way] == 0) {
            if (pc_cold2[pcidx] < 3) pc_cold2[pcidx]++;
        } else {
            if (pc_cold2[pcidx] > 0) pc_cold2[pcidx]--;
        }
        leader_pcidx[slot][way] = 0;
    }

    return way;
}

// ---------------- Update state on hit/fill ------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // housekeeping (decay, selector epochs)
    periodic_housekeeping();

    // Ignore writebacks entirely
    if (type == ACCESS_WRITEBACK) return;

    // Update TinyLFU on all references by PC
    uint32_t pcidx = pc_index(PC);
    sat_inc_u4(pc_use4[pcidx]);

    // Leader accounting for selector (demand hits only)
    if (hit && is_demand(type)) {
        if (LEADER_A(set)) leaderA_hits_epoch++;
        else if (LEADER_B(set)) leaderB_hits_epoch++;
    }

    // Demand stream detector (helps lbm/milc/zeusmp)
    bool stream = false;
    if (is_demand(type)) {
        stream = detect_and_update_stream(PC, paddr);
    }

    if (hit) {
        // Track reuse for leader training
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1; // saw reuse before eviction
        }

        // Multi-hit gated promotion; never promote on first hit or on prefetch hits
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        if (stream_lock[set][way]) {
            // Quarantined stream: demote-on-touch; escape to MRU only on 2nd demand hit
            if (STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_STR) {
                rrpv_set(set, way, 0);  // MRU escape
                stream_lock[set][way] = 0;
            }
        } else {
            // Non-stream: MRU promote only on 2nd+ demand hit
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_NS) {
                rrpv_set(set, way, 0);
            }
        }
        return;
    }

    // Miss fill path (no bypass on writeback; prefetch tail; demand adaptive)
    uint8_t insert_depth = INSERT_COLD_DEPTH;
    uint8_t runlock_flag = 0; // 1 if quarantined stream

    bool use_modeB = modeB_enabled(set);

    if (is_prefetch(type)) {
        // Always quarantine prefetches at the hard tail
        insert_depth = maxRRPV;
        runlock_flag = 1;
    } else {
        if (use_modeB) {
            // Mode B: StreamSieve + TinyLFU/coldness
            if (stream) {
                insert_depth = maxRRPV; // hard tail quarantine
                runlock_flag = 1;
            } else {
                bool hot = (pc_use4[pcidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pcidx] <= 1);
                insert_depth = hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                runlock_flag = 0;
            }
        } else {
            // Mode A: Hawkeye-like SHCT insertion
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx_from_sig(sig);
            bool friendly = (shct[idx] >= SHCT_FRIENDLY_THRES);
            insert_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            runlock_flag = 0;
        }
    }

    // Perform insertion
    rrpv_set(set, way, std::min<uint8_t>(insert_depth, maxRRPV));
    hitcnt[set][way] = 0;
    stream_lock[set][way] = runlock_flag;

    // Record training metadata in leaders
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Mode A: remember signature for eviction-based SHCT training
        uint16_t sig = pc_sig12(PC);
        hawk_sig[slot][way]  = sig;
        hawk_used[slot][way] = 0;
        // Mode B: remember creator PC index for deadness training
        leader_pcidx[slot][way] = (uint8_t)pcidx;
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}