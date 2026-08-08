#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

// ChampSim CRC2 constants
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

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 sets: low 6 bits == next 6 bits
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
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;     // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;     // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;     // TinyLFU hot PCs (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;  // periodic decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;  // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;    // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;    // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;     // per-slot enable threshold

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b (also used to quarantine prefetch)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set tiny reuse-distance bias ---------------
static int8_t rd_bias[LLC_SETS]; // [-8..7], >0 => insert warmer; <0 => insert colder

// ---------------- Selector: per-slot bandit + global gate --------
static int8_t  bandit_score_slot[64];      // [-8..7]
static uint16_t leader_hits_A_slot[64];    // epoch counters
static uint16_t leader_hits_B_slot[64];
static uint8_t GSEL = 0;                   // global preference 0..31

// ---------------- Mode A (Hawkeye-like SHiP) ---------------------
// Tiny SHCT (1024 x 5b)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS)
#define SHCT_MAX 31
static uint8_t shct[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // compact 12b signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature + training flags per A-leader slot
static uint16_t hawk_sig_slot[64][LLC_WAYS];   // 12b effective
static uint8_t  hawk_used_slot[64][LLC_WAYS];  // reuse seen
static uint8_t  hawk_pref_slot[64][LLC_WAYS];  // filled by prefetch

// ---------------- Mode B (TinyRD-StreamGuard) --------------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
// 64B lines -> 10b line-id per PC
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_dead2[PC_TBL_SIZE];       // 2b deadness (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

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
    if (LEADER_B(set)) return true;       // force Mode B on B leaders
    if (LEADER_A(set)) return false;      // force Mode A on A leaders
    uint32_t slot = LEADER_SLOT(set);
    return (GSEL >= GSEL_THRES) && (bandit_score_slot[slot] >= MODEB_ENABLE_THRESH);
}

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}

// Periodic housekeeping
static uint32_t op_count = 0;
static inline void periodic_housekeeping() {
    op_count++;
    // Decay TinyLFU and deadness (lazy)
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_dead2[i] > 0) pc_dead2[i]--;
        }
    }
    // Bandit epoch update
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        // Global preference
        uint32_t totalA = 0, totalB = 0;
        for (uint32_t s = 0; s < 64; s++) { totalA += leader_hits_A_slot[s]; totalB += leader_hits_B_slot[s]; }
        if (totalB > totalA) { if (GSEL < GSEL_MAX) GSEL++; }
        else if (totalA > totalB) { if (GSEL > 0) GSEL--; }
        // Per-slot bandit scores and reset
        for (uint32_t s = 0; s < 64; s++) {
            if (leader_hits_B_slot[s] > leader_hits_A_slot[s]) sat_inc_i8(bandit_score_slot[s]);
            else if (leader_hits_A_slot[s] > leader_hits_B_slot[s]) sat_dec_i8(bandit_score_slot[s]);
            leader_hits_A_slot[s] = leader_hits_B_slot[s] = 0;
        }
    }
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        rd_bias[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct, 0, sizeof(shct));
    std::memset(hawk_sig_slot, 0, sizeof(hawk_sig_slot));
    std::memset(hawk_used_slot, 0, sizeof(hawk_used_slot));
    std::memset(hawk_pref_slot, 0, sizeof(hawk_pref_slot));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_dead2, 0, sizeof(pc_dead2));

    std::memset(bandit_score_slot, 0, sizeof(bandit_score_slot));
    std::memset(leader_hits_A_slot, 0, sizeof(leader_hits_A_slot));
    std::memset(leader_hits_B_slot, 0, sizeof(leader_hits_B_slot));
    GSEL = 0;
    op_count = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // 1) Prefer invalid way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find a line at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging until a victim appears
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback (should not happen)
    return 0;
}

// ---------------- Update state on hit/fill -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    periodic_housekeeping();

    const bool demand = is_demand(type);
    const bool pref   = is_prefetch(type);
    const bool wb     = is_writeback(type);
    uint32_t slot = LEADER_SLOT(set);

    // Leader hit accounting for bandit (only count hits)
    if (hit) {
        if (LEADER_A(set)) leader_hits_A_slot[slot]++;
        if (LEADER_B(set)) leader_hits_B_slot[slot]++;
    }

    // Ignore replacement tuning on writebacks beyond minimal metadata updates
    if (wb) {
        if (!hit) {
            // Writeback fills are not bypassed; insert low priority
            rrpv_set(set, way, INSERT_COLD_DEPTH);
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 0;
        } else {
            // On writeback hit, slight aging resistance
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
        return;
    }

    // Mode selection
    const bool use_modeB = modeB_enabled(set);

    if (hit) {
        // Update PC tables (utility up, deadness down)
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
        sat_dec_u2(pc_dead2[pidx]);

        // TinyRD sketch: strengthen/weakening bias based on where we hit
        uint8_t r = rrpv[set][way];
        if (r <= 2) sat_inc_i8(rd_bias[set]);    // hit near MRU -> more warm inserts
        else if (r >= 5) sat_dec_i8(rd_bias[set]); // hit deep -> cooler inserts

        // Multi-hit promotion gating
        if (demand) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        }

        // Stream quarantine demote-on-touch
        if (stream_lock[set][way]) {
            // Demote on first touch to keep scans cold
            if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }

        // Promote only on sufficient demand hits
        if (demand) {
            uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] + 1 >= need) { // +1 because hitcnt counted previous hits
                rrpv_set(set, way, 0);
                stream_lock[set][way] = 0; // escape quarantine
            }
        }

        // Mode A training: mark reuse for A-leader sets
        if (LEADER_A(set)) {
            hawk_used_slot[slot][way] = 1;
        }

        return;
    }

    // -------- Miss path (eviction training then insertion) --------

    // Train Mode A SHCT on eviction from A-leader sets
    if (LEADER_A(set)) {
        // The victim's recorded signature/flags are in the A-leader slot table
        uint16_t old_sig = hawk_sig_slot[slot][way];
        uint8_t  saw_reuse = hawk_used_slot[slot][way];
        // Update predictor
        uint32_t idx = shct_idx(old_sig);
        if (saw_reuse) shct_inc(shct[idx]);
        else           shct_dec(shct[idx]);
        // Clear for reuse on new fill (set below)
        hawk_used_slot[slot][way] = 0;
        hawk_pref_slot[slot][way] = 0;
        hawk_sig_slot[slot][way]  = 0;
    }

    // Decide insertion depth and quarantine
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t lock_stream = 0;

    if (use_modeB) {
        const bool is_stream = detect_and_update_stream(PC, paddr);
        uint32_t pidx = pc_index(PC);
        // Update PC coldness on miss
        sat_inc_u2(pc_dead2[pidx]);
        // Base depth: hot PCs insert warmer; cold PCs insert colder
        bool lfu_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        ins_depth = lfu_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

        // Adjust by per-set RD bias
        if (rd_bias[set] > 0 && ins_depth > 0) ins_depth--;
        else if (rd_bias[set] < 0 && ins_depth < maxRRPV) ins_depth++;

        // Prefetches and streams are quarantined at tail (bypass-like)
        if (pref || is_stream) {
            ins_depth = maxRRPV; // hard tail
            lock_stream = 1;     // demote-on-touch; needs 2nd demand hit to escape
        }
    } else {
        // Mode A (Hawkeye-like SHiP): insert warm for friendly sigs
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        bool friendly = (shct[idx] > (SHCT_MAX >> 1)); // >15
        ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    }

    // Insert the new line
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock_stream || (pref ? 1 : 0);

    // Mode A: record signature and prefetch flag for A-leader sets
    if (LEADER_A(set)) {
        hawk_sig_slot[slot][way]  = pc_sig12(PC);
        hawk_pref_slot[slot][way] = pref ? 1 : 0;
        hawk_used_slot[slot][way] = 0;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}