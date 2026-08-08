#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim-CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables (adjust to balance lbm vs mcf/gcc) ----
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd+ demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream MRU at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15)
static constexpr uint8_t GSEL_MAX            = 31; // global selector range
static constexpr uint8_t GSEL_THRES          = 6;  // need clear B advantage to switch followers

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// Leader-only per-line hit counter (2b) for selector + SHCT training
static uint8_t leader_hits2[LLC_SETS][LLC_WAYS]; // only used when SEL_SAMPLED(set)

// ---------------- Global selector (leader set dueling) -----------
static uint8_t GSEL = 0; // 0=prefer Mode A; >=GSEL_THRES -> enable Mode B on followers

static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}

// ---------------- Mode A (Hawkeye-like SHCT) ---------------------
// Tiny SHCTs (demand & prefetch), trained in sampled leader sets
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature (lower 12 bits of PC) + prefetched flag for leader training
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // 12b effective (stored in 16b)
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Mode B (RunGuard + TinyLFU) --------------------
// 256-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15) TinyLFU
static uint8_t  pc_phase2[PC_TBL_SIZE];      // 2b hit phase for multi-hit gating (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            leader_hits2[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_phase2, 0, sizeof(pc_phase2));
    GSEL = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Age bounded passes (ensure termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: return the way with the highest RRPV (ties -> highest index)
    uint32_t best = 0, best_rrpv = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best_rrpv) { best_rrpv = rrpv[set][w]; best = w; }
    }
    return best;
}

// ---------------- Insertion depths -------------------------------
static inline uint8_t insert_depth_modeA(uint64_t PC, uint32_t type) {
    // Hawkeye-like: friendly PCs high, averse low
    if (type == ACCESS_PREFETCH) return maxRRPV;
    uint32_t idx = shct_idx(PC);
    bool friendly = is_demand(type) ? (shct_demand[idx] > (SHCT_MAX/2)) : (shct_prefetch[idx] > (SHCT_MAX/2));
    return friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
}

static inline uint8_t insert_depth_modeB(uint64_t PC, uint64_t paddr, uint32_t type, bool& is_stream_pc) {
    if (type == ACCESS_PREFETCH) { is_stream_pc = true; return maxRRPV; } // quarantine prefetches
    is_stream_pc = detect_and_update_stream(PC, paddr);
    if (is_stream_pc) return maxRRPV; // stream quarantine
    // Non-stream: TinyLFU warm/cold gate
    uint32_t idx = pc_index(PC);
    bool hot = (pc_use4[idx] >= PC_USE_HOT_THRESH);
    return hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
}

// ---------------- External interface -----------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Never act on writebacks
    if (type == ACCESS_WRITEBACK) return;

    // PC tables update on demand references
    if (is_demand(type)) {
        // Update stream detector (for next time) and TinyLFU/usefulness on demand hits
        (void)detect_and_update_stream(PC, paddr);
    }

    // On hit: multi-hit promotion gating + stream demotion-on-touch
    if (hit) {
        // Leader training: count hits for sampled sets (for selector + SHCT training)
        if (SEL_SAMPLED(set)) {
            if (leader_hits2[set][way] < 3) leader_hits2[set][way]++; // saturate to 3
        }

        // Update per-PC phase for multi-hit gating (demand only)
        if (is_demand(type)) {
            uint32_t pidx = pc_index(PC);
            if (pc_phase2[pidx] < 3) pc_phase2[pidx]++; // age toward "seen multiple hits"
            if (pc_use4[pidx] < 15) pc_use4[pidx]++;    // TinyLFU usefulness on demand hits
        }

        // Stream-aware demote on touch unless multi-hit confirms
        bool is_stream_pc = (pc_stream_conf[pc_index(PC)] >= STREAM_CONF_THRESH);
        uint8_t need_hits = is_stream_pc ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
        bool promote = false;
        if (is_demand(type)) {
            promote = (pc_phase2[pc_index(PC)] >= (need_hits - 1));
        } else {
            promote = false; // prefetch hits never promote on first touch
        }

        if (promote) rrpv[set][way] = 0;
        else {
            if (STREAM_DEMOTE_TOUCH && is_stream_pc && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }
        return;
    }

    // Miss/Fill path: choose victim was already returned; train leaders and insert
    // Leader training on victim eviction outcome
    if (SEL_SAMPLED(set)) {
        uint8_t h = leader_hits2[set][way];
        bool good = (h >= 2);
        if (LEADER_B(set)) {
            // Mode B leader
            if (good) { if (GSEL < GSEL_MAX) GSEL++; }
            else      { if (GSEL > 0)       GSEL--; }
        } else if (LEADER_A(set)) {
            // Mode A leader
            if (good) { if (GSEL > 0)       GSEL--; }
            else      { if (GSEL < GSEL_MAX) GSEL++; }
        }
        // SHCT training (Hawkeye-like): credit if reused before eviction
        uint16_t sig = hawk_signatures[set][way] & (SHCT_SIZE - 1);
        if (hawk_prefetched[set][way]) {
            if (good) shct_inc(shct_prefetch[sig]);
            else      shct_dec(shct_prefetch[sig]);
        } else {
            if (good) shct_inc(shct_demand[sig]);
            else      shct_dec(shct_demand[sig]);
        }
        // Decay PC usefulness for Mode B when line died young
        if (!good) {
            uint32_t pidx = pc_index((uint64_t)sig);
            sat_dec_u4(pc_use4[pidx]);
            if (pc_phase2[pidx] > 0) pc_phase2[pidx]--; // decay hit phase slightly
        }
    }

    // Record signature for future training on sampled sets
    if (SEL_SAMPLED(set)) {
        hawk_signatures[set][way] = (uint16_t)(PC & 0xFFFu);
        hawk_prefetched[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        leader_hits2[set][way] = 0; // reset
    }

    // Choose mode and insertion depth
    bool modeB = use_modeB(set);
    bool is_stream_pc = false;
    uint8_t depth = modeB ? insert_depth_modeB(PC, paddr, type, is_stream_pc)
                          : insert_depth_modeA(PC, type);

    // Apply insertion depth (hard-tail quarantine for streams/prefetch)
    rrpv[set][way] = std::min<uint8_t>(depth, maxRRPV);
}

void PrintStats_Heartbeat() {}
void PrintStats() {}