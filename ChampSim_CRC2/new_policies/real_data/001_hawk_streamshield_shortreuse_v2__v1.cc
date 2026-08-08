#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types aligned to ChampSim CRC2
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

// ---------------- Tunables (retuned for lbm/zeusmp balance) -------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // followers enable B at >=2
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15)

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector ------------------------
static int8_t bandit_score[LLC_SETS]; // followers use Mode B only if >= MODEB_ENABLE_THRESH

// ---------------- Mode A (Hawkeye-like) --------------------------
// Tiny SHCTs (demand & prefetch), trained on sampled leader sets
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature (lower 12 bits of PC) + prefetched flag per line for training
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // 12b effective (stored in 16b)
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Mode B (StreamShield + TinyLFU) ----------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

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
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
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
    // 4) Fallback: pick the way with largest RRPV
    uint32_t v = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; v = w; }
    }
    return v;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update replacement state -----------------------
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
    // Mode selection & stream detection
    bool mB = modeB_enabled(set);
    bool is_stream_now = detect_and_update_stream(PC, paddr);
    uint32_t pcidx = pc_index(PC);

    // Bandit learning (leader-only)
    if (LEADER_B(set)) {
        // If Mode B sees a stream at fill time frequently, it's likely beneficial
        if (!hit && is_demand(type) && is_stream_now) sat_inc_i8(bandit_score[set]);
    }

    // Training for TinyLFU usefulness on demand hits
    if (hit && is_demand(type)) {
        sat_inc_u4(pc_use4[pcidx]);
    }

    // --------------------- HIT path ---------------------
    if (hit) {
        // Hawkeye train on hit (leader sets only)
        if (SEL_SAMPLED(set)) {
            uint16_t sig = hawk_signatures[set][way] & 0x0FFFu;
            if (hawk_prefetched[set][way]) shct_inc(shct_prefetch[shct_idx(sig)]);
            else                           shct_inc(shct_demand[shct_idx(sig)]);
        }

        // Demotion for quarantined lines and gated promotion
        if (is_demand(type)) {
            if (stream_lock[set][way]) {
                if (STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= (HITS_PROMOTE_STR - 1)) {
                    // Escape quarantine on short reuse
                    stream_lock[set][way] = 0;
                    rrpv[set][way] = 0; // MRU
                }
                // Leader B penalty if quarantine was too strict
                if (LEADER_B(set) && hitcnt[set][way] == 1) {
                    sat_dec_i8(bandit_score[set]);
                }
            } else {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= (HITS_PROMOTE_NS - 1)) {
                    rrpv[set][way] = 0; // MRU
                } else {
                    // gentle nudge toward MRU, but not full promotion
                    if (rrpv[set][way] > INSERT_WARM_DEPTH) rrpv[set][way] = INSERT_WARM_DEPTH;
                }
            }
        } else if (type == ACCESS_PREFETCH) {
            // Prefetch hits do not promote; keep near tail
            if (rrpv[set][way] > (maxRRPV - 1)) rrpv[set][way] = maxRRPV - 1;
        } else if (type == ACCESS_WRITEBACK) {
            // Do nothing on writeback hit
        }
        return;
    }

    // --------------------- MISS/INSERT path ---------------------
    // Train Hawkeye on victim deadness (leader sets only)
    if (SEL_SAMPLED(set)) {
        uint16_t vsig = hawk_signatures[set][way] & 0x0FFFu;
        // If victim never had a demand hit, it's dead; train down
        if (hitcnt[set][way] == 0) {
            if (hawk_prefetched[set][way]) shct_dec(shct_prefetch[shct_idx(vsig)]);
            else                           shct_dec(shct_demand[shct_idx(vsig)]);
        }
    }

    // Reset per-line metadata on insertion
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    // Default insertion depths
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    bool    set_stream_lock = false;

    // Prefetch: strict quarantine at tail
    if (type == ACCESS_PREFETCH) {
        ins_rrpv = maxRRPV;
        set_stream_lock = true;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass writebacks; insert modestly
        ins_rrpv = INSERT_WARM_DEPTH;
    } else {
        // Demand miss
        if (mB) {
            // Mode B: StreamShield + TinyLFU
            if (is_stream_now) {
                ins_rrpv = maxRRPV;          // hard tail
                set_stream_lock = true;      // quarantine and never early promote
            } else {
                bool pc_hot = (pc_use4[pcidx] >= PC_USE_HOT_THRESH);
                ins_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        } else {
            // Mode A: Hawkeye-like guided insertion
            uint32_t sig = (uint32_t)(PC & 0x0FFFu);
            bool pred_hot = false;
            if (SEL_SAMPLED(set)) {
                // Leaders use the table of their own type
                pred_hot = (shct_demand[shct_idx(sig)] > (SHCT_MAX >> 1));
            } else {
                // Followers use demand table as default
                pred_hot = (shct_demand[shct_idx(sig)] > (SHCT_MAX >> 1));
            }
            ins_rrpv = pred_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    // Apply insertion
    rrpv[set][way] = std::min<uint8_t>(ins_rrpv, maxRRPV);
    stream_lock[set][way] = set_stream_lock ? 1 : 0;

    // Record signature for Hawkeye training (leaders store; followers may store but only leaders are used)
    hawk_signatures[set][way] = (uint16_t)(PC & 0x0FFFu);
    hawk_prefetched[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;

    // Leader B reinforcement for strong streaming (no reuse expected)
    if (LEADER_B(set) && is_demand(type) && is_stream_now) {
        sat_inc_i8(bandit_score[set]);
    }
}

// ---------------- Print hooks (kept blank) -----------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}