#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type codes (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline bool is_pf(uint32_t type) { return type == ACCESS_PREFETCH; }
static inline bool is_wb(uint32_t type) { return type == ACCESS_WRITEBACK; }

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 leaders: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables (retuned) ------------------------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // 2nd demand hit -> MRU
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape on 2nd demand hit
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // conservative
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 5;  // TinyLFU hot threshold (0..15), slightly lower
// TinyLFU decay: every 16 accesses, decay one entry (ring)
static constexpr uint32_t PC_DECAY_PERIOD = 16;

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, seen_once:1b, stream_flag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t seen_once[LLC_SETS][LLC_WAYS];
static uint8_t stream_flag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector -------------------------
static int8_t bandit_score[LLC_SETS]; // followers enable Mode B only if >= MODEB_ENABLE_THRESH

// ---------------- Mode A (Hawkeye-like) ---------------------------
// Tiny SHCTs (demand & prefetch), trained only in leader sets
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-only per-line signature and pf-flag for training
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // store SHCT index (11b effective)
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Mode B (StreamShield + TinyLFU) -----------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b stored in 16b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_stream_toggle[PC_TBL_SIZE]; // 1b toggle
static uint32_t pc_decay_ptr = 0;
static uint64_t access_tick = 0;

// ---------------- Helpers -----------------------------------------
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

static inline void tiny_lfu_decay_tick() {
    access_tick++;
    if ((access_tick % PC_DECAY_PERIOD) == 0) {
        uint32_t i = pc_decay_ptr++ & (PC_TBL_SIZE - 1);
        if (pc_use4[i] > 0) pc_use4[i]--;
    }
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            seen_once[s][w] = 0;
            stream_flag[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_stream_toggle, 0, sizeof(pc_stream_toggle));
    pc_decay_ptr = 0;
    access_tick = 0;
}

// ---------------- Victim selection (RRIP with preference) ---------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) First pass: any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging until some reach maxRRPV (ensure termination)
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

// ---------------- API: Find victim --------------------------------
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

// ---------------- Train Hawkeye (leader sets only) -----------------
static inline void hawk_train_on_eviction(uint32_t set, uint32_t way) {
    if (!SEL_SAMPLED(set)) return;
    uint16_t sig = hawk_signatures[set][way];
    if (sig == 0) return; // ignore empty
    bool reused = (seen_once[set][way] != 0);
    if (hawk_prefetched[set][way]) {
        if (reused) shct_inc(shct_prefetch[sig]);
        else        shct_dec(shct_prefetch[sig]);
    } else {
        if (reused) shct_inc(shct_demand[sig]);
        else        shct_dec(shct_demand[sig]);
    }
}

// ---------------- API: Update replacement state -------------------
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
    // Advance TinyLFU decay
    if (!is_wb(type)) tiny_lfu_decay_tick();

    if (hit) {
        // Bandit: reward whichever mode owns leader set on demand hits
        if (is_demand(type)) {
            if (LEADER_B(set)) sat_inc_i8(bandit_score[set]);
            else if (LEADER_A(set)) sat_dec_i8(bandit_score[set]);
        }

        // Stream quarantine demotion on touch (unless escaping now)
        bool demand = is_demand(type);
        if (stream_flag[set][way]) {
            if (demand && seen_once[set][way] >= (HITS_PROMOTE_STR - 1)) {
                // Escape quarantine on 2nd demand hit
                stream_flag[set][way] = 0;
                rrpv[set][way] = 0; // MRU
            } else {
                // Demote quarantined lines on any touch
                for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++)
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            }
        }

        // Multi-hit gating: only demand hits count toward promotion
        if (demand) {
            if (seen_once[set][way] >= (HITS_PROMOTE_NS - 1)) {
                rrpv[set][way] = 0; // MRU
            } else {
                seen_once[set][way] = 1;
            }
            // Train TinyLFU usefulness for Mode B
            uint32_t pidx = pc_index(PC);
            sat_inc_u4(pc_use4[pidx]);
        }
        return;
    }

    // Miss path: we are installing a new line at (set, way).
    // Train Hawkeye (leader sets only) on the evicted line outcome before overwrite.
    hawk_train_on_eviction(set, way);

    // Reset line metadata for the new fill
    seen_once[set][way] = 0;
    stream_flag[set][way] = 0;
    hawk_prefetched[set][way] = is_pf(type) ? 1 : 0;
    hawk_signatures[set][way] = (uint16_t)shct_idx(PC);

    bool use_modeB = modeB_enabled(set);

    if (use_modeB) {
        // Mode B: StreamShield + TinyLFU
        bool demand = is_demand(type);
        uint32_t pidx = pc_index(PC);
        bool is_stream = false;
        if (!is_wb(type)) {
            // Detect stream only on non-writeback
            is_stream = detect_and_update_stream(PC, paddr);
        }

        if (is_pf(type)) {
            // Prefetch quarantine
            rrpv[set][way] = maxRRPV;
            stream_flag[set][way] = 1;
        } else if (demand && is_stream) {
            // Deterministic bypass: alternate hard-tail with one-shot set aging
            uint8_t tog = pc_stream_toggle[pidx];
            pc_stream_toggle[pidx] ^= 1u;
            rrpv[set][way] = maxRRPV; // hard tail
            stream_flag[set][way] = 1;
            if (tog) {
                // One-shot aging pulse to hasten eviction pressure for this streaming fill
                for (uint32_t w = 0; w < LLC_WAYS; w++) {
                    if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
                }
            }
        } else if (demand) {
            // Non-stream demand: TinyLFU-guided insertion
            if (pc_use4[pidx] >= PC_USE_HOT_THRESH) {
                rrpv[set][way] = INSERT_WARM_DEPTH;
            } else {
                rrpv[set][way] = INSERT_COLD_DEPTH;
            }
        } else if (is_wb(type)) {
            // Never bypass writebacks: modestly warm insert
            rrpv[set][way] = INSERT_WARM_DEPTH;
        } else {
            // Fallback
            rrpv[set][way] = INSERT_COLD_DEPTH;
        }
    } else {
        // Mode A: Hawkeye-like
        if (is_pf(type)) {
            rrpv[set][way] = maxRRPV; // prefetch quarantine
            stream_flag[set][way] = 1;
        } else if (is_demand(type)) {
            uint32_t idx = shct_idx(PC);
            uint8_t conf = shct_demand[idx];
            if (conf > (SHCT_MAX >> 1)) rrpv[set][way] = INSERT_WARM_DEPTH;
            else                        rrpv[set][way] = INSERT_COLD_DEPTH;
        } else if (is_wb(type)) {
            rrpv[set][way] = INSERT_WARM_DEPTH; // keep writebacks
        } else {
            rrpv[set][way] = INSERT_COLD_DEPTH;
        }
    }
}

// ---------------- Stats (remain blank) -----------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}