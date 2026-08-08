#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types per ChampSim-CRC2
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) { return (type == ACCESS_LOAD) || (type == ACCESS_RFO); }

// ---------------- Sampling and leader sets (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Select 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (SEL_SAMPLED(set) ? (set & 63u) : 0xFFFFu); }

// ---------------- Tunables ----------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU depth for hot
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail for cold
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream promote at 2nd+ demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 3;  // stream promote at 3rd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote stream lines on hit
static constexpr uint8_t PC_USE_HOT_THRESH   = 7;  // TinyLFU hot threshold (0..15)
static constexpr uint8_t GSEL_MAX            = 31; // global selector range
static constexpr uint8_t GSEL_THRES          = 8;  // need strong B advantage for followers
static constexpr uint32_t LFU_DECAY_PERIOD   = 4096; // periodic decay

// ---------------- Per-line RRIP ----------------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// ---------------- Global selector (policy dueling) ----------------
static uint8_t GSEL = 0; // 0=prefer Mode A; >=GSEL_THRES -> enable Mode B on followers
static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}

// ---------------- Mode A (Hawkeye-like SHCT with leader training) ---
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx_pc(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline uint32_t shct_idx_sig(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-only per-line tags (64 leader sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // store 12-bit PC signature
static uint8_t  hawk_used[64][LLC_WAYS];    // 1-bit: was re-referenced while resident
static uint8_t  hawk_prefetch[64][LLC_WAYS];// 1-bit: was inserted by prefetch

// ---------------- Mode B (RunGuard + TinyLFU) --------------------
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10 bits

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_phase2[PC_TBL_SIZE];      // 2b demand hit phase (0..3)

// Tiny helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }

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

static inline bool pseudo_80pct_gate(uint64_t PC, uint32_t set) {
    // Deterministic 80% gate using a simple hash
    uint32_t h = (pc_index(PC) ^ (set * 131u)) & 0xFFu;
    return (h % 10u) < 8u; // true ~80% of the time
}

// Periodic decay for TinyLFU
static uint32_t op_count = 0;
static inline void maybe_decay_lfu() {
    op_count++;
    if ((op_count & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;
            // phase naturally saturates; no decay needed
        }
    }
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_prefetch, 0, sizeof(hawk_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_phase2, 0, sizeof(pc_phase2));
    GSEL = 0;
    op_count = 0;
}

// ---------------- Victim selection (RRIP + leader training/dueling) ----------------
static inline void train_hawkeye_on_victim(uint32_t set, uint32_t way) {
    if (!SEL_SAMPLED(set)) return;
    uint32_t slot = LEADER_SLOT(set);
    uint16_t sig = hawk_sig[slot][way];
    uint8_t used = hawk_used[slot][way];
    uint8_t was_pf = hawk_prefetch[slot][way];
    if (sig) {
        uint32_t idx = shct_idx_sig(sig);
        if (was_pf) {
            if (used) shct_inc(shct_prefetch[idx]); else shct_dec(shct_prefetch[idx]);
        } else {
            if (used) shct_inc(shct_demand[idx]); else shct_dec(shct_demand[idx]);
        }
    }
    hawk_used[slot][way] = 0; // reset for next occupant
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            // Dueling: count demand misses for selector even when using invalid slots
            if (is_demand(type)) {
                if (LEADER_A(set) && GSEL < GSEL_MAX) GSEL++;
                else if (LEADER_B(set) && GSEL > 0) GSEL--;
            }
            return w;
        }
    }

    // 2) Search for a line at maxRRPV
    for (;;) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                // Train Hawkeye only when evicting a valid line in a leader set
                train_hawkeye_on_victim(set, w);
                // Dueling: demand miss attribution
                if (is_demand(type)) {
                    if (LEADER_A(set) && GSEL < GSEL_MAX) GSEL++;
                    else if (LEADER_B(set) && GSEL > 0) GSEL--;
                }
                return w;
            }
        }
        // Age all lines (bounded; terminates because values saturate to maxRRPV)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// ---------------- Update state (hits and fills) ----------------
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
    bool modeB = use_modeB(set);

    // Mark Hawkeye "used" for leader sets on any hit
    if (hit) {
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
    }

    if (hit) {
        if (modeB) {
            if (is_demand(type)) {
                uint32_t idx = pc_index(PC);
                sat_inc_u4(pc_use4[idx]);
                if (pc_phase2[idx] < 3) pc_phase2[idx]++;

                bool stream_now = (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
                if (stream_now) {
                    if (STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                    // allow short reuse: clear stream flag after 2 demand hits
                    if (pc_phase2[idx] >= 2) pc_stream_conf[idx] = 0;
                    // promote only after 3rd demand hit (post de-flag)
                    if (pc_stream_conf[idx] == 0 && pc_phase2[idx] >= HITS_PROMOTE_STR) {
                        rrpv[set][way] = 0;
                    }
                } else {
                    if (pc_phase2[idx] >= HITS_PROMOTE_NS) {
                        rrpv[set][way] = 0;
                    } else {
                        // light nudge toward MRU without full promotion
                        if (rrpv[set][way] > 0) rrpv[set][way]--;
                    }
                }
            } else {
                // Prefetch/writeback hits: no aggressive promotion in Mode B
                if (rrpv[set][way] > 0) rrpv[set][way]--;
            }
        } else {
            // Mode A: classic hit-to-MRU
            rrpv[set][way] = 0;
        }
        maybe_decay_lfu();
        return;
    }

    // Miss/fill path
    if (SEL_SAMPLED(set)) {
        // Record leader tags for Hawkeye training
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig12 = (uint16_t)(PC & 0xFFFu);
        hawk_sig[slot][way] = sig12;
        hawk_used[slot][way] = 0;
        hawk_prefetch[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
    }

    if (modeB) {
        if (type == ACCESS_WRITEBACK) {
            // Never bypass WB: conservative cold insertion
            rrpv[set][way] = INSERT_COLD_DEPTH;
        } else if (type == ACCESS_PREFETCH) {
            // Quarantine prefetches at tail
            rrpv[set][way] = maxRRPV;
        } else {
            // Demand
            uint32_t idx = pc_index(PC);
            bool is_stream = detect_and_update_stream(PC, paddr);
            pc_phase2[idx] = 0; // reset phase on new fill for this PC

            if (is_stream) {
                // Deterministic 80% pseudo-bypass -> hard tail; remaining also tail but may be touched
                (void)pseudo_80pct_gate(PC, set); // gate kept for determinism; both paths insert tail safely
                rrpv[set][way] = maxRRPV;
            } else {
                // TinyLFU usefulness gate
                if (pc_use4[idx] >= PC_USE_HOT_THRESH) rrpv[set][way] = INSERT_WARM_DEPTH;
                else rrpv[set][way] = INSERT_COLD_DEPTH;
            }
        }
    } else {
        // Mode A (Hawkeye-like) insertion
        if (type == ACCESS_WRITEBACK) {
            rrpv[set][way] = INSERT_COLD_DEPTH;
        } else if (type == ACCESS_PREFETCH) {
            rrpv[set][way] = maxRRPV;
        } else {
            uint32_t idx = shct_idx_pc(PC);
            uint8_t ctr = shct_demand[idx];
            if (ctr >= (SHCT_MAX / 2)) rrpv[set][way] = INSERT_WARM_DEPTH;
            else rrpv[set][way] = INSERT_COLD_DEPTH;
        }
    }

    maybe_decay_lfu();
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}