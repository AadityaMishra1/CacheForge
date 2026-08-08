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

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Leader-set sampling (64 leaders; 32 A + 32 B) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t A_LEADER_IDX(uint32_t set) { return ((set & 63u) >> 1); } // 0..31 for leader-A sets

// ---------------- Tunables (lbm-first, zeusmp-safe) ----------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP (0..7)
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU insert
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail insert
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps to tag as stream
static constexpr uint8_t STRONG_STREAM_CONF  = 3;  // 3rd forward step => hard-tail (bypass-equivalent)
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream promote at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 3;  // stream escape at 3rd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined stream on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 5;  // TinyLFU hot threshold (0..15)
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // followers enable B only if >= 2

// ---------------- Per-line packed metadata (conceptual bit-packing) ----------
// rrpv:3b, hit_once:1b, stream_tag:1b, two_hit_stream:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hit_once[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];
static uint8_t two_hit_stream[LLC_SETS][LLC_WAYS];

// ---------------- Victim bookkeeping (to train on real evictions) ------------
static uint8_t pending_eviction[LLC_SETS]; // 0=no eviction (invalid way used), 1=eviction occurred

// ---------------- Global bandit selector (conservative) ----------------------
static int8_t global_advantage = 0; // followers use Mode B only if >= MODEB_ENABLE_THRESH
static inline void bandit_inc(int8_t d = 1) {
    int16_t t = (int16_t)global_advantage + d;
    if (t > 32) t = 32;
    global_advantage = (int8_t)t;
}
static inline void bandit_dec(int8_t d = 1) {
    int16_t t = (int16_t)global_advantage - d;
    if (t < -32) t = -32;
    global_advantage = (int8_t)t;
}
static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (global_advantage >= MODEB_ENABLE_THRESH);
}

// ---------------- Mode A (Hawkeye-like) -------------------------------------
// Tiny SHCTs (demand & prefetch), trained on leader-A sets only
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS) // 2048
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature (lower 12 bits of PC) + prefetched flag per line for training (leader-A sets only)
static uint16_t hawk_sigA[32][LLC_WAYS]; // 12b effective in 16b storage
static uint8_t  hawk_prefA[32][LLC_WAYS]; // 0/1

// ---------------- Mode B (StreamGuard + TinyLFU) -----------------------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)

// Simple decay for TinyLFU
static uint32_t decay_ptr = 0;
static uint32_t decay_epoch = 0;

// ---------------- Helpers ----------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t& x) { if (x > 0) x--; }
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

static inline bool strong_stream(uint64_t PC) {
    return (pc_stream_conf[pc_index(PC)] >= STRONG_STREAM_CONF);
}

// ---------------- Initialization ---------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        pending_eviction[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hit_once[s][w] = 0;
            stream_tag[s][w] = 0;
            two_hit_stream[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sigA, 0, sizeof(hawk_sigA));
    std::memset(hawk_prefA, 0, sizeof(hawk_prefA));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    decay_ptr = 0;
    decay_epoch = 0;
    global_advantage = 0;
}

// ---------------- Victim selection (RRIP) ------------------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            pending_eviction[set] = 0; // no eviction training
            return w;
        }
    }
    // 2) RRIP victim search with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] >= maxRRPV) {
                pending_eviction[set] = 1; // a valid victim will be evicted
                return w;
            }
        }
        // Age all lines (saturate)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// ---------------- Find victim in the set -------------------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update replacement state -----------------------------------
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
    (void)cpu; (void)victim_addr;

    // TinyLFU update + stream detector update on demand/prefetch accesses
    if (is_demand(type) || type == ACCESS_PREFETCH) {
        uint32_t pidx = pc_index(PC);
        if (is_demand(type)) sat_inc_u4(pc_use4[pidx]);
        // light decay
        decay_epoch++;
        if ((decay_epoch & 31u) == 0u) {
            uint32_t d = decay_ptr++ & (PC_TBL_SIZE - 1);
            sat_dec_u4(pc_use4[d]);
        }
        // update run detector
        (void)detect_and_update_stream(PC, paddr);
    }

    // On hit: multi-hit gating and stream demote-on-touch
    if (hit) {
        if (type != ACCESS_WRITEBACK) {
            // Stream demotion on touch
            if (STREAM_DEMOTE_TOUCH && stream_tag[set][way]) {
                if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            }
            // Multi-hit gating (demand only)
            if (is_demand(type)) {
                if (!stream_tag[set][way]) {
                    // Non-stream: promote on 2nd+ demand hit
                    if (hit_once[set][way] == 0) {
                        hit_once[set][way] = 1; // first demand hit
                    } else {
                        rrpv[set][way] = 0; // MRU
                    }
                } else {
                    // Stream: require 3rd demand hit to escape
                    if (hit_once[set][way] == 0) {
                        hit_once[set][way] = 1; // first demand hit
                    } else if (two_hit_stream[set][way] == 0) {
                        two_hit_stream[set][way] = 1; // second demand hit
                    } else {
                        // third demand hit: escape quarantine
                        stream_tag[set][way] = 0;
                        rrpv[set][way] = 0; // MRU
                    }
                }
            }
        }
        return;
    }

    // Miss path: train (if real eviction), then insert new line
    // Training on eviction outcomes for leaders
    if (pending_eviction[set]) {
        // Determine dead vs reused (any demand hit seen)
        bool victim_dead = (hit_once[set][way] == 0);
        if (LEADER_B(set)) {
            // Mode B leader: harsher penalty for dead lines, mild reward for reuse
            if (victim_dead) bandit_dec(2);
            else bandit_inc(1);
        }
        if (LEADER_A(set)) {
            // Train Hawkeye SHCT (leader-A only)
            uint32_t aidx = A_LEADER_IDX(set);
            uint16_t sig12 = hawk_sigA[aidx][way] & 0x0FFFu;
            uint32_t si = sig12 & (SHCT_SIZE - 1);
            if (hawk_prefA[aidx][way]) {
                if (victim_dead) shct_dec(shct_prefetch[si]);
                else             shct_inc(shct_prefetch[si]);
            } else {
                if (victim_dead) shct_dec(shct_demand[si]);
                else             shct_inc(shct_demand[si]);
            }
            // Mode comparison from leader-A perspective (dead under A => nudge toward B; reuse => nudge toward A)
            if (victim_dead) bandit_inc(1);
            else             bandit_dec(1);
        }
    }
    pending_eviction[set] = 0; // reset for next access

    // Determine mode for insertion
    bool useB = modeB_enabled(set);

    // Decide insertion RRPV and tags
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t is_stream = 0;

    if (type == ACCESS_WRITEBACK) {
        ins_rrpv = INSERT_WARM_DEPTH; // never bypass on writeback
    } else if (!useB) {
        // Mode A (Hawkeye-like)
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // quarantine prefetches
        } else {
            uint32_t si = shct_idx(PC);
            bool friendly = (shct_demand[si] >= 8);
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    } else {
        // Mode B (StreamGuard + TinyLFU)
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // always quarantine prefetch
        } else {
            bool stream2 = detect_and_update_stream(PC, paddr); // already called above; safe to re-evaluate
            bool strong  = strong_stream(PC);
            if (strong || stream2) {
                ins_rrpv = maxRRPV; // hard-tail quarantine (bypass-equivalent)
                is_stream = 1;
            } else {
                // TinyLFU-guided warm/cold
                uint32_t pidx = pc_index(PC);
                bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
                ins_rrpv = hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        }
    }

    // Install new line state
    rrpv[set][way] = ins_rrpv;
    stream_tag[set][way] = is_stream;
    hit_once[set][way] = 0;
    two_hit_stream[set][way] = 0;

    // Record Hawkeye signature for leader-A sets only
    if (LEADER_A(set)) {
        uint32_t aidx = A_LEADER_IDX(set);
        hawk_sigA[aidx][way] = (uint16_t)(PC & 0x0FFFu);
        hawk_prefA[aidx][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}