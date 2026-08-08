#include <cstdint>
#include <cstring>
#include <algorithm>
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

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (R2) -----------------------------------
static constexpr uint8_t  maxRRPV               = 7;    // 3-bit RRIP max
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;    // near-MRU (RRIP=2)
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;    // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;    // need 2 forward (+1/+2) steps
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;    // non-stream MRU starting at 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;    // stream escape/promo at 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH   = 2;    // demote quarantined on touch by 2
static constexpr uint8_t  PREFETCH_DEMOTE_TOUCH = 2;    // demote on prefetch touch by 2
static constexpr uint8_t  PC_USE_HOT_THRESH     = 7;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024; // halve every N demand ops
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;    // enable Mode B if deadB better by >=2
static constexpr uint64_t STREAM_SAMPLE_MASK    = 7ull; // 1/8 sampling
static constexpr uint64_t STREAM_SAMPLE_MATCH   = 0ull; // sample when ((line_id & 7)==0)

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Mode selector: leader scoreboard ----------------
// Track dead-on-evict (no demand hits) in leader sets
static uint8_t deadA[64];
static uint8_t deadB[64];

// Followers consult slot outcome; keep a small per-set sticky score (optional)
static int8_t bandit_score[LLC_SETS]; // informational; decision uses leader deltas

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    uint32_t slot = LEADER_SLOT(set);
    int16_t delta = (int16_t)deadA[slot] - (int16_t)deadB[slot];
    bandit_score[set] = (int8_t)std::max<int16_t>(-8, std::min<int16_t>(7, delta));
    return (delta >= MODEB_ENABLE_THRESH);
}

// ---------------- Mode A (Hawkeye-like SHiP) ----------------------
// Tiny SHCTs (trained in leader-A sets)
#define SHCT_SIZE_BITS 10
#define SHCT_SIZE (1u << SHCT_SIZE_BITS)   // 1024 entries
#define SHCT_MAX 31                        // 5-bit counters
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }
static constexpr uint8_t SHCT_FRIENDLY_THR = 16;

// Leader-A per-line bookkeeping (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit semantic
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1-bit semantic

// ---------------- Mode B (VFQ + StreamGuardian) ------------------
// 512-entry per-PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10-bit semantic
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit semantic (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4-bit TinyLFU (0..15)

// ---------------- Helpers ----------------------------------------
static inline void rrpv_bump_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}
static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote_all_the_way(uint32_t set, uint32_t way) {
    rrpv[set][way] = 0;
}
static inline void rrpv_demote_step(uint32_t set, uint32_t way, uint8_t step) {
    uint8_t v = rrpv[set][way];
    v = (uint8_t)std::min<uint8_t>(maxRRPV, (uint8_t)(v + step));
    rrpv[set][way] = v;
}
static inline bool any_rrpv_at_max(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] == maxRRPV) return true;
    return false;
}

// Forward-run detector: returns true if in stream after update (only for demand)
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (fwd) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static uint32_t demand_op_count = 0;
static inline void maybe_decay_lfu(uint32_t type) {
    if (!is_demand(type)) return;
    demand_op_count++;
    if ((demand_op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1; // halve counts
    }
}
static inline void lfu_touch(uint64_t PC, uint32_t type) {
    if (!is_demand(type)) return;
    uint32_t idx = pc_index(PC);
    if (pc_use4[idx] < 15) pc_use4[idx]++;
}

static inline bool sample_stream(uint64_t paddr) {
    uint64_t line = (paddr >> 6);
    return ((line & STREAM_SAMPLE_MASK) == STREAM_SAMPLE_MATCH);
}

// ---------------- Initialize replacement state --------------------
void InitReplacementState() {
    std::memset(rrpv, 0, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start old
        }
    }
    std::memset(deadA, 0, sizeof(deadA));
    std::memset(deadB, 0, sizeof(deadB));
    std::memset(bandit_score, 0, sizeof(bandit_score));

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));

    demand_op_count = 0;
}

// ---------------- Victim selection (RRIP) -------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP: find a line with RRPV==max; age if none
    while (!any_rrpv_at_max(set)) {
        rrpv_bump_all(set);
    }
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    return 0; // fallback
}

// ---------------- Update state on hit/fill ------------------------
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
    maybe_decay_lfu(type);
    if (is_demand(type)) {
        lfu_touch(PC, type);
        // keep stream detector current on demand references
        (void)detect_and_update_stream(PC, paddr);
    }

    if (hit) {
        // On hit: multi-hit gate and quarantines
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_promote_all_the_way(set, way);
                } else {
                    rrpv_demote_step(set, way, STREAM_DEMOTE_TOUCH);
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_promote_all_the_way(set, way);
                } else {
                    // first demand hit: hot-PC nudge (RRIP -2, but keep >=1)
                    uint32_t idx = pc_index(PC);
                    if (pc_use4[idx] >= PC_USE_HOT_THRESH) {
                        uint8_t v = rrpv[set][way];
                        if (v > 2) rrpv[set][way] = v - 2;
                        else rrpv[set][way] = 1;
                    }
                }
            }
        } else {
            // prefetch touch never promotes; if quarantined, demote harder
            if (stream_lock[set][way]) {
                rrpv_demote_step(set, way, PREFETCH_DEMOTE_TOUCH);
            } else {
                rrpv_demote_step(set, way, 1);
            }
        }
        return;
    }

    // Miss fill: train evicted line (leader sets), then choose insertion policy
    // Leader training on eviction: use prior line's lifetime (hitcnt==0 => dead)
    if (LEADER_A(set) || LEADER_B(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Dead-on-evict if no demand hits
        bool was_dead = (hitcnt[set][way] == 0);
        if (LEADER_A(set)) {
            if (was_dead) { if (deadA[slot] < 255) deadA[slot]++; }
            // SHiP training with previous occupant's signature (leader-A only)
            uint16_t prev_sig = hawk_sig[slot][way];
            uint8_t is_pref = hawk_is_pref[slot][way];
            if (prev_sig) {
                uint32_t idx = shct_idx(prev_sig);
                if (is_pref) {
                    if (was_dead) shct_dec(shct_prefetch[idx]);
                    else          shct_inc(shct_prefetch[idx]);
                } else {
                    if (was_dead) shct_dec(shct_demand[idx]);
                    else          shct_inc(shct_demand[idx]);
                }
            }
        } else { // LEADER_B
            if (was_dead) { if (deadB[slot] < 255) deadB[slot]++; }
        }
    }

    // Decide mode for this set (followers guarded by leaders)
    bool use_modeB = modeB_enabled(set);

    // Compute insertion for the new line
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t lock = 0;
    uint8_t new_hitcnt = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass WB; insert relatively warm
        ins_rrpv = INSERT_WARM_DEPTH;
        lock = 0;
    } else if (!use_modeB) {
        // Mode A: Hawkeye-like SHiP
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        bool friendly = false;
        if (type == ACCESS_PREFETCH) friendly = (shct_prefetch[idx] >= SHCT_FRIENDLY_THR);
        else                         friendly = (shct_demand[idx]  >= SHCT_FRIENDLY_THR);

        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // prefetch quarantine tail
            lock = 1;
        } else {
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock = 0;
        }

        // Record signature for leader-A training
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        }
    } else {
        // Mode B: VFQ + StreamGuardian
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // quarantine at tail
            lock = 1;
        } else {
            // demand
            bool is_stream = false;
            // detector already updated for this demand above; re-check confidence
            uint32_t pc_idx = pc_index(PC);
            is_stream = (pc_stream_conf[pc_idx] >= STREAM_CONF_THRESH);
            if (is_stream) {
                // approximate bypass: hard-tail + quarantine; sample 1/8 (still tail)
                (void)sample_stream(paddr); // deterministic sample, same action under ChampSim
                ins_rrpv = maxRRPV;
                lock = 1;
            } else {
                // cold/hot by TinyLFU
                uint8_t use = pc_use4[pc_idx];
                ins_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                lock = 0;
            }
        }
    }

    // Install new metadata
    rrpv_set(set, way, ins_rrpv);
    stream_lock[set][way] = lock;
    hitcnt[set][way] = new_hitcnt;
}

// ---------------- Stats hooks (silent) ----------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}