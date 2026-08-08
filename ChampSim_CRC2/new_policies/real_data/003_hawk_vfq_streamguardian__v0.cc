#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (ChampSim CRC2)
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
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 valid if sampled

// ---------------- Tunables ----------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // need 2 forward (+1/+2) steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU starting at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape/promo at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined on touch by 1
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048; // halve every N demand ops
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // enable Mode B in followers when >=2

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector ------------------------
static int8_t bandit_score[LLC_SETS]; // followers use Mode B only if >= MODEB_ENABLE_THRESH

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Mode A (Hawkeye-like SHiP) ---------------------
// Tiny SHCTs (demand & prefetch), trained in leader-A sets
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

// Tiny per-region (page) age (256 entries x 2b)
static constexpr uint32_t REGION_TBL_SIZE = 256;
static uint8_t region_age2[REGION_TBL_SIZE]; // 0..3
static inline uint32_t region_index(uint64_t paddr) { return (uint32_t)((paddr >> 12) & (REGION_TBL_SIZE - 1u)); }

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
    v = (uint8_t)std::min<uint8_t>(maxRRPV, v + step);
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
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(region_age2, 0, sizeof(region_age2));
    demand_op_count = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes (ensure termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // 4) Fallback: choose the way with largest RRPV, break ties by highest index
    uint8_t best_v = 0;
    uint32_t best_w = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best_v) { best_v = rrpv[set][w]; best_w = w; }
    }
    return best_w;
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    maybe_decay_lfu(type);
    if (hit) {
        // Hits: multi-hit gating and stream demotion
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            // Escape rule for stream-locked lines
            if (stream_lock[set][way]) {
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_promote_all_the_way(set, way); // MRU
                } else {
                    // demote quarantined lines on first touch
                    rrpv_demote_step(set, way, STREAM_DEMOTE_TOUCH);
                }
            } else {
                // Non-stream promotion only starting on 2nd demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_promote_all_the_way(set, way); // MRU
                }
            }
        } else {
            // Prefetch hits do not promote; optionally demote quarantined a bit
            if (stream_lock[set][way]) rrpv_demote_step(set, way, STREAM_DEMOTE_TOUCH);
        }
        // Region becomes hotter on reuse
        uint32_t ridx = region_index(paddr);
        if (region_age2[ridx] < 3) region_age2[ridx]++;
        return;
    }

    // From here: this is a fill/placement
    // Train on eviction of previous occupant (using its residual per-line state)
    bool prev_used = (hitcnt[set][way] > 0);
    uint32_t victim_ridx = region_index(victim_addr);
    if (prev_used) {
        if (region_age2[victim_ridx] < 3) region_age2[victim_ridx]++;
    } else {
        if (region_age2[victim_ridx] > 0) region_age2[victim_ridx]--;
    }

    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Leader-A: train SHCT
        if (LEADER_A(set)) {
            uint16_t old_sig = hawk_sig[slot][way];
            bool old_pref = (hawk_is_pref[slot][way] != 0);
            if (old_sig) {
                uint32_t idx = shct_idx(old_sig);
                if (prev_used) {
                    if (old_pref) shct_inc(shct_prefetch[idx]); else shct_inc(shct_demand[idx]);
                } else {
                    if (old_pref) shct_dec(shct_prefetch[idx]); else shct_dec(shct_demand[idx]);
                }
            }
            // Bandit learns negatively on dead blocks from Mode A leaders
            if (!prev_used && bandit_score[set] > -16) bandit_score[set]--;
        }
        // Leader-B: bandit learns positively on live blocks
        if (LEADER_B(set)) {
            if (prev_used && bandit_score[set] < 15) bandit_score[set]++;
        }
    }

    // Reset per-line state for the new line
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    // Writebacks: never bypass; insert reasonably warm, no training
    if (type == ACCESS_WRITEBACK) {
        rrpv_set(set, way, INSERT_WARM_DEPTH);
        return;
    }

    // Update PC frequency and stream detector for demand
    if (is_demand(type)) lfu_touch(PC, type);
    bool is_stream = false;
    if (is_demand(type)) is_stream = detect_and_update_stream(PC, paddr);

    // Mode selection
    bool useB = modeB_enabled(set);

    if (!useB) {
        // ---------------- Mode A: Hawkeye-like SHiP insertion ----------------
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t ctr = is_demand(type) ? shct_demand[idx] : shct_prefetch[idx];
        bool friendly = (ctr >= SHCT_FRIENDLY_THR);

        uint8_t ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        if (type == ACCESS_PREFETCH) ins_depth = maxRRPV; // prefetch quarantine always
        rrpv_set(set, way, ins_depth);

        // Bookkeeping for future training in leader-A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        }
    } else {
        // ---------------- Mode B: VFQ + StreamGuardian ----------------
        uint32_t pcidx = pc_index(PC);
        uint32_t ridx = region_index(paddr);
        bool pc_hot = (pc_use4[pcidx] >= PC_USE_HOT_THRESH);
        bool region_hot = (region_age2[ridx] >= 2);

        if (type == ACCESS_PREFETCH) {
            // Prefetch quarantine at tail; demote-on-touch applies via stream_lock
            rrpv_set(set, way, maxRRPV);
            stream_lock[set][way] = 1;
        } else {
            if (is_stream) {
                // Aggressive quarantine for streams; allow carve-out for hot PCs
                uint8_t depth = pc_hot ? (uint8_t)5 : maxRRPV;
                rrpv_set(set, way, depth);
                stream_lock[set][way] = 1;
                // Under pressure, hasten eviction of quarantined content
                if (!any_rrpv_at_max(set)) rrpv_bump_all(set);
            } else {
                // Non-stream: combine PC hotness and region heat
                uint8_t depth = (pc_hot || region_hot) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                rrpv_set(set, way, depth);
                // not stream-locked
            }
        }
        // No extra per-line bookkeeping needed for Mode B
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}