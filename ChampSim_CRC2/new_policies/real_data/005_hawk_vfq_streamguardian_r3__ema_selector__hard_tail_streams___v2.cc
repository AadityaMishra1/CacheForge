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

// ---------------- Tunables (R3) -----------------------------------
static constexpr uint8_t  maxRRPV               = 7;    // 3-bit RRIP max
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;    // near-MRU (RRIP=2)
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;    // near-tail (RRIP=6)
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;    // need 2 forward (+1/+2) steps
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;    // non-stream: MRU on 2nd+ demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;    // stream-locked escape at 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH   = 2;    // demote quarantined on demand touch
static constexpr uint8_t  PREFETCH_DEMOTE_TOUCH = 2;    // demote on prefetch touch
static constexpr uint8_t  PC_USE_HOT_THRESH     = 7;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024; // halve every N demand ops
static constexpr uint64_t STREAM_SAMPLE_MASK    = 7ull; // 1/8 sampling
static constexpr uint64_t STREAM_SAMPLE_MATCH   = 0ull; // sample when ((line_id & 7)==0)

// EMA selector params
static constexpr uint8_t  MODEB_EMA_SHIFT       = 3;    // decay: ema -= ema>>3
static constexpr uint8_t  MODEB_EMA_INC         = 8;    // dead contribution
static constexpr uint8_t  MODEB_ENABLE_DELTA    = 16;   // enable Mode B if emaA - emaB >= 16 (~2)

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Mode selector: leader EMA scoreboard ------------
static uint8_t emaA[64];
static uint8_t emaB[64];

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    uint32_t slot = LEADER_SLOT(set);
    int16_t delta = (int16_t)emaA[slot] - (int16_t)emaB[slot];
    return (delta >= MODEB_ENABLE_DELTA);
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

// ---------------- Mode B (VFQ + StreamGuardian) -------------------
// 512-entry per-PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10-bit semantic
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit semantic (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4-bit TinyLFU (0..15)
static uint32_t demand_tick = 0;

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

static inline void lfu_on_demand(uint64_t PC) {
    demand_tick++;
    uint32_t idx = pc_index(PC);
    if (pc_use4[idx] < 15) pc_use4[idx]++;
    if ((demand_tick & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
}

static inline void ema_decay_add(uint8_t& ema, bool add) {
    ema = (uint8_t)(ema - (ema >> MODEB_EMA_SHIFT));
    if (add) {
        uint16_t tmp = (uint16_t)ema + MODEB_EMA_INC;
        ema = (uint8_t)std::min<uint16_t>(255u, tmp);
    }
}

// ---------------- CRC2 Hooks --------------------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv)); // initialize all to max
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));

    std::memset(emaA, 0, sizeof(emaA));
    std::memset(emaB, 0, sizeof(emaB));

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    demand_tick = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim selection
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_bump_all(set); // saturating bump; loop will terminate
    }
    // unreachable
    return 0;
}

// Update replacement state
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
    bool demand = is_demand(type);

    // Update per-PC structures on demand references
    if (demand) {
        lfu_on_demand(PC);
        (void)detect_and_update_stream; // silence unused if no demand
    }

    // HIT path
    if (hit) {
        if (demand) {
            // Update stream detector for runs
            (void)detect_and_update_stream(PC, paddr);

            // Increment per-line demand hit count (cap at 3)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    rrpv_promote_all_the_way(set, way);
                    stream_lock[set][way] = 0;
                } else {
                    rrpv_demote_step(set, way, STREAM_DEMOTE_TOUCH);
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_promote_all_the_way(set, way);
                }
            }

            // SHiP reinforcement (leader-A only)
            if (LEADER_A(set)) {
                uint32_t slot = LEADER_SLOT(set);
                uint16_t sig = hawk_sig[slot][way];
                if (sig) {
                    if (hawk_is_pref[slot][way]) {
                        uint32_t idx = shct_idx(sig);
                        shct_inc(shct_prefetch[idx]);
                    } else {
                        uint32_t idx = shct_idx(sig);
                        shct_inc(shct_demand[idx]);
                    }
                }
            }
        } else if (type == ACCESS_PREFETCH) {
            // Prefetch touch: keep it cold
            rrpv_demote_step(set, way, PREFETCH_DEMOTE_TOUCH);
        }
        return;
    }

    // MISS path (about to fill 'way'): train on eviction of previous line
    bool was_dead = (hitcnt[set][way] == 0);

    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        ema_decay_add(emaA[slot], was_dead);
        // SHiP decay (on dead) for previous line, using recorded sig
        uint16_t osig = hawk_sig[slot][way];
        if (osig) {
            uint32_t idx = shct_idx(osig);
            if (hawk_is_pref[slot][way]) shct_dec(shct_prefetch[idx]);
            else                         shct_dec(shct_demand[idx]);
        }
    } else if (LEADER_B(set)) {
        uint32_t slot = LEADER_SLOT(set);
        ema_decay_add(emaB[slot], was_dead);
    }

    // Reset per-line metadata for the new line
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    // WRITEBACKs: always insert (no bypass), moderate depth
    if (type == ACCESS_WRITEBACK) {
        rrpv_set(set, way, INSERT_WARM_DEPTH);
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = 0;
            hawk_is_pref[slot][way] = 0;
        }
        return;
    }

    // PREFETCH: quarantine at tail, never early-promoted
    if (type == ACCESS_PREFETCH) {
        rrpv_set(set, way, maxRRPV);
        stream_lock[set][way] = 1;
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_is_pref[slot][way] = 1;
        }
        return;
    }

    // DEMAND fill
    bool is_stream = detect_and_update_stream(PC, paddr);

    if (is_stream) {
        // Hard-tail stream quarantine; 1/8 sampled to RRIP=6
        uint64_t line_id = (paddr >> 6);
        bool sample = ((line_id & STREAM_SAMPLE_MASK) == STREAM_SAMPLE_MATCH);
        rrpv_set(set, way, sample ? (maxRRPV - 1) : maxRRPV);
        stream_lock[set][way] = 1;
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_is_pref[slot][way] = 0;
        }
        return;
    }

    // Non-stream demand: choose insertion via selector
    bool use_modeB = modeB_enabled(set);
    if (LEADER_A(set)) use_modeB = false;
    if (LEADER_B(set)) use_modeB = true;

    uint8_t ins_depth = INSERT_COLD_DEPTH;

    if (use_modeB) {
        // Mode B: TinyLFU nudge for hot PCs
        uint32_t idx = pc_index(PC);
        ins_depth = INSERT_COLD_DEPTH; // default cold
        if (pc_use4[idx] >= PC_USE_HOT_THRESH) {
            int d = (int)ins_depth - 2;
            if (d < 1) d = 1;
            ins_depth = (uint8_t)d;
        }
        // no Mode-B per-line extra metadata
    } else {
        // Mode A: Hawkeye-like SHiP
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t conf = shct_demand[idx];
        bool friendly = (conf >= SHCT_FRIENDLY_THR);
        ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

        // Record per-line sig for training (leader-A only)
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_is_pref[slot][way] = 0;
        }
    }

    rrpv_set(set, way, ins_depth);
    stream_lock[set][way] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}