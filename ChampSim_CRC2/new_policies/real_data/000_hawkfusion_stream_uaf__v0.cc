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
static inline bool is_demand(uint32_t t) { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048) = 11
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 sampled sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (carefully set for lbm/mcf balance) ----------------
static constexpr uint8_t  maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;  // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;  // near-tail insertion
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;  // need 2 forward steps (+1/+2)
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;  // TinyLFU hot PCs carveout
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // decay period (ops)
static constexpr uint32_t BANDIT_EPOCH        = 4096; // epoch to nudge per-set selector
static constexpr uint8_t  GSEL_MAX            = 31;   // global selector range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set score gate

// ---------------- Per-line metadata (bit-packed conceptually) ----------------
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set/epoch selector -------------------------
static int8_t bandit_score[LLC_SETS]; // followers enable Mode B if >= MODEB_ENABLE_THRESH
static uint8_t GSEL = 0;              // global conservative selector (0..31)

// ---------------- Mode A: Hawkeye-like SHiP (leader-A only) ------
// Tiny SHCTs (1K x 5b) for demand and prefetch
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Per-line training state for leader-A sets only (64 x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit effective stored in 16-bit
static uint8_t  hawk_used[64][LLC_WAYS];    // first demand reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by a prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b folded PC signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B: Stream-UAF (PC stride + TinyLFU) -------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b line id (in 16b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 4b (0..15)

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

// Forward-run detector: returns true if now in streaming run after update
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

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Decay TinyLFU and nudge per-set bandit scores slowly toward global decision
static uint32_t op_count = 0;
static inline void periodic_housekeeping() {
    op_count++;
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        // Nudge all follower sets toward global decision
        bool preferB = (GSEL >= GSEL_THRES);
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (SEL_SAMPLED(s)) continue; // leaders fixed
            if (preferB) sat_inc_i8(bandit_score[s]);
            else         sat_dec_i8(bandit_score[s]);
        }
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
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    GSEL = 0;
    op_count = 0;
}

// ---------------- Victim selection (3-bit RRIP) -------------------
static inline uint32_t rrip_get_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Return any invalid first
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) If any line at maxRRPV, evict it
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes to ensure termination
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

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_get_victim(set, current_set);
}

// ---------------- Update state on hit/fill ------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    periodic_housekeeping();

    // Normalize to line granularity
    paddr = (paddr >> 6) << 6;

    // Demand activity contributes to TinyLFU
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    // Stream detector (update on all accesses)
    bool is_stream = detect_and_update_stream(PC, paddr);

    // Leader-A training for Hawkeye-like SHiP
    if (SEL_SAMPLED(set) && LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hit) {
            // Positive training on first demand hit
            if (is_demand(type) && (hawk_used[slot][way] == 0)) {
                hawk_used[slot][way] = 1;
                uint16_t sig = hawk_sig[slot][way];
                uint32_t idx = shct_idx(sig);
                if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
                else                         shct_inc(shct_demand[idx]);
            }
        } else {
            // On fill, train the evicted old occupant negatively if it never saw demand reuse
            // Use leader-A metadata stored in our arrays for 'way'
            if (hawk_used[slot][way] == 0) {
                uint16_t esig = hawk_sig[slot][way];
                uint32_t eidx = shct_idx(esig);
                if (hawk_is_pref[slot][way]) shct_dec(shct_prefetch[eidx]);
                else                         shct_dec(shct_demand[eidx]);
            }
            // Initialize leader-A metadata for the new line
            hawk_sig[slot][way]     = pc_sig12(PC);
            hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
            hawk_used[slot][way]    = 0;
        }
    }

    // Selector: train global GSEL using leader sets on eviction outcome
    if (!hit && SEL_SAMPLED(set)) {
        // Evicted line is the previous content at (set, way); judge it by hitcnt
        bool evicted_dead = (hitcnt[set][way] == 0);
        if (LEADER_A(set)) {
            // If Mode A's set yields dead lines, move toward enabling B
            if (evicted_dead) gsel_inc();
            else              gsel_dec();
        } else if (LEADER_B(set)) {
            // Reward B when it keeps lines alive
            if (evicted_dead) gsel_dec();
            else              gsel_inc();
        }
    }

    // WRITEBACK: do not bypass; treat conservatively (tail-ish) and return
    if (type == ACCESS_WRITEBACK) {
        if (!hit) {
            rrpv_set(set, way, maxRRPV); // tail insertion for WB
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 0;
        }
        return;
    }

    // ---------------- On cache hit ----------------
    if (hit) {
        // Prefetch touches do not promote; demand hits may promote by gating
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                // Demote quarantined streams on touch; escape on sufficient reuse
                rrpv_demote(set, way);
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, 0); // MRU on escape
                }
            } else {
                // Non-stream: promote to MRU starting on 2nd demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                } else {
                    // light promotion
                    rrpv_promote(set, way);
                }
            }
        }
        return;
    }

    // ---------------- On cache fill (miss) ----------------
    // Decide which mode governs insertion
    bool useB = modeB_enabled(set);

    uint8_t new_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_stream_lock = 0;

    if (type == ACCESS_PREFETCH) {
        // Prefetches always at tail; quarantine streams
        new_rrpv = maxRRPV;
        new_stream_lock = (useB && is_stream) ? 1 : 0;
    } else {
        // Demand fill
        if (useB) {
            // Mode B: Stream-UAF
            uint32_t pidx = pc_index(PC);
            bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

            if (is_stream) {
                new_stream_lock = 1;
                new_rrpv = pc_hot ? 5 : maxRRPV; // hot streamers get shallow tail
            } else {
                // Non-stream: LFU-hot near-MRU; cold near-tail
                new_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        } else {
            // Mode A: Hawkeye-like SHiP insertion by PC friendliness
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t score = shct_demand[idx]; // demand governs demand fills
            bool friendly = (score >= (SHCT_MAX / 2));
            new_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_lock = 0;
        }
    }

    // Initialize per-line state
    rrpv_set(set, way, new_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = new_stream_lock;
}

void PrintStats_Heartbeat() {}
void PrintStats() {}