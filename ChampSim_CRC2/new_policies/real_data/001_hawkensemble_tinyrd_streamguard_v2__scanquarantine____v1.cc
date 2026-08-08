#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

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

// ---------------- Tunables (retuned) ----------------
static constexpr uint8_t  maxRRPV             = 7;     // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;     // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;     // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;     // +1/+2 forward steps needed
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;     // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 3;     // stream escape on 3rd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 5;     // TinyLFU hot PCs (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;  // periodic decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;  // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;    // global gate range
static constexpr uint8_t  GSEL_THRES          = 20;    // followers choose Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 3;     // per-slot enable threshold

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b (also used to quarantine prefetch)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set tiny reuse-distance bias ---------------
static int8_t rd_bias[LLC_SETS]; // [-8..7], >0 => insert warmer; <0 => insert colder

// ---------------- Selector: per-slot bandit + global gate --------
static int8_t   bandit_score_slot[64];     // [-8..7]
static uint16_t leader_hits_A_slot[64];    // epoch counters
static uint16_t leader_hits_B_slot[64];
static uint8_t  GSEL = 0;                  // global preference 0..31

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

// ---------------- Mode B (TinyRD-StreamGuard v2) -----------------
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
    // +1/+2 forward run detector with hysteresis (decay on break)
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (fwd) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        if (pc_stream_conf[idx] > 0) pc_stream_conf[idx]--; // hysteresis
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
static inline void rrpv_inc(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static uint32_t op_count = 0;
static inline void periodic_housekeeping() {
    op_count++;
    // Decay TinyLFU and deadness (lightweight) periodically
    if ((op_count & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;            // halve LFU
            if (pc_dead2[i] > 0) pc_dead2[i]--; // mild forgiveness
        }
    }
    // Selector epoch update
    if ((op_count & (BANDIT_EPOCH - 1)) == 0) {
        int32_t sumA = 0, sumB = 0;
        for (uint32_t s = 0; s < 64; s++) {
            if (leader_hits_B_slot[s] > leader_hits_A_slot[s]) {
                sat_inc_i8(bandit_score_slot[s]);
            } else if (leader_hits_B_slot[s] < leader_hits_A_slot[s]) {
                sat_dec_i8(bandit_score_slot[s]);
            }
            sumA += leader_hits_A_slot[s];
            sumB += leader_hits_B_slot[s];
            leader_hits_A_slot[s] = 0;
            leader_hits_B_slot[s] = 0;
        }
        if (sumB > sumA) {
            if (GSEL < GSEL_MAX) GSEL++;
        } else if (sumB < sumA) {
            if (GSEL > 0) GSEL--;
        }
    }
}

// ================= Interface functions ===========================

// Initialize replacement state
void InitReplacementState() {
    // Per-line state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        rd_bias[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    // Mode A (Hawkeye-like)
    for (uint32_t i = 0; i < SHCT_SIZE; i++) shct[i] = SHCT_MAX / 2; // neutral
    for (uint32_t s = 0; s < 64; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig_slot[s][w]  = 0;
            hawk_used_slot[s][w] = 0;
            hawk_pref_slot[s][w] = 0;
        }
    }
    // Mode B PC tables
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_dead2, 0, sizeof(pc_dead2));

    // Selector
    std::memset(bandit_score_slot, 0, sizeof(bandit_score_slot));
    std::memset(leader_hits_A_slot, 0, sizeof(leader_hits_A_slot));
    std::memset(leader_hits_B_slot, 0, sizeof(leader_hits_B_slot));
    GSEL = 0;
    op_count = 0;
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
    // SRRIP: search for RRPV==max; if none, increment all and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] >= maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv_inc(set, w);
        }
    }
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
    periodic_housekeeping();

    // Update PC tables early (observations)
    bool is_stream = detect_and_update_stream(PC, paddr);
    uint32_t pidx = pc_index(PC);
    if (hit) {
        if (is_demand(type)) {
            sat_inc_u4(pc_use4[pidx]);     // credit useful PCs
            if (pc_dead2[pidx] > 0) pc_dead2[pidx]--; // alive signal
        }
    } else {
        if (is_demand(type)) sat_inc_u2(pc_dead2[pidx]); // missing PC tends to be dead/cold
    }

    // Leader-set accounting for selector (count demand hits)
    if (hit && is_demand(type) && SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (LEADER_A(set)) leader_hits_A_slot[slot]++;
        else if (LEADER_B(set)) leader_hits_B_slot[slot]++;
    }

    // On HIT: gated promotion + scan demote
    if (hit) {
        // Update per-set RD bias from hit depth
        if (rrpv[set][way] <= 2) sat_inc_i8(rd_bias[set]);
        else if (rrpv[set][way] >= 6) sat_dec_i8(rd_bias[set]);

        if (stream_lock[set][way]) {
            // Quarantined (stream/prefetch): demote-on-touch; allow escape after threshold demand hits
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++; // count demands only
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0; // escape quarantine
                    hitcnt[set][way] = HITS_PROMOTE_STR;
                    rrpv_set(set, way, 0);     // now promote to MRU
                } else {
                    rrpv_inc(set, way);        // keep it old
                }
            } else {
                rrpv_inc(set, way);            // prefetch touch stays cold
            }
        } else {
            // Normal line: promote only on 2nd+ demand hit
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU
                } else {
                    // keep it warm but not MRU
                    if (rrpv[set][way] == 0) rrpv_set(set, way, 1);
                }
            } else {
                // prefetch hit: keep modest
                if (rrpv[set][way] == 0) rrpv_set(set, way, 1);
            }
        }

        // Mode A training: mark reuse on A leaders
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used_slot[slot][way] = 1;
        }
        return;
    }

    // On MISS/FILL: choose mode and insertion depth
    bool use_modeB = modeB_enabled(set);

    // Mode A: Hawkeye-like SHiP insertion and training on leaders
    if (!use_modeB) {
        uint16_t sig = pc_sig12(PC);
        uint8_t pred = shct[shct_idx(sig)];
        uint8_t ins = INSERT_COLD_DEPTH; // default cold
        if (is_prefetch(type)) {
            ins = maxRRPV; // quarantine prefetch
        } else if (pred >= (SHCT_MAX / 2)) {
            ins = INSERT_WARM_DEPTH;
        } else {
            ins = INSERT_COLD_DEPTH;
        }
        rrpv_set(set, way, ins);
        hitcnt[set][way] = 0;
        stream_lock[set][way] = is_prefetch(type) ? 1u : 0u;

        // Train only on A leaders: update old occupant outcome and store new signature
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            // Train previous signature in this slot/way
            uint16_t old_sig = hawk_sig_slot[slot][way];
            uint8_t& ctr = shct[shct_idx(old_sig)];
            if (hawk_used_slot[slot][way]) shct_inc(ctr);
            else                           shct_dec(ctr);

            // Record new
            hawk_sig_slot[slot][way]  = sig;
            hawk_used_slot[slot][way] = 0;
            hawk_pref_slot[slot][way] = is_prefetch(type) ? 1u : 0u;
        }
        return;
    }

    // Mode B: TinyRD-StreamGuard v2 insertion
    uint8_t ins = INSERT_COLD_DEPTH;
    bool cold_pc   = (pc_use4[pidx] < PC_USE_HOT_THRESH);
    bool deadish   = (pc_dead2[pidx] >= 1);
    bool strong_stream = (pc_stream_conf[pidx] >= (STREAM_CONF_THRESH + 1));

    if (is_prefetch(type)) {
        ins = maxRRPV;               // strong quarantine for prefetches
        stream_lock[set][way] = 1u;  // quarantine flag
    } else if (is_stream) {
        // Streaming scan: aggressive tail insertion (approximate bypass)
        if (cold_pc || deadish || strong_stream) {
            ins = maxRRPV;
        } else {
            ins = INSERT_COLD_DEPTH; // allow rare reuse to escape quicker
        }
        stream_lock[set][way] = 1u;  // never promote until multiple demand hits
    } else {
        // Non-stream: PC utility + deadness + per-set RD bias
        if (!deadish && !cold_pc) {
            ins = INSERT_WARM_DEPTH; // warm PCs, predicted useful
        } else {
            ins = INSERT_COLD_DEPTH;
        }
        // Per-set RD bias: warm if many short-depth hits; cool if tail hits dominate
        if (rd_bias[set] >= 2 && ins > 0) ins--;
        else if (rd_bias[set] <= -2 && ins < maxRRPV) ins++;
        stream_lock[set][way] = 0u;
    }

    rrpv_set(set, way, ins);
    hitcnt[set][way] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // intentionally blank
}