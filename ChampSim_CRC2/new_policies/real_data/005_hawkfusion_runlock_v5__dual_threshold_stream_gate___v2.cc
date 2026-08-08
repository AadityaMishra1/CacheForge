#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2-compatible)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)   { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t) { return (t == ACCESS_PREFETCH); }

// ---------------- Leader-set sampling (64 leaders via 6-bit folding) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (retuned for lbm/zeusmp + irregulars) --------------------
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP (0..7)
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU insertion depth
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail insertion depth
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 forward-run confidence needed
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream: escape quarantine on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote stream-locked on each touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 5;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // periodic decay of TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 8192; // selector epoch in accesses
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 20;   // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // slot-level bandit enable threshold
static constexpr uint8_t  SHCT_FRIENDLY_THRES = 8;    // Hawkeye insertion threshold

// ---------------- Per-line metadata (conceptually bit-packed) -----------------------
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Selector state ----------------------------------------------------
static int8_t   bandit_slot[64];       // [-8..7], slot-level bias for Mode B
static uint8_t  GSEL = 0;              // global gate (0..31)
static uint8_t  leader_hits_A[64];     // per-slot hit counters for leader-A sets
static uint8_t  leader_hits_B[64];     // per-slot hit counters for leader-B sets
static uint64_t access_ctr = 0;        // for epoching/decay

// ---------------- Mode A: Hawkeye-like SHiP fallback -------------------------------
// Tiny SHCT (demand only)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // low-cost mix for 12-bit signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Per-line training only for leader sets (64 x 16)
static uint16_t hawk_sig[64][LLC_WAYS];   // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];  // saw reuse (0/1) via demand hit

// ---------------- Mode B: RunLock Stream Shield + TinyLFU --------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b line id (stored in 16b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// ---------------- Helpers -----------------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    uint32_t slot = LEADER_SLOT(set);
    return (GSEL >= GSEL_THRES) && (bandit_slot[slot] >= MODEB_ENABLE_THRESH);
}

static inline void selector_epoch_try_update() {
    access_ctr++;
    // Periodic TinyLFU/coldness decay
    if ((access_ctr % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
        }
    }
    // Selector epoch
    if ((access_ctr % BANDIT_EPOCH) == 0) {
        uint32_t sumA = 0, sumB = 0;
        for (uint32_t i = 0; i < 64; i++) {
            sumA += leader_hits_A[i];
            sumB += leader_hits_B[i];
            if (leader_hits_B[i] > leader_hits_A[i]) sat_inc_i8(bandit_slot[i]);
            else if (leader_hits_A[i] > leader_hits_B[i]) sat_dec_i8(bandit_slot[i]);
            leader_hits_A[i] = 0;
            leader_hits_B[i] = 0;
        }
        if (sumB > sumA) { if (GSEL < GSEL_MAX) GSEL++; }
        else if (sumA > sumB) { if (GSEL > 0) GSEL--; }
    }
}

void InitReplacementState() {
    std::memset(rrpv, 0, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start cold
        }
    }
    std::memset(bandit_slot, 0, sizeof(bandit_slot));
    std::memset(leader_hits_A, 0, sizeof(leader_hits_A));
    std::memset(leader_hits_B, 0, sizeof(leader_hits_B));
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    access_ctr = 0;
}

// SRRIP victim with invalid check and safe aging
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP search with bounded aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Update replacement state on hit/fill
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
    selector_epoch_try_update();

    // Per-PC stats update
    uint32_t pidx = pc_index(PC);
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pidx]); // credit demand uses
        if (hit) {
            if (pc_cold2[pidx] > 0) pc_cold2[pidx]--; // observed reuse -> less cold
        } else {
            if (pc_cold2[pidx] < 3) pc_cold2[pidx]++; // miss -> colder
        }
    }

    bool modeB = use_modeB(set);
    bool stream_seen = detect_and_update_stream(PC, paddr);

    // Leader bookkeeping for selector and Hawkeye training
    if (hit) {
        if (LEADER_A(set)) leader_hits_A[LEADER_SLOT(set)]++;
        if (LEADER_B(set)) leader_hits_B[LEADER_SLOT(set)]++;
        // Mark reuse for Hawkeye training only in leader sets
        if (SEL_SAMPLED(set)) {
            hawk_used[LEADER_SLOT(set)][way] = 1;
        }
    }

    if (hit) {
        // Hit handling: multi-hit promotion with stream-aware demotion
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                // Stream-locked: demote on touch, escape only after 2 demand hits
                for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;       // escape quarantine
                    rrpv_set(set, way, INSERT_WARM_DEPTH); // lift near MRU
                }
            } else {
                // Non-stream: only promote on 2nd+ demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU
                }
            }
        } else {
            // Prefetch touch: never promote
            if (stream_lock[set][way]) {
                for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);
            }
        }
        return;
    }

    // Miss / Fill path: train Hawkeye on eviction in leader sets
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train with the evicted line that lived in (set,way)
        uint16_t sig = hawk_sig[slot][way];
        uint32_t idx = shct_idx(sig);
        if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
        else                      shct_dec(shct_demand[idx]);
        // Prepare per-line training state for the new line
        hawk_sig[slot][way]  = pc_sig12(PC);
        hawk_used[slot][way] = 0;
    }

    // Decide insertion policy
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t make_stream_lock = 0;

    if (!modeB) {
        // Mode A: Hawkeye-like insertion from SHCT
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t ctr = shct_demand[idx];
        if (ctr >= SHCT_FRIENDLY_THRES) ins_rrpv = INSERT_WARM_DEPTH;
        else                            ins_rrpv = INSERT_COLD_DEPTH;
        // Prefetches always cold
        if (is_prefetch(type)) ins_rrpv = maxRRPV;
    } else {
        // Mode B: RunLock stream shield + TinyLFU/coldness
        bool hot_pc  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        bool cold_pc = (pc_cold2[pidx] >= 2) || !hot_pc;

        if (stream_seen) {
            // Dual-threshold stream gate: hard-tail insert; "bypass" via RRIP=7
            ins_rrpv = maxRRPV;
            make_stream_lock = 1; // quarantine until proven reuse
        } else {
            // Non-stream: adaptive insertion by PC heat/coldness
            ins_rrpv = cold_pc ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
        }
        // Prefetches quarantined aggressively
        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV;
            make_stream_lock = 1;
        }
    }

    // Never bypass on WRITEBACK; for others, hard-tail acts as soft-bypass
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = make_stream_lock ? 1 : 0;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}