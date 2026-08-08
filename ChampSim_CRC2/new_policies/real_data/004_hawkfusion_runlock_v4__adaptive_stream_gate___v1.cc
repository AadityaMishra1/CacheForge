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
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU insertion depth
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail insertion depth
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 forward-run confidence
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream: escape quarantine on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote stream-locked on each touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 5;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // periodic decay of TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 8192; // selector epoch (ops)
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
static int8_t  bandit_slot[64];          // [-8..7], slot-level bias for Mode B
static uint8_t GSEL = 0;                 // global gate (0..31)
static uint16_t leader_hits_A[64];       // per-slot hit counters for leader-A sets
static uint16_t leader_hits_B[64];       // per-slot hit counters for leader-B sets

// ---------------- Mode A: Hawkeye-like SHiP fallback -------------------------------
// Tiny SHCTs (demand & prefetch)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // low-cost mix for 12-bit signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Per-line training only for leader sets (64 x 16)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1) via demand hit
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

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

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // Mode B on B-leaders
    if (LEADER_A(set)) return false;   // Mode A on A-leaders
    // Followers: enable Mode B only if both global and slot-level suggest it
    uint32_t slot = LEADER_SLOT(set);
    return (GSEL >= GSEL_THRES) && (bandit_slot[slot] >= MODEB_ENABLE_THRESH);
}

// ---------------- Global clocks -----------------------------------------------------
static uint64_t access_count = 0;

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    // Light positive prior
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 1;
        shct_prefetch[i] = 1;
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    std::memset(bandit_slot, 0, sizeof(bandit_slot));
    std::memset(leader_hits_A, 0, sizeof(leader_hits_A));
    std::memset(leader_hits_B, 0, sizeof(leader_hits_B));
    GSEL = 0;
    access_count = 0;
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
    // RRIP victim selection
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines (saturate at maxRRPV)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
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
    access_count++;

    // Per-PC maintenance
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
        sat_dec_u2(pc_cold2[pc_index(PC)]); // demand hits/misses will reduce coldness over time
    }
    if (type != ACCESS_WRITEBACK) {
        // Update run detector on all non-WB accesses
        (void)detect_and_update_stream(PC, paddr);
    }

    // Selector accounting (leader hits)
    if (hit) {
        if (LEADER_A(set)) leader_hits_A[LEADER_SLOT(set)]++;
        else if (LEADER_B(set)) leader_hits_B[LEADER_SLOT(set)]++;
    }

    // Periodic decay for TinyLFU/coldness and selector epoch update
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
            sat_dec_u2(pc_cold2[i]);
        }
    }
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Compare per-slot leaders and update global/slot gates
        for (uint32_t s = 0; s < 64; s++) {
            if (leader_hits_B[s] > leader_hits_A[s]) {
                if (GSEL < GSEL_MAX) GSEL++;
                sat_inc_i8(bandit_slot[s]);
            } else if (leader_hits_B[s] < leader_hits_A[s]) {
                if (GSEL > 0) GSEL--;
                sat_dec_i8(bandit_slot[s]);
            }
            leader_hits_A[s] = leader_hits_B[s] = 0;
        }
    }

    // Hit path: multi-hit promotion and stream demotion
    if (hit) {
        // Stream resistance: demote on every touch if currently stream-locked
        if (stream_lock[set][way]) {
            for (uint8_t k = 0; k < STREAM_DEMOTE_TOUCH; k++) rrpv_demote(set, way);
        }
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                rrpv_set(set, way, 0); // promote to MRU on 2nd demand hit
                if (stream_lock[set][way]) {
                    // Short-reuse escape for stream lines
                    stream_lock[set][way] = 0;
                }
            }
        }
        // Hawkeye reuse training: mark used on leader sets upon a demand hit
        if (SEL_SAMPLED(set) && is_demand(type)) {
            hawk_used[LEADER_SLOT(set)][way] = 1;
        }
        return;
    }

    // Miss path: prepare insertion policy (train victim if leader)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train previous occupant (victim) of 'way'
        uint16_t vsig = hawk_sig[slot][way];
        uint8_t used  = hawk_used[slot][way];
        uint8_t was_pref = hawk_is_pref[slot][way];
        if (vsig) {
            uint32_t idx = shct_idx(vsig);
            if (was_pref)
                (used ? shct_inc(shct_prefetch[idx]) : shct_dec(shct_prefetch[idx]));
            else
                (used ? shct_inc(shct_demand[idx]) : shct_dec(shct_demand[idx]));
        }
        // Install new training context for the incoming line
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }

    // Decide which mode to apply for insertion
    bool useB = modeB_enabled(set);

    // Insertion RRIP depth and stream lock
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t ins_stream = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert cold
        ins_rrpv = INSERT_COLD_DEPTH;
        ins_stream = 0;
    } else if (is_prefetch(type)) {
        // Prefetch quarantine at hard tail; never first-hit promoted
        ins_rrpv = maxRRPV;
        ins_stream = 1; // keep demotion-on-touch behavior for safety
    } else { // Demand miss
        if (useB) {
            bool stream_now = detect_and_update_stream(PC, paddr);
            if (stream_now) {
                // Aggressive shield for scans/streams
                ins_rrpv = maxRRPV;
                ins_stream = 1;
            } else {
                // PC heat and coldness guide insertion
                uint32_t pidx = pc_index(PC);
                bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
                bool very_cold = (pc_cold2[pidx] >= 2);
                if (very_cold)       ins_rrpv = maxRRPV;
                else if (hot)        ins_rrpv = INSERT_WARM_DEPTH;
                else                 ins_rrpv = INSERT_COLD_DEPTH;
                ins_stream = 0;
            }
        } else {
            // Hawkeye-style friendly/averse insertion
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t conf = shct_demand[idx];
            bool friendly = (conf > SHCT_FRIENDLY_THRES);
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            ins_stream = 0;
        }
    }

    // Apply insertion for the filled line
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;              // never promote on first hit
    stream_lock[set][way] = ins_stream;
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank
}