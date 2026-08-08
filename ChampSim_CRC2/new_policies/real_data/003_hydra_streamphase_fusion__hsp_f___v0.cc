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

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // force Mode A
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // force Mode B
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for sampled sets

// ---------------- Tunables ---------------------------------------
// RRIP depths (3-bit)
static constexpr uint8_t maxRRPV             = 7;
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail (quarantine)
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for hot-PC streams

// Stream detector (+1/+2 forward run)
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // after 2 fwd steps: arm

// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS      = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR     = 2; // stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT = 1; // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH  = 1; // demote quarantined streams on touch

// Hawkeye-lite SHCT
static constexpr uint8_t  SHCT_FRIENDLY_THRESH = 16; // 0..31 (>= => friendly)
#define SHCT_SIZE (1u << 10)  // 1024
#define SHCT_MAX 31

// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Hawkeye-lite

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt(demand hits):2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Per-set selector --------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode A unless confident
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_score = 0;      // hits - misses on leader A sets
static int32_t leaderB_score = 0;      // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-lite via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (trained only in leaders)
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b xor-folded hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamPhase-Guard PC state) --------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc, uint64_t paddr) {
    // phase-mixed: PC xor region (page-ish) for better phase/object sensitivity
    uint64_t mix = (pc >> 1) ^ (pc >> 7) ^ (paddr >> 12);
    return (uint32_t)mix & (PC_TBL_SIZE - 1u);
}
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC, paddr);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv)); // start aged
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_tag, 0, sizeof(stream_tag));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t s = 0; s < 64; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[s][w] = 0;
            hawk_used[s][w] = 0;
            hawk_is_pref[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i]        = 0;
    }
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count  = 0;
}

// ---------------- Victim selection --------------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP selection: age until a line reaches maxRRPV
    for (int iter = 0; iter < 8; iter++) { // guaranteed to terminate within 8 steps
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // Fallback (should not happen): choose way with highest RRPV
    uint32_t best = 0, best_rrpv = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best_rrpv) { best_rrpv = rrpv[set][w]; best = w; }
    }
    return best;
}

// ---------------- State update ------------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Writebacks never change policy state
    if (is_writeback(type)) return;

    // Demand-side TinyLFU update + periodic decay
    access_count++;
    if (is_demand(type)) {
        uint32_t idx = pc_index(PC, paddr);
        sat_inc_u4(pc_use4[idx]);
    }
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Leader accounting for selector
    if (LEADER_A(set)) {
        leaderA_score += (hit ? 1 : -1);
    } else if (LEADER_B(set)) {
        leaderB_score += (hit ? 1 : -1);
    } else {
        // Followers: adjust per-set bandit based on local outcome and current mode choice
        if (modeB_enabled(set)) {
            if (hit) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
        } else {
            // Mild reinforcement for A as default: penalize misses, small reward for hits
            if (hit) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
        }
    }
    // Epoch end: update global gate with bias toward Mode A
    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = ((leaderB_score - leaderA_score) > GATE_MARGIN);
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Multi-hit promotion and stream quarantine handling on hits
    if (hit) {
        if (is_demand(type)) {
            // Demote quarantined streams slightly on touch to keep them in quarantine
            if (STREAM_DEMOTE_TOUCH && stream_tag[set][way] && (rrpv[set][way] < maxRRPV)) {
                sat_inc_u3(rrpv[set][way]); // demote by +1 toward tail
            }
            // Multi-hit gate
            uint32_t idx = pc_index(PC, paddr);
            bool hot = (pc_use4[idx] >= PC_USE_HOT_THRESH);
            uint8_t need = stream_tag[set][way] ? (hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR) : HITS_PROMOTE_NS;

            if (hitcnt[set][way] < 3) hitcnt[set][way]++; // saturating
            if (hitcnt[set][way] + 0 /* counts after this hit */ >= need) {
                rrpv_set(set, way, 0); // MRU
            }
        }
        return;
    }

    // Miss/Fill path below
    // Leader training for Hawkeye-lite (train on the line being evicted in sampled sets)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train the to-be-evicted line in this way (before we overwrite the per-line buffers)
        uint16_t old_sig = hawk_sig[slot][way];
        uint8_t  old_used = hawk_used[slot][way];
        uint8_t  old_pref = hawk_is_pref[slot][way];

        if (old_sig) {
            uint32_t idx = shct_idx(old_sig);
            if (old_used) {
                if (old_pref) shct_inc(shct_prefetch[idx]);
                else          shct_inc(shct_demand[idx]);
            } else {
                if (old_pref) shct_dec(shct_prefetch[idx]);
                else          shct_dec(shct_demand[idx]);
            }
        }
        // Reset training buffers for this way; will set new below
        hawk_sig[slot][way] = 0;
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = 0;
    }

    // Choose mode for insertion behavior
    bool use_modeB = modeB_enabled(set);

    // Compute stream and hotness
    uint32_t lidx = pc_index(PC, paddr);
    bool is_stream = detect_and_update_stream(PC, paddr);
    bool pc_hot    = (pc_use4[lidx] >= PC_USE_HOT_THRESH);

    // Insertion policies
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t is_stream_line = 0;

    if (use_modeB) {
        if (is_prefetch(type)) {
            ins_depth = STREAM_TAIL_DEPTH;    // quarantine prefetches
            is_stream_line = 1;
        } else if (is_stream) {
            ins_depth = pc_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            is_stream_line = 1;
        } else {
            // Non-stream: TinyLFU-gated insertion
            ins_depth = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    } else {
        // Mode A: Hawkeye-lite insertion from SHCT (trained only on leaders)
        uint16_t sig = pc_sig12(PC);
        uint32_t sidx = shct_idx(sig);
        uint8_t conf = is_prefetch(type) ? shct_prefetch[sidx] : shct_demand[sidx];
        bool friendly = (conf >= SHCT_FRIENDLY_THRESH);
        ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        is_stream_line = 0;
    }

    // Apply insertion: never bypass writebacks (already returned), prefetches insert at tail
    if (is_prefetch(type)) {
        rrpv_set(set, way, STREAM_TAIL_DEPTH);
        stream_tag[set][way] = 1;
        hitcnt[set][way] = 0;
    } else {
        rrpv_set(set, way, ins_depth);
        stream_tag[set][way] = is_stream_line ? 1 : 0;
        hitcnt[set][way] = 0;
    }

    // Record Hawkeye-lite training info for sampled sets
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0; // will be set on a future demand hit
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}