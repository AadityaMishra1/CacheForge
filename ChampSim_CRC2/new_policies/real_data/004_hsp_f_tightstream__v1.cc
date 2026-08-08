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
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // reserved (unused here, keep for safety)

// Stream detector (+1/+2 forward run)
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // after 2 forward steps: arm

// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS       = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 3; // stream: MRU on 3rd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT  = 2; // TinyLFU-hot stream: MRU on 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH   = 1; // demote quarantined streams on touch

// Hawkeye-lite SHCT
static constexpr uint8_t  SHCT_FRIENDLY_THRESH = 18; // >= => friendly
#define SHCT_SIZE (1u << 10)  // 1024
#define SHCT_MAX 31

// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 2048; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 4;    // global bias toward Hawkeye-lite

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
    uint16_t cur = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = false;
    if (last != 0xFFFFu) {
        int32_t d = (int32_t)cur - (int32_t)last;
        fwd = (d == 1) || (d == 2);
    }
    if (fwd) sat_inc_u2(pc_stream_conf[idx]);
    else     sat_dec_u2(pc_stream_conf[idx]);
    pc_last_line10[idx] = cur;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled_for_set(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    if (!prefer_B) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        // neutral-ish start, still averse to meet tighter friendly threshold
        shct_demand[i]   = SHCT_FRIENDLY_THRESH > 2 ? (SHCT_FRIENDLY_THRESH - 2) : SHCT_FRIENDLY_THRESH;
        shct_prefetch[i] = SHCT_FRIENDLY_THRESH > 2 ? (SHCT_FRIENDLY_THRESH - 2) : SHCT_FRIENDLY_THRESH;
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i]        = 0;
    }
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w]     = 0;
            hawk_used[ls][w]    = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim selection: look for maxRRPV; age until found
    for (int iter = 0; iter < 8; iter++) { // bounded to avoid infinite loops
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // Fallback: return way 0
    return 0;
}

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

    // Epoch/decay housekeeping
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = (leaderB_score > (leaderA_score + GATE_MARGIN));
        leaderA_score = 0;
        leaderB_score = 0;
    }
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        // periodic decay to reduce noise
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;
        }
    }

    // Update selector statistics for leaders
    if (LEADER_A(set)) {
        leaderA_score += (hit ? 1 : -1);
    } else if (LEADER_B(set)) {
        leaderB_score += (hit ? 1 : -1);
    } else {
        // followers: fast reversion if global gate off
        if (!prefer_B) sat_dec_i8(bandit_score[set]);
    }

    // TinyLFU update on demand accesses
    uint32_t pidx = pc_index(PC, paddr);
    if (is_demand(type)) sat_inc_u4(pc_use4[pidx]);
    bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    // Mode selection
    bool use_modeB = modeB_enabled_for_set(set);

    // Leader training: on hit, mark reuse; on miss, train evicted line and store new signature info
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hit) {
            hawk_used[slot][way] = 1;
        } else {
            // train the evicted line for leaders only
            uint16_t esig = hawk_sig[slot][way] & 0x0FFFu;
            uint32_t sidx = shct_idx(esig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[sidx]);
                else                      shct_dec(shct_prefetch[sidx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[sidx]);
                else                      shct_dec(shct_demand[sidx]);
            }
            // install new metadata for the incoming line
            hawk_sig[slot][way]     = pc_sig12(PC);
            hawk_used[slot][way]    = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    }

    // Stream detector: only meaningful for non-writeback accesses
    bool stream_armed = false;
    if (!is_writeback(type)) {
        stream_armed = detect_and_update_stream(PC, paddr);
    }

    // Per-access policy
    if (hit) {
        // Enforce multi-hit gating and scan resistance
        if (is_demand(type)) {
            // Increment demand hitcount (saturates)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            uint8_t thr = HITS_PROMOTE_NS;
            if (stream_tag[set][way]) {
                thr = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
            }
            if (hitcnt[set][way] >= thr) {
                rrpv_set(set, way, 0); // promote to MRU
            } else {
                // quarantined streams: demote on early touches
                if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) {
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                }
            }
        }
        // Do not promote on prefetch hits
        return;
    }

    // Miss: choose insertion policy (never bypass on WRITEBACK)
    hitcnt[set][way] = 0;

    if (is_writeback(type)) {
        stream_tag[set][way] = 0;
        rrpv_set(set, way, INSERT_COLD_DEPTH);
        return;
    }

    // Prefetches: deep quarantine at tail
    if (is_prefetch(type)) {
        stream_tag[set][way] = 1;
        rrpv_set(set, way, STREAM_TAIL_DEPTH);
        return;
    }

    // Demand miss insertion
    uint16_t sig12 = pc_sig12(PC);
    uint32_t sidx  = shct_idx(sig12);

    if (!use_modeB) {
        // Mode A (Hawkeye-lite): friendly near-MRU, averse near-tail
        bool friendly = (shct_demand[sidx] >= SHCT_FRIENDLY_THRESH);
        stream_tag[set][way] = 0;
        rrpv_set(set, way, friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
    } else {
        // Mode B (StreamPhase-Guard)
        if (stream_armed) {
            // streaming quarantine: hard tail
            stream_tag[set][way] = 1;
            rrpv_set(set, way, STREAM_TAIL_DEPTH);
        } else {
            // PC cold bypass (SHCT-averse and not TinyLFU-hot): push to hard tail
            bool pc_cold = (shct_demand[sidx] + (pc_hot ? 2 : 0)) < SHCT_FRIENDLY_THRESH;
            if (pc_cold && !pc_hot) {
                stream_tag[set][way] = 0;
                rrpv_set(set, way, STREAM_TAIL_DEPTH);
            } else {
                // TinyLFU-gated depth
                stream_tag[set][way] = 0;
                rrpv_set(set, way, pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
            }
        }
    }

    // Follower per-set bandit score (only when global prefers B)
    if (!SEL_SAMPLED(set) && prefer_B) {
        if (use_modeB) {
            if (is_demand(type)) {
                // Reward/penalize Mode B based on observed outcome (miss happened now)
                sat_dec_i8(bandit_score[set]);
            }
        } else {
            // If we stayed in A while global prefers B and we still missed, nudge toward B slightly
            sat_inc_i8(bandit_score[set]);
        }
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}