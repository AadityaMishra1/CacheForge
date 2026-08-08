#include <vector>
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
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail quarantine

// Stream detector (+1/+2 forward run)
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // after 2 forward steps: arm

// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS       = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 3; // stream: MRU on 3rd demand hit (default)
static constexpr uint8_t HITS_PROMOTE_STR_HOT  = 2; // TinyLFU-hot stream: MRU on 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH   = 1; // demote quarantined streams on touch

// Hawkeye-lite SHCT
static constexpr uint8_t  SHCT_FRIENDLY_THRESH = 18; // >= => friendly
#define SHCT_SIZE (1u << 10)  // 1024
#define SHCT_MAX 31

// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates (retuned)
static constexpr uint32_t BANDIT_EPOCH        = 4096; // longer epoch: more stable gate
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence to enable B
static constexpr int32_t  GATE_MARGIN         = 6;    // +6 bias toward Hawkeye-lite

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt(demand hits):2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode A unless confident
static bool    prefer_B = false;        // global gate from leaders
static int32_t leaderA_score = 0;       // demand hits - misses on leader A sets
static int32_t leaderB_score = 0;       // demand hits - misses on leader B sets
static uint64_t access_count = 0;       // also drives epoching and LFU decay

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
// 512-entry PC tables: last line (10b), valid (1b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc, uint64_t paddr) {
    // phase-mixed: PC xor region for better phase/object sensitivity
    uint64_t mix = (pc >> 1) ^ (pc >> 7) ^ (paddr >> 12);
    return (uint32_t)mix & (PC_TBL_SIZE - 1u);
}
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_last_valid[PC_TBL_SIZE];
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
    if (pc_last_valid[idx]) {
        int32_t delta = (int32_t)cur - (int32_t)pc_last_line10[idx];
        if (delta == 1 || delta == 2) sat_inc_u2(pc_stream_conf[idx]);
        else sat_dec_u2(pc_stream_conf[idx]);
    } else {
        pc_last_valid[idx] = 1;
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = cur;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Required interface ------------------------------

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
    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[i][w] = 0;
            hawk_used[i][w] = 0;
            hawk_is_pref[i][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0;
        pc_last_valid[i]  = 0;
        pc_stream_conf[i] = 0;
        pc_use4[i]        = 0;
    }
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count  = 0;
}

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
    // SRRIP victim selection with bounded aging
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // Fallback (should not happen)
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

    // Global accounting and epochs
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        // Global TinyLFU decay by half
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Global gate: Mode B preferred only if it beats A by > GATE_MARGIN
        prefer_B = ((leaderB_score - leaderA_score) > GATE_MARGIN);
        leaderA_score = 0;
        leaderB_score = 0;
        // Light decay of per-set bandits toward 0
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (bandit_score[s] > 0) bandit_score[s]--;
            else if (bandit_score[s] < 0) bandit_score[s]++;
        }
    }

    const bool mb = modeB_enabled(set);
    const bool sampledA = LEADER_A(set);
    const bool sampledB = LEADER_B(set);

    // Leader tallies: demand-only hits/misses
    if (SEL_SAMPLED(set) && is_demand(type)) {
        if (sampledA) leaderA_score += (hit ? 1 : -1);
        else if (sampledB) leaderB_score += (hit ? 1 : -1);
    }

    // Follower bandit: reward/punish Mode B performance (demand only)
    if (!SEL_SAMPLED(set) && is_demand(type)) {
        if (mb) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else { sat_dec_i8(bandit_score[set]); sat_dec_i8(bandit_score[set]); } // punish more on misses
        } else {
            // When not using B, gently decay
            if (bandit_score[set] > -1) ; // no-op; bias stays near A unless B excels
        }
    }

    // TinyLFU update on any demand access
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC, paddr);
        sat_inc_u4(pc_use4[pidx]);
    }

    if (hit) {
        // Mark reuse for Hawkeye leaders
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Stream detection feedback (kept up-to-date in both modes)
        (void)detect_and_update_stream(PC, paddr);

        // Multi-hit gated promotion
        if (is_demand(type)) {
            uint8_t prev = hitcnt[set][way];
            if (prev < 3) hitcnt[set][way] = prev + 1;

            bool was_stream = (stream_tag[set][way] != 0);
            uint32_t pidx = pc_index(PC, paddr);
            bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            uint8_t need = was_stream ? (hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR) : HITS_PROMOTE_NS;

            if (hitcnt[set][way] >= need) {
                rrpv_set(set, way, 0); // MRU
                if (was_stream) stream_tag[set][way] = 0; // LiteReuse escape
            } else {
                // Stream touches demote if still quarantined
                if (was_stream) {
                    uint8_t newv = std::min<uint8_t>((uint8_t)maxRRPV, (uint8_t)(rrpv[set][way] + STREAM_DEMOTE_TOUCH));
                    rrpv_set(set, way, newv);
                }
            }
        }
        // no promotion for prefetch hits (quarantine holds)
        return;
    }

    // Miss path: train Hawkeye-lite on eviction in leader sets
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t osig = hawk_sig[slot][way];
        uint32_t hidx = shct_idx(osig);
        if (hawk_is_pref[slot][way]) {
            if (hawk_used[slot][way]) shct_inc(shct_prefetch[hidx]);
            else                      shct_dec(shct_prefetch[hidx]);
        } else {
            if (hawk_used[slot][way]) shct_inc(shct_demand[hidx]);
            else                      shct_dec(shct_demand[hidx]);
        }
    }

    // Decide insertion policy
    uint8_t ins_rrip = INSERT_COLD_DEPTH;
    uint8_t ins_stream = 0;

    if (is_writeback(type)) {
        // Never bypass writebacks; keep tail-biased
        ins_rrip = INSERT_COLD_DEPTH;
        ins_stream = 0;
    } else if (mb) {
        // Mode B: Stream guard + TinyLFU + PC-SHCT cold-bypass
        bool armed = detect_and_update_stream(PC, paddr);
        uint32_t pidx = pc_index(PC, paddr);
        bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        uint16_t sig = pc_sig12(PC);
        uint32_t sidx = shct_idx(sig);
        bool pc_cold = (shct_demand[sidx] < SHCT_FRIENDLY_THRESH);

        if (is_prefetch(type)) {
            ins_rrip = STREAM_TAIL_DEPTH; // quarantine all prefetches
            ins_stream = armed ? 1 : 0;
        } else {
            if (armed) {
                ins_rrip = STREAM_TAIL_DEPTH; // emulate bypass
                ins_stream = 1;
            } else if (pc_cold && !hot) {
                ins_rrip = STREAM_TAIL_DEPTH; // PC-cold bypass
                ins_stream = 0;
            } else if (hot) {
                ins_rrip = INSERT_WARM_DEPTH;
                ins_stream = 0;
            } else {
                ins_rrip = INSERT_COLD_DEPTH;
                ins_stream = 0;
            }
        }
    } else {
        // Mode A: Hawkeye-lite (use demand SHCT for insertion distance)
        uint16_t sig = pc_sig12(PC);
        uint32_t sidx = shct_idx(sig);
        bool friendly = (shct_demand[sidx] >= SHCT_FRIENDLY_THRESH);
        ins_rrip = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        ins_stream = 0;
        // Keep stream detector warm in background
        (void)detect_and_update_stream(PC, paddr);
    }

    // Perform insertion
    rrpv_set(set, way, ins_rrip);
    stream_tag[set][way] = ins_stream;
    hitcnt[set][way] = 0;

    // Record new Hawkeye training info for leaders
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way] = 0;
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}