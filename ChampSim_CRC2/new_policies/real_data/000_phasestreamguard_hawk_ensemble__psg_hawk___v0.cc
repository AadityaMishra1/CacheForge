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
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (balanced for streaming + irregular) ---
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for TinyLFU-hot streams

static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm stream

static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream: MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream: MRU at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined stream on touch

static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay TinyLFU periodically

static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Mode A (Hawkeye-like)

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt(demand hits):2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch (1) vs normal (0)

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] per-set confidence for Mode B
static bool    prefer_B = false;       // global gate from leaders
static int32_t leaderA_good = 0;       // hits - misses on leader A sets
static int32_t leaderB_good = 0;       // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-lite via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
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
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective signature
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen on demand (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (Stream + TinyLFU + Region gate) ---------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3, forward-run strength
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// 2-bit per-page region-life (hashed 1K pages)
static constexpr uint32_t REG_TBL_SIZE = 1024;
static inline uint32_t region_index(uint64_t paddr) { return (uint32_t)((paddr >> 12) & (REG_TBL_SIZE - 1u)); }
static uint8_t region_life[REG_TBL_SIZE]; // 0..3

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x) { if (x > 0) x--; }
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
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
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
    std::memset(rrpv, maxRRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_tag, 0, sizeof(stream_tag));

    std::memset(bandit_score, 0, sizeof(bandit_score));
    prefer_B     = false;
    leaderA_good = 0;
    leaderB_good = 0;
    access_count = 0;

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    std::memset(region_life, 0, sizeof(region_life));
}

// ---------------- Victim selection --------------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid line immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Try to find a line at maxRRPV; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set); // saturates at maxRRPV, guarantees termination
    }
    // Unreachable
    return 0;
}

// ---------------- State update ------------------------------------
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

    // Ignore writebacks: never bypass on WB, and no training
    if (is_writeback(type)) return;

    access_count++;

    // Periodic TinyLFU decay
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Update TinyLFU and Stream detector on access
    uint32_t pc_idx = pc_index(PC);
    if (is_demand(type) || is_prefetch(type)) sat_inc_u4(pc_use4[pc_idx]);
    bool stream_armed = detect_and_update_stream(PC, paddr);

    // Region-life update
    uint32_t reg_idx = region_index(paddr);
    if (hit && is_demand(type)) sat_inc_u2(region_life[reg_idx]); // useful reuse
    if (!hit) sat_dec_u2(region_life[reg_idx]);                   // miss -> colder region

    // Leader bookkeeping for selector reward
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);

    // Hit handling: multi-hit promotion with stream quarantine
    if (hit) {
        // Only demand hits count toward promotion
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            // Decide promotion thresholds
            uint8_t need_hits = stream_tag[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;

            if (hitcnt[set][way] >= need_hits) {
                // Escape quarantine and promote to MRU
                stream_tag[set][way] = 0;     // escape stream quarantine after sufficient reuse
                rrpv_set(set, way, 0);
            } else {
                // Do not early-promote; for streams, gently demote to resist scans
                if (stream_tag[set][way]) {
                    uint8_t v = rrpv[set][way];
                    v = std::min<uint8_t>((uint8_t)(v + STREAM_DEMOTE_TOUCH), maxRRPV);
                    rrpv_set(set, way, v);
                }
            }
        } else {
            // Prefetch hit: keep at current depth, remain quarantined if tagged
        }

        // Leader training: mark reuse on demand only
        if (SEL_SAMPLED(set) && is_demand(type)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Selector rewards
        if (leaderA) leaderA_good += 1;
        if (leaderB) leaderB_good += 1;

        // Followers: bandit confidence update (reinforce current choice)
        if (!SEL_SAMPLED(set)) {
            if (modeB_enabled(set)) sat_inc_i8(bandit_score[set]);
            else                     sat_dec_i8(bandit_score[set]); // bias toward A
        }

        return;
    }

    // Miss/Fill path below
    // Determine which mode to use on this access
    bool use_modeB = modeB_enabled(set);

    // Mode B policy signals
    bool lfu_hot     = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);
    bool region_cold = (region_life[reg_idx] == 0);

    // Mode A policy signal (PC-friendly vs averse)
    uint16_t sig12 = pc_sig12(PC);
    uint32_t sidx  = shct_idx(sig12);
    uint8_t sh_val = is_prefetch(type) ? shct_prefetch[sidx] : shct_demand[sidx];
    bool pc_friendly = (sh_val >= 8); // tuned threshold (0..31)

    // Decide insertion depth and quarantine
    uint8_t ins_depth = INSERT_WARM_DEPTH;
    uint8_t quarantine = 0;

    if (is_prefetch(type)) {
        // Prefetch: always hard-tail quarantine
        ins_depth = STREAM_TAIL_DEPTH;
        quarantine = 1;
    } else if (use_modeB) {
        if (stream_armed) {
            // Stream armed: tail quarantine; allow shallow tail if PC is hot
            ins_depth = lfu_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            quarantine = 1;
        } else if (!lfu_hot || region_cold) {
            // Cold PC or cold region: near-tail
            ins_depth = INSERT_COLD_DEPTH;
            quarantine = 0;
        } else {
            ins_depth = INSERT_WARM_DEPTH;
            quarantine = 0;
        }
    } else {
        // Mode A (Hawkeye-like via SHCT)
        ins_depth = pc_friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        quarantine = 0;
    }

    // Initialize line metadata on fill
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way]   = 0;
    stream_tag[set][way] = quarantine ? 1 : 0;

    // Leader training for Mode A SHCT (only sampled sets)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train on the evicted resident of 'way'
        if (hawk_sig[slot][way] != 0 || hawk_used[slot][way] != 0 || hawk_is_pref[slot][way] != 0) {
            uint16_t vsig = hawk_sig[slot][way];
            uint32_t vidx = shct_idx(vsig);
            if (hawk_used[slot][way]) {
                if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[vidx]);
                else                          shct_inc(shct_demand[vidx]);
            } else {
                if (hawk_is_pref[slot][way]) shct_dec(shct_prefetch[vidx]);
                else                          shct_dec(shct_demand[vidx]);
            }
        }
        // Install new metadata
        hawk_sig[slot][way]     = sig12;
        hawk_used[slot][way]    = 0; // no reuse yet
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }

    // Selector rewards on miss
    if (leaderA) leaderA_good -= 1;
    if (leaderB) leaderB_good -= 1;

    // Followers: bandit confidence update
    if (!SEL_SAMPLED(set)) {
        if (use_modeB) sat_dec_i8(bandit_score[set]); // miss while using B -> penalize
        else           sat_inc_i8(bandit_score[set]); // miss while using A -> bias against B
    }

    // Epoch end: update global gate
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Prefer Mode B only if it clearly wins on leaders
        int32_t diff = (leaderB_good - leaderA_good);
        prefer_B = (diff > GATE_MARGIN);
        leaderA_good = 0;
        leaderB_good = 0;
    }
}

// ---------------- Stats -------------------------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}