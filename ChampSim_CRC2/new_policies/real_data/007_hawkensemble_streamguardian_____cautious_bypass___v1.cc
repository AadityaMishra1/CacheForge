#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type helpers
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// Sets: 2048 -> 11 bits
static constexpr uint32_t LLC_SET_BITS = 11;

// Leader set sampling: 64 leaders
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets using a simple signature: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Tunables (retuned per feedback)
static constexpr uint8_t  maxRRPV               = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;    // near MRU
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;    // near tail
static constexpr uint8_t  PREFETCH_TAIL         = 7;    // quarantine
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;    // need 2 forward steps
static constexpr uint8_t  HITS_PROMOTE_MRU      = 2;    // promote only on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_ON_TOUCH= 1;    // demote stream-locked lines on touch
static constexpr uint8_t  PC_USE_HOT_THRESH     = 7;    // TinyLFU hot if >=
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024; // faster decay
static constexpr uint32_t BANDIT_EPOCH          = 2048; // shorter selector epoch
static constexpr uint8_t  GSEL_MAX              = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES            = 20;   // followers prefer B if GSEL >= THRES
static constexpr int8_t   MODEB_ENABLE_THRESH   = 3;    // per-set score threshold
static constexpr uint8_t  ZEUS_ASSIST_DEPTH     = 3;    // shallow insert for hot streams

// Per-line packed state (conceptual): rrpv:3b, hitcnt:2b, stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// Per-set selectors
static int8_t  bandit_score[LLC_SETS];     // -8..+7 saturating
static uint8_t stream_evidence[LLC_SETS];  // 2-bit recent stream evidence (0..3)

// Global selector
static uint8_t  GSEL = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;
static uint64_t access_count = 0;

// Mode A (Hawkeye-like SHiP)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit counters
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit counters

// Leader-A per-line signature/flags (64 sets * 16 ways = 1024 lines)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // 1b reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1b prefetched

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Mode B (StreamGuardian+++) PC tables (smaller = less noise)
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b stored in 16b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness

// Helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t v) {
    rrpv[set][way] = (v > maxRRPV) ? maxRRPV : v;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

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
    if (LEADER_B(set)) return true;     // force B on leader-B
    if (LEADER_A(set)) return false;    // force A on leader-A
    // Followers: global + local + recent stream evidence
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH) && (stream_evidence[set] > 0);
}

static inline void epoch_tasks() {
    // Global selector update with margin
    if (leaderB_hits_epoch > leaderA_hits_epoch + 8) {
        if (GSEL < GSEL_MAX) GSEL++;
    } else if (leaderA_hits_epoch > leaderB_hits_epoch + 8) {
        if (GSEL > 0) GSEL--;
    }
    leaderA_hits_epoch = leaderB_hits_epoch = 0;

    // Decay TinyLFU and coldness
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_use4[i] >>= 1; // fast decay
        sat_dec_u2(pc_cold2[i]);
    }
    // Per-set evidence and bandit gentle decay
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (stream_evidence[s] > 0) stream_evidence[s]--;
        if (bandit_score[s] > 0) bandit_score[s]--;
        else if (bandit_score[s] < 0) bandit_score[s]++;
    }
}

// Initialize replacement state
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(stream_evidence, 0, sizeof(stream_evidence));

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    GSEL = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
    access_count = 0;
}

// Find victim in the set (SRRIP)
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Immediate allocation to invalid way if available
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection with aging
    for (;;) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Update replacement state
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Epoch maintenance
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) epoch_tasks();

    // Mode decision (used for fill path)
    bool use_modeB = modeB_enabled(set);

    // STREAM detection and LFU update on every access
    bool stream_seen = detect_and_update_stream(PC, paddr);
    if (stream_seen) {
        if (stream_evidence[set] < 3) stream_evidence[set]++;
        sat_inc_i8(bandit_score[set]);
    } else {
        sat_dec_i8(bandit_score[set]);
    }

    uint32_t pc_idx = pc_index(PC);
    if (hit) {
        // Hit path
        if (is_demand(type)) {
            if (pc_use4[pc_idx] < 15) pc_use4[pc_idx]++;
            sat_dec_u2(pc_cold2[pc_idx]);
        }

        // Count demand hits toward multi-hit gate
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        }
        // Stream demotion on touch unless escaping via 2nd demand hit
        if (stream_lock[set][way] && hitcnt[set][way] < HITS_PROMOTE_MRU && STREAM_DEMOTE_ON_TOUCH) {
            rrpv_demote(set, way);
        }
        // Promote only on 2nd demand hit
        if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_MRU) {
            rrpv_set(set, way, 0);
            stream_lock[set][way] = 0; // escape quarantine
        } else if (!stream_lock[set][way]) {
            // gentle promotion for non-stream-locked lines
            rrpv_promote(set, way);
        }

        // Leader hit accounting
        if (LEADER_A(set)) leaderA_hits_epoch++;
        if (LEADER_B(set)) leaderB_hits_epoch++;

        // SHiP reuse mark for leader-A
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // Miss / Fill path
    // Train Hawkeye on leader-A eviction (the line being replaced)
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig = hawk_sig[slot][way];
        uint32_t idx = shct_idx(sig);
        if (hawk_is_pref[slot][way]) {
            if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
            else                     shct_dec(shct_prefetch[idx]);
        } else {
            if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
            else                     shct_dec(shct_demand[idx]);
        }
        // Install new leader-A metadata
        hawk_sig[slot][way]      = pc_sig12(PC);
        hawk_used[slot][way]     = 0;
        hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
    }

    // Decide insertion policy
    uint8_t ins_rrpv = INSERT_WARM_DEPTH;
    uint8_t lock_stream = 0;

    if (use_modeB) {
        // Mode B: Stream-guarded adaptive insertion
        bool hot_pc  = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);
        bool cold_pc = (pc_cold2[pc_idx] >= 2);

        if (is_prefetch(type)) {
            ins_rrpv = PREFETCH_TAIL;
            lock_stream = 1; // quarantine prefetches
        } else if (is_demand(type) && stream_seen) {
            // Aggressive stream quarantine; shallow insert only for hot PCs
            if (hot_pc) {
                ins_rrpv = ZEUS_ASSIST_DEPTH;
            } else {
                ins_rrpv = maxRRPV;
            }
            lock_stream = 1;
        } else {
            // Non-stream or writeback: bias by coldness/usefulness
            if (cold_pc)       ins_rrpv = INSERT_COLD_DEPTH;
            else               ins_rrpv = INSERT_WARM_DEPTH;
        }
    } else {
        // Mode A: Compact Hawkeye-like SHiP
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_prefetch(type)) {
            ins_rrpv = PREFETCH_TAIL;
            lock_stream = 1; // quarantine prefetches in Mode A too
        } else if (is_demand(type)) {
            uint8_t ctr = shct_demand[idx];
            ins_rrpv = (ctr >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        } else {
            // writebacks: do not bypass; insert cold
            ins_rrpv = INSERT_COLD_DEPTH;
        }
    }

    // Update per-PC coldness on miss/fill
    if (is_demand(type)) {
        sat_inc_u2(pc_cold2[pc_idx]);
    }

    // Install state
    rrpv_set(set, way, ins_rrpv);
    stream_lock[set][way] = lock_stream ? 1 : 0;
    hitcnt[set][way] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank
}