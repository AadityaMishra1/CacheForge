#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// RRIP parameters and tuned thresholds
static constexpr uint8_t maxRRPV             = 7;  // 3-bit SRRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // friendly insert
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // unfriendly insert
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // aggressive near-bypass
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // zeusmp-friendly
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // two forward strides
static constexpr uint8_t HITS_PROMOTE_HOT    = 2;  // promote on 2nd demand hit if PC-hot
static constexpr uint8_t HITS_PROMOTE_COLD   = 3;  // promote on 3rd demand hit if PC-cold
static constexpr uint8_t PC_USE_HOT_THRESH   = 5;  // TinyLFU (3b) >=5 => hot
static constexpr uint8_t SHCT_WARM_THRESH    = 12; // Hawkeye friendly threshold

// Selector hysteresis
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // per-set bandit threshold
static constexpr int32_t GLOBAL_B_MARGIN     = 8;  // leader-B must exceed A by this margin
static constexpr uint32_t BANDIT_DECAY_EPOCH = 4096;
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;

// Set sampling for leaders (64 sets)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) { return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u)); }
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Per-line metadata (packed conceptually: rrpv:3b, hitcnt:2b)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS]; // demand-hit counter (0..3)

// Per-set bandit score for Mode B ([-8..7])
static int8_t bandit_score[LLC_SETS];

// Global leader duel score
static int32_t leaderA_good = 0;
static int32_t leaderB_good = 0;
static uint64_t access_count = 0;

// Tiny SHCT (Hawkeye-like), separate demand/prefetch
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Leader training state (64 sampled sets x 16 ways = 1024 entries)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (store in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // prefetch fill (0/1)
static uint8_t  leader_valid[64][LLC_WAYS]; // valid line in leader (0/1)

// Mode B PC stream detector (+1/+2 forward) and TinyLFU-3b
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line within 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b 0..3
static uint8_t  pc_use3[PC_TBL_SIZE];        // 3b 0..7
static uint32_t lfu_decay_ptr = 0;

// Helpers
static inline void sat_inc_u5(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }

// Simple 12b PC signature (xor-fold)
static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }

// +1/+2 forward-run detector with 2-run window; returns stream after update
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward1 = (ln == (uint16_t)(last + 1));
    bool forward2 = (ln == (uint16_t)(last + 2));
    if (forward1 || forward2) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_allowed(uint32_t set) {
    bool global_ok = (leaderB_good > (leaderA_good + GLOBAL_B_MARGIN));
    return global_ok && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Initialization
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    // Light warm-up to avoid over-pessimism at start
    for (uint32_t i = 0; i < SHCT_SIZE; i++) shct_demand[i] = 1;

    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(leader_valid, 0, sizeof(leader_valid));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use3, 0, sizeof(pc_use3));
    lfu_decay_ptr = 0;

    leaderA_good = 0;
    leaderB_good = 0;
    access_count = 0;
}

// Victim selection: SRRIP with invalid check
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Return an invalid way if any
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Find a line with RRPV == max; age if none
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Update replacement state: insertion, promotion, training, selection updates
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    access_count++;

    // TinyLFU periodic decay (one-slot per period)
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        uint32_t idx = lfu_decay_ptr & (PC_TBL_SIZE - 1u);
        if (pc_use3[idx] > 0) pc_use3[idx]--;
        lfu_decay_ptr = (lfu_decay_ptr + 1) & (PC_TBL_SIZE - 1u);
    }

    // Run detector update (used by Mode B and bandit)
    bool stream_now = detect_and_update_stream(PC, paddr);
    uint32_t pc_idx = pc_index(PC);
    // Demand accesses update TinyLFU
    if (is_demand(type)) sat_inc_u3(pc_use3[pc_idx]);

    // Leader-duel accounting
    if (LEADER_A(set)) {
        if (hit) leaderA_good++; else leaderA_good--;
    } else if (LEADER_B(set)) {
        if (hit) leaderB_good++; else leaderB_good--;
    }

    // Bandit update (learn only from stream contexts)
    if (stream_now) {
        if (modeB_allowed(set)) {
            if (hit) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
        } else {
            // Mode A active in a stream context; a miss suggests trying B
            if (!hit) sat_inc_i8(bandit_score[set]);
        }
    } else {
        // Non-stream: bias against B on hits to keep A as default
        if (hit) sat_dec_i8(bandit_score[set]);
    }
    // Periodic hysteresis decay toward 0
    if ((access_count % BANDIT_DECAY_EPOCH) == 0) {
        if (bandit_score[set] > 0) bandit_score[set]--;
        else if (bandit_score[set] < 0) bandit_score[set]++;
    }

    // Handle hits: strict multi-hit gate; prefetch hits do not promote
    if (hit) {
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            uint8_t threshold = (pc_use3[pc_idx] >= PC_USE_HOT_THRESH) ? HITS_PROMOTE_HOT : HITS_PROMOTE_COLD;
            if (hitcnt[set][way] >= threshold) {
                rrpv_set(set, way, 0); // MRU
            } else {
                rrpv_promote(set, way); // gentle one-step
            }
        }
        // Mark reuse for leader training if sampled
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // Miss path: potential eviction and new insertion
    // If sampled set, train SHCT on the evicted line
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (leader_valid[slot][way]) {
            uint16_t sig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) sat_inc_u5(shct_prefetch[idx]);
                else                        sat_dec_u5(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) sat_inc_u5(shct_demand[idx]);
                else                        sat_dec_u5(shct_demand[idx]);
            }
        }
        // Install new leader metadata
        leader_valid[slot][way] = 1;
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way] = 0;
    }

    // Decide insertion depth
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    if (is_prefetch(type)) {
        // Quarantine all prefetches
        ins_rrpv = STREAM_TAIL_DEPTH;
    } else {
        bool use_modeB = modeB_allowed(set) && stream_now;
        if (use_modeB) {
            // Stream-protect, allow short-reuse escape for hot PCs
            ins_rrpv = (pc_use3[pc_idx] >= PC_USE_HOT_THRESH) ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
        } else {
            // Hawkeye-like SHiP guidance
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t score = shct_demand[idx];
            ins_rrpv = (score >= SHCT_WARM_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    // Insert new line
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0; // reset multi-hit counter
}

void PrintStats() {}
void PrintStats_Heartbeat() {}