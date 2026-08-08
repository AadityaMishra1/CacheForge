#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types aligned to ChampSim CRC2
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// Replacement constants
static constexpr uint8_t maxRRPV             = 7; // 3-bit SRRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7; // aggressive near-bypass
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5; // short-reuse escape (zeusmp-friendly)
static constexpr uint8_t STREAM_CONF_THRESH  = 2; // two forward strides to become "stream"
static constexpr uint8_t HITS_PROMOTE_HOT    = 2; // promote on 2nd demand hit when PC-hot
static constexpr uint8_t HITS_PROMOTE_COLD   = 3; // otherwise 3rd demand hit
static constexpr uint8_t PC_USE_HOT_THRESH   = 5; // TinyLFU 3b: >=5 considered hot
static constexpr uint8_t SHCT_WARM_THRESH    = 12; // 5b counter >=12 -> friendly
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // per-set bandit threshold
static constexpr int32_t GLOBAL_B_MARGIN     = 8;  // global margin to allow Mode B

// Set hashing for leader sampling
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // Stream-Lifetime
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Per-line metadata (packed conceptually: rrpv:3b, hitcnt:2b)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS]; // demand-hit counter (0..3)

// Per-set selector (bandit with hysteresis)
static int8_t bandit_score[LLC_SETS]; // [-8..7]

// Global selector (leaders duel)
static int32_t leaderA_good = 0; // hits - misses on A leaders
static int32_t leaderB_good = 0; // hits - misses on B leaders
static uint64_t access_count = 0;

// SHCT: tiny Hawkeye-style predictor
#define SHCT_SIZE (1u << 10) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Leader-A training state (64 sampled sets * 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // store 12b effective (in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // prefetch fill (0/1)
static uint8_t  leader_valid[64][LLC_WAYS]; // valid line in leader (0/1)

// Mode B: PC stream detector (+1/+2 run) and TinyLFU 3b
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b effective
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use3[PC_TBL_SIZE];        // 3b TinyLFU (0..7)
static uint32_t lfu_decay_ptr = 0;           // rotating decay pointer

// Helpers
static inline void sat_inc_u5(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }

// +1/+2 forward-run detector with 2-run confidence; reset on non-forward
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

static inline bool modeB_enabled(uint32_t set) {
    // Global wins must favor Mode B and local bandit must be confident
    bool global_ok = (leaderB_good > (leaderA_good + GLOBAL_B_MARGIN));
    return global_ok && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Initialize replacement state
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
    // Lightly warm SHCT to avoid over-pessimism
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 1;
        shct_prefetch[i] = 0;
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(leader_valid, 0, sizeof(leader_valid));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use3, 0, sizeof(pc_use3));

    leaderA_good = 0;
    leaderB_good = 0;
    access_count = 0;
    lfu_decay_ptr = 0;
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
    // SRRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all ways (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Unreachable
    // return 0;
}

// Update replacement state
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

    access_count++;

    // Periodic TinyLFU decay: one entry every 8 accesses (O(1))
    if ((access_count & 7ull) == 0ull) {
        pc_use3[lfu_decay_ptr] = (uint8_t)(pc_use3[lfu_decay_ptr] >> 1);
        lfu_decay_ptr = (lfu_decay_ptr + 1) & (PC_TBL_SIZE - 1u);
    }

    // Leader-set global goodness tracking
    if (LEADER_A(set)) {
        leaderA_good += (hit ? 1 : -1);
    } else if (LEADER_B(set)) {
        leaderB_good += (hit ? 1 : -1);
    }

    // Hysteresis: slowly decay local bandit toward 0
    if ((access_count & 0x3FFull) == 0ull) {
        if (bandit_score[set] > 0) bandit_score[set]--;
        else if (bandit_score[set] < 0) bandit_score[set]++;
    }

    // Common indices/state
    uint32_t pc_idx = pc_index(PC);
    uint16_t sig12  = pc_sig12(PC);
    uint32_t sh_idx = shct_idx(sig12);

    if (hit) {
        // Mark reuse in leader-A for training
        if (SEL_SAMPLED(set) && LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // On demand hits, multi-hit promotion; prefetch hits do not count
        if (is_demand(type)) {
            // TinyLFU credit for demand
            if (pc_use3[pc_idx] < 7) pc_use3[pc_idx]++;

            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            bool pc_hot = (pc_use3[pc_idx] >= PC_USE_HOT_THRESH) ||
                          (shct_demand[sh_idx] >= SHCT_WARM_THRESH);
            uint8_t need_hits = pc_hot ? HITS_PROMOTE_HOT : HITS_PROMOTE_COLD;

            if (hitcnt[set][way] >= need_hits) {
                rrpv_set(set, way, 0); // MRU on 2nd/3rd demand hit
                // Local bandit learns that this set exhibits reuse (discourage stream mode)
                sat_dec_i8(bandit_score[set]);
            } else {
                // Gentle promotion toward MRU
                rrpv_promote(set, way);
            }
        }
        return;
    }

    // MISS path: train Hawkeye leaders and insert according to selected mode
    if (SEL_SAMPLED(set) && LEADER_A(set)) {
        // Train previous occupant on eviction from this way
        uint32_t slot = LEADER_SLOT(set);
        if (leader_valid[slot][way]) {
            uint16_t ev_sig = hawk_sig[slot][way];
            uint32_t ev_idx = shct_idx(ev_sig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) sat_inc_u5(shct_prefetch[ev_idx]);
                else                       sat_dec_u5(shct_prefetch[ev_idx]);
            } else {
                if (hawk_used[slot][way]) sat_inc_u5(shct_demand[ev_idx]);
                else                       sat_dec_u5(shct_demand[ev_idx]);
            }
        }
        // Record new occupant
        hawk_sig[slot][way]     = sig12;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]    = 0;
        leader_valid[slot][way] = 1;
    }

    // Update stream detector and TinyLFU
    bool stream_conf = detect_and_update_stream(PC, paddr);
    if (is_demand(type)) {
        if (pc_use3[pc_idx] < 7) pc_use3[pc_idx]++;
    }

    // Local bandit reinforcement for streaming behavior
    if (stream_conf) {
        sat_inc_i8(bandit_score[set]);
    }

    // Decide insertion depth
    uint8_t depth = INSERT_COLD_DEPTH;

    if (is_writeback(type)) {
        depth = INSERT_WARM_DEPTH; // never bypass writebacks
    } else if (is_prefetch(type)) {
        depth = STREAM_TAIL_DEPTH; // quarantine prefetches at tail
    } else {
        bool use_modeB = modeB_enabled(set) || LEADER_B(set);
        bool use_modeA = (!use_modeB) || LEADER_A(set);

        if (use_modeB) {
            if (stream_conf) {
                // Aggressive stream handling; shallow-tail if PC-hot for short reuse
                depth = (pc_use3[pc_idx] >= PC_USE_HOT_THRESH) ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            } else {
                bool pc_hot = (pc_use3[pc_idx] >= PC_USE_HOT_THRESH) ||
                              (shct_demand[sh_idx] >= SHCT_WARM_THRESH);
                depth = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        } else if (use_modeA) {
            // Hawkeye-like SHiP insertion
            uint8_t ctr = shct_demand[sh_idx];
            depth = (ctr >= SHCT_WARM_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    rrpv_set(set, way, depth);
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