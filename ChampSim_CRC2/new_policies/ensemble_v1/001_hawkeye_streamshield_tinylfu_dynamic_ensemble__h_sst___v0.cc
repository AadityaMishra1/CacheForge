#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ============================================================================
// Leader-Follower Sampling (64 sampled sets)
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// Tunables (exposed for exploration)
// ============================================================================
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU

// Selector tunables
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const int32_t EVAL_PERIOD = 4096;   // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 8;

// Stream detector tunables
static const uint8_t STREAM_WINDOW = 2;    // accept +1/+2 strides
static const uint8_t RUN_LEN_TO_STREAM = 2;// 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ = 2;  // confidence to declare stream

// Promotion gate tunables
static const uint8_t HITS_TO_PROMOTE = 2;  // demand hits to promote stream-tagged lines

// TinyLFU tunables
static const uint8_t FREQ_THRESHOLD = 6;   // PC must reach this to be "hot"
static const uint32_t FREQ_AGE_PERIOD = 8192;

// ============================================================================
// Global selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];   // Per-set soft state (drifts)

// ============================================================================
// Mode A: Hawkeye-like (SHCT + SRRIP)
// ============================================================================
#define SHCT_ENTRIES 2048
static uint8_t SHCT[SHCT_ENTRIES]; // 5-bit in concept; stored in uint8
static const uint8_t SHCT_MAX = 31;
static const uint8_t SHCT_INC = 1;
static const uint8_t SHCT_DEC = 1;
static const uint8_t SHCT_FRIENDLY_TH = 8;

static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)(pc) & (SHCT_ENTRIES - 1); }

static inline bool hawkeye_predict_friendly(uint64_t PC) {
    return SHCT[pc_index(PC)] >= SHCT_FRIENDLY_TH;
}

static inline void hawkeye_train_on_access(uint64_t PC, uint8_t hit) {
    uint32_t idx = pc_index(PC);
    if (hit) {
        if (SHCT[idx] + SHCT_INC <= SHCT_MAX) SHCT[idx] += SHCT_INC;
        else SHCT[idx] = SHCT_MAX;
    } else {
        if (SHCT[idx] >= SHCT_DEC) SHCT[idx] -= SHCT_DEC;
        else SHCT[idx] = 0;
    }
}

// ============================================================================
// Mode B: StreamShield + TinyLFU
// ============================================================================
#define PC_TABLE_SIZE 1024
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7

#define PC_FREQ_SIZE 1024
static uint8_t pc_freq[PC_FREQ_SIZE]; // 4-bit conceptual

static inline uint32_t pc_tab_index(uint64_t PC) { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }
static inline uint32_t pc_freq_index(uint64_t PC) { return (uint32_t)PC & (PC_FREQ_SIZE - 1); }

static inline void freq_on_access(uint64_t PC) {
    uint32_t idx = pc_freq_index(PC);
    if (pc_freq[idx] < 15) pc_freq[idx]++;
}

static inline bool freq_is_hot(uint64_t PC) {
    return pc_freq[pc_freq_index(PC)] >= FREQ_THRESHOLD;
}

static inline void periodic_freq_age() {
    if ((access_count % FREQ_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_FREQ_SIZE; i++) pc_freq[i] >>= 1;
    }
}

// Stream detection: detect forward +1/+2 runs with small confidence
static inline bool detect_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t line = (uint16_t)(paddr_line & 0xFFFFu);
    uint16_t last = pc_last_line[idx];

    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(line - last);
        forward = (diff == 1) || (diff == 2);
    }

    if (forward) {
        if (pc_forward_run[idx] < 7) pc_forward_run[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_forward_run[idx] = 0;
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--; // mild decay
    }

    pc_last_line[idx] = line;

    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

// ============================================================================
// Per-line metadata (bit-packed in accounting; stored as bytes in code)
// ============================================================================
static uint8_t rrpv[LLC_SETS][LLC_WAYS];          // 3-bit conceptual
static uint8_t line_hits[LLC_SETS][LLC_WAYS];     // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];   // 1-bit conceptual

// ============================================================================
// Helpers
// ============================================================================
static inline bool is_writeback(uint32_t type) { return type == 3; }
static inline bool is_demand(uint32_t type) { return (type == 0) || (type == 1); }
static inline bool is_prefetch(uint32_t type) { return !is_writeback(type) && !is_demand(type); }

static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;
    int delta = hit ? 1 : -1;

    if (SAMPLED_SET(set)) {
        if (LEADER_A(set)) leaderA_score += delta;
        else if (LEADER_B(set)) leaderB_score += delta;
    }

    if ((access_count % EVAL_PERIOD) == 0) {
        // Global preference
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        // Soft drift for the last-touched set
        if (prefer_mode_B) {
            if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
        } else {
            if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
        }
        // mild decay to keep scores bounded
        leaderA_score /= 2;
        leaderB_score /= 2;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Force Mode A
    if (LEADER_B(set)) return true;   // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

static inline uint32_t find_rrip_victim(uint32_t set, bool prefer_streamy) {
    // SRRIP search; optionally prefer stream-tagged victims if available
    while (true) {
        int candidate = -1;
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] == RRPV_MAX) {
                if (!prefer_streamy) return way;
                if (line_stream_tag[set][way] || line_pref_q[set][way]) return way; // prefer stream/quarantine
                if (candidate == -1) candidate = way; // fallback if no stream-tag found
            }
        }
        if (candidate != -1) return (uint32_t)candidate;
        // Age all lines
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] < RRPV_MAX) rrpv[set][way]++;
        }
    }
}

static inline void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type) {
    // Insert based on SHCT prediction
    bool friendly = hawkeye_predict_friendly(PC);
    uint8_t ins_rrpv;
    if (friendly) ins_rrpv = 0;
    else ins_rrpv = RRPV_DISTANT;
    if (is_prefetch(type)) ins_rrpv = RRPV_MAX; // quarantine prefetches in baseline, too
    rrpv[set][way] = ins_rrpv;
}

static inline void modeB_insertion(uint32_t set, uint32_t way, uint64_t PC, uint64_t paddr, uint32_t type) {
    bool stream = detect_stream(PC, (paddr >> 6));
    bool hot = freq_is_hot(PC);

    uint8_t ins_rrpv = RRPV_DISTANT;
    if (is_prefetch(type)) {
        ins_rrpv = RRPV_MAX; // quarantine all prefetches
        line_pref_q[set][way] = 1;
    } else if (stream && !is_writeback(type)) {
        ins_rrpv = RRPV_MAX; // hard-tail streams (simulate bypass safely)
        line_stream_tag[set][way] = 1;
    } else if (!hot) {
        ins_rrpv = RRPV_DISTANT; // deprioritize cold PCs
    } else {
        ins_rrpv = RRPV_PROTECT; // moderately protect hot non-stream data
    }
    rrpv[set][way] = ins_rrpv;
}

static inline void on_hit_promote(uint32_t set, uint32_t way, uint32_t type, bool use_mode_B) {
    if (use_mode_B) {
        // Multi-hit gate for stream/quarantine lines
        if (is_demand(type)) {
            if (line_hits[set][way] < 3) line_hits[set][way]++;
        }
        if (line_stream_tag[set][way] || line_pref_q[set][way]) {
            if (line_hits[set][way] >= HITS_TO_PROMOTE) {
                rrpv[set][way] = 0;
                // Optional: clear stream tag after proven reuse
                line_stream_tag[set][way] = 0;
                line_pref_q[set][way] = 0;
            } else {
                // gentle nudge, but keep at a distance
                if (rrpv[set][way] > 1) rrpv[set][way] -= 1;
            }
            return;
        }
        // Normal line under Mode B -> promote to MRU
        rrpv[set][way] = 0;
    } else {
        // Mode A: promote on hit
        rrpv[set][way] = 0;
    }
}

// Initialize replacement state
void InitReplacementState() {
    std::memset(rrpv, RRPV_MAX, sizeof(rrpv));
    std::memset(line_hits, 0, sizeof(line_hits));
    std::memset(line_stream_tag, 0, sizeof(line_stream_tag));
    std::memset(line_pref_q, 0, sizeof(line_pref_q));
    std::memset(SHCT, 0, sizeof(SHCT));
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
    access_count = 0;

    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    std::memset(pc_freq, 0, sizeof(pc_freq));
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
    // Bypass is not used due to API constraints (never bypass on WRITEBACK)
    // 1) Find invalid way first
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) {
            return way;
        }
    }

    // 2) Choose mode and pick victim
    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        return find_rrip_victim(set, /*prefer_streamy=*/true);
    } else {
        return find_rrip_victim(set, /*prefer_streamy=*/false);
    }
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
    // Selector accounting
    update_selector(set, hit);
    // Always keep frequency and stream tables up-to-date
    freq_on_access(PC);
    periodic_freq_age();
    (void)detect_stream(PC, (paddr >> 6)); // update internal state; ignore return here

    bool use_mode_B = should_use_mode_B(set);

    // Always train Mode A (Hawkeye-like) predictor regardless of active mode
    hawkeye_train_on_access(PC, hit);

    if (hit) {
        // On hit: promotion logic
        on_hit_promote(set, way, type, use_mode_B);
    } else {
        // On miss/fill: initialize per-line metadata and insert according to chosen mode
        line_hits[set][way] = 0;
        line_stream_tag[set][way] = 0;
        line_pref_q[set][way] = 0;

        if (use_mode_B) {
            modeB_insertion(set, way, PC, paddr, type);
        } else {
            hawkeye_insertion(set, way, PC, type);
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