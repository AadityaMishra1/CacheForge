#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
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
// Tunables (exploration knobs)
// ============================================================================
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU

// Selector tunables
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096;  // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 3;          // conceptual 2-bit confidence

// Stream detector tunables
static const uint8_t RUN_LEN_TO_STREAM = 2;    // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ = 2;      // confidence to declare stream

// Promotion gate tunables
static const uint8_t HITS_TO_PROMOTE = 2;  // demand hits to promote stream/quarantine-tagged lines

// TinyLFU tunables
static const uint8_t FREQ_THRESHOLD = 6;       // PC must reach this to be "hot"
static const uint32_t FREQ_AGE_PERIOD = 8192;  // aging period

// ============================================================================
// Global selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
// Conceptually 2 bits per set (packed); stored as int8_t for code simplicity
static int8_t mode_confidence[LLC_SETS];

// ============================================================================
// Per-line metadata (conceptually bit-packed; stored as bytes in code)
// ============================================================================
static uint8_t rrpv[LLC_SETS][LLC_WAYS];          // 3-bit conceptual
static uint8_t line_hits[LLC_SETS][LLC_WAYS];     // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];   // 1-bit conceptual

// ============================================================================
// Mode A: Hawkeye-like (SHCT + SRRIP). Always trained; safe fallback.
// ============================================================================
#define SHCT_ENTRIES 2048
static uint8_t SHCT[SHCT_ENTRIES]; // conceptual 5-bit counters
static const uint8_t SHCT_MAX = 31;
static const uint8_t SHCT_INC = 1;
static const uint8_t SHCT_DEC = 1;
// Friendliness threshold (tunable)
static const uint8_t SHCT_FRIENDLY_TH = 8;

static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (SHCT_ENTRIES - 1); }

static inline bool hawkeye_predict_friendly(uint64_t PC) {
    return SHCT[pc_index(PC)] >= SHCT_FRIENDLY_TH;
}

static inline void hawkeye_train(uint64_t PC, uint8_t hit) {
    // Train PC-based friendliness predictor
    uint32_t idx = pc_index(PC);
    if (hit) {
        SHCT[idx] = (SHCT[idx] < SHCT_MAX) ? (SHCT[idx] + SHCT_INC) : SHCT_MAX;
    } else {
        SHCT[idx] = (SHCT[idx] >= SHCT_DEC) ? (uint8_t)(SHCT[idx] - SHCT_DEC) : 0;
    }
}

static inline uint32_t hawkeye_victim(uint32_t set) {
    // Prefer max-RRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == RRPV_MAX) return i;
    }
    // If none, age until one appears (SRRIP)
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] == RRPV_MAX) return i;
        }
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < RRPV_MAX) rrpv[set][i]++;
        }
    }
}

static inline void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type) {
    // Prefetch always quarantined to the tail
    if (type == PREFETCH) {
        rrpv[set][way] = RRPV_MAX;
        return;
    }
    // WRITEBACK: never bypass, modest priority
    if (type == WRITEBACK) {
        rrpv[set][way] = RRPV_DISTANT;
        return;
    }
    // Demand: friendly PCs inserted higher, averse lower
    if (hawkeye_predict_friendly(PC)) {
        rrpv[set][way] = RRPV_PROTECT;
    } else {
        rrpv[set][way] = RRPV_DISTANT;
    }
}

static inline void hawkeye_on_hit(uint32_t set, uint32_t way) {
    // SRRIP hit promotion (not full MRU)
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}

// ============================================================================
// Mode B: StrideGuard + TinyLFU (stream/scan detector + frequency filter)
// ============================================================================
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3 (conceptual 2-bit)
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7 (conceptual 3-bit)

#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE]; // conceptual 4-bit

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
static inline bool detect_stream(uint64_t PC, uint64_t line_addr) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t line = (uint16_t)(line_addr & 0xFFFFu);
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
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
    }

    pc_last_line[idx] = line;
    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

static inline uint32_t modeB_victim(uint32_t set) {
    // Standard SRRIP victim selection
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == RRPV_MAX) return i;
    }
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] == RRPV_MAX) return i;
        }
        for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv[set][i] < RRPV_MAX) rrpv[set][i]++;
    }
}

static inline bool modeB_should_bypass(uint32_t type, uint64_t PC, uint64_t line_addr) {
    if (type == WRITEBACK) return false; // never bypass on writeback
    bool stream = detect_stream(PC, line_addr);
    bool hot = freq_is_hot(PC);
    // Demand: bypass noisy forward streams unless PC is hot
    return (type != PREFETCH) && stream && !hot;
}

static inline void modeB_insertion(uint32_t set, uint32_t way, uint32_t type, uint64_t PC, uint64_t line_addr) {
    if (type == WRITEBACK) {
        rrpv[set][way] = RRPV_DISTANT;
        line_stream_tag[set][way] = 0;
        line_pref_q[set][way] = 0;
        line_hits[set][way] = 0;
        return;
    }

    bool stream = detect_stream(PC, line_addr);
    bool hot = freq_is_hot(PC);

    if (type == PREFETCH) {
        // Prefetch quarantine
        rrpv[set][way] = RRPV_MAX;
        line_pref_q[set][way] = 1;
        line_stream_tag[set][way] = stream ? 1 : 0;
        line_hits[set][way] = 0;
        return;
    }

    // Demand
    line_pref_q[set][way] = 0;
    line_hits[set][way] = 0;
    if (stream && !hot) {
        // Hard-tail insert for forward streams (if not bypassed by caller)
        rrpv[set][way] = RRPV_MAX;
        line_stream_tag[set][way] = 1;
    } else if (hot) {
        // Hot PCs get protection
        rrpv[set][way] = RRPV_PROTECT;
        line_stream_tag[set][way] = 0;
    } else {
        // Default distant insertion
        rrpv[set][way] = RRPV_DISTANT;
        line_stream_tag[set][way] = stream ? 1 : 0;
    }
}

static inline void modeB_on_hit(uint32_t set, uint32_t way, uint32_t type) {
    // Generic SRRIP hit promotion
    if (rrpv[set][way] > 0) rrpv[set][way]--;

    // Track per-line short reuse
    if (line_hits[set][way] < 3) line_hits[set][way]++;

    // Prefetch quarantine escapes on first demand hit
    if (type != PREFETCH && line_pref_q[set][way]) {
        line_pref_q[set][way] = 0;
        rrpv[set][way] = (rrpv[set][way] > RRPV_PROTECT) ? RRPV_PROTECT : rrpv[set][way];
    }

    // Stream-tagged lines escape after a few hits
    if (line_stream_tag[set][way] && line_hits[set][way] >= HITS_TO_PROMOTE) {
        rrpv[set][way] = 0; // strong promotion to rescue short reuse
        line_stream_tag[set][way] = 0;
    }
}

// ============================================================================
// Selector Policy
// ============================================================================
static void reset_selector_counters() {
    leaderA_score = 0;
    leaderB_score = 0;
}

static void drift_confidence_all(bool toward_B) {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (SAMPLED_SET(s)) continue; // don't push leaders
        if (toward_B) {
            if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
        } else {
            if (mode_confidence[s] > 0) mode_confidence[s]--;
        }
    }
}

static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;

    // Update frequency aging and any global timers
    periodic_freq_age();

    // For leader sets, tally performance
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    // Periodically re-evaluate preference and drift followers
    if ((access_count % EVAL_PERIOD) == 0) {
        bool new_prefer_B = (leaderB_score > (leaderA_score + BIAS));
        prefer_mode_B = new_prefer_B;
        drift_confidence_all(prefer_mode_B);
        reset_selector_counters();
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Force Mode A
    if (LEADER_B(set)) return true;   // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// ChampSim-required API
// ============================================================================
void InitReplacementState()
{
    // Initialize per-line state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
            line_hits[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
        mode_confidence[s] = 0;
    }

    // Initialize Mode A predictor
    memset(SHCT, 0, sizeof(SHCT));

    // Initialize Mode B tables
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    memset(pc_freq, 0, sizeof(pc_freq));

    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;

    std::cout << "Initialize H+SGL (Hawkeye baseline + StrideGuard-LFU Mode B) state" << std::endl;
}

// Find victim in the set
// Return 0..15 or 16 (bypass). Never bypass on WRITEBACK.
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    (void)cpu;
    // Prefer invalid way if available (correctly handle empties)
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);
    uint64_t line_addr = (paddr >> 6);

    // Mode B can optionally bypass on demand streams (never on writeback)
    if (use_mode_B && modeB_should_bypass(type, PC, line_addr)) {
        return 16; // bypass
    }

    // Choose victim per active mode
    if (use_mode_B) {
        return modeB_victim(set);
    } else {
        return hawkeye_victim(set);
    }
}

// Update replacement state on every access/fill
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

    // Always update selector accounting first
    update_selector(set, hit);

    // Always train Mode A PC predictor (safety net)
    hawkeye_train(PC, hit);

    // Frequency tracker for Mode B (PC hotness)
    freq_on_access(PC);

    // Ignore writebacks for predictor training decisions beyond insertion priority
    // but still must not bypass on WRITEBACK per spec.
    uint64_t line_addr = (paddr >> 6);

    bool use_mode_B = should_use_mode_B(set);

    // On hit: perform promotions
    if (hit) {
        // Common SRRIP promotion for both modes
        hawkeye_on_hit(set, way);

        // Mode B extra behaviors
        if (use_mode_B) {
            modeB_on_hit(set, way, type);
        }
        return;
    }

    // Miss/fill path
    // If bypassed earlier, ChampSim passes way == LLC_WAYS (16). Guard actions accordingly.
    if (way >= LLC_WAYS) {
        // We chose bypass (only for non-WB). Still trained Mode A above.
        return;
    }

    // Reset per-line aux state on allocation (for Mode B)
    line_hits[set][way] = 0;
    line_stream_tag[set][way] = 0;
    line_pref_q[set][way] = 0;

    if (use_mode_B) {
        modeB_insertion(set, way, type, PC, line_addr);
    } else {
        hawkeye_insertion(set, way, PC, type);
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank for eval harness
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank for eval harness
}