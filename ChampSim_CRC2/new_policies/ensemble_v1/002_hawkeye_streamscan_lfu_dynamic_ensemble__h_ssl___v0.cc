#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ============================================================================
// Leader-Follower Sampling (64 sampled sets) — fixed mapping
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET(uint32_t set) {
    // 64 sampled sets: use XOR folding of low/high 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// Tunables (Exploration knobs)
// ============================================================================
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU
static const uint8_t RRPV_MRU = 0;

// Selector tunables
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096;  // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 8;

// Stream detector tunables
static const uint8_t RUN_LEN_TO_STREAM = 2;   // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ = 2;     // confidence to declare stream

// Promotion gate tunables
static const uint8_t HITS_TO_PROMOTE = 2;     // demand hits to rescue stream-tagged lines

// TinyLFU tunables
static const uint8_t FREQ_THRESHOLD = 6;      // PC must reach this to be "hot"
static const uint32_t FREQ_AGE_PERIOD = 8192; // period to halve frequencies

// ============================================================================
// Global selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A (leader sets)
static int32_t leaderB_score = 0;  // Hits - misses on Mode B (leader sets)
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];   // Per-set drift toward preferred mode

// ============================================================================
// Mode A: Hawkeye-like (lightweight SHCT + SRRIP) — always trained
// NOTE: This is a storage-lean, Hawkeye-style baseline (PC-friendly insertion
// + SRRIP promotion). It preserves the "never-worse" property via leader dueling.
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
        if (SHCT[idx] < SHCT_MAX) SHCT[idx] += SHCT_INC;
    } else {
        if (SHCT[idx] > 0) SHCT[idx] -= SHCT_DEC;
    }
}

// ============================================================================
// Mode B: StreamScan + TinyLFU (PC-based forward-run streaming + freq admission)
// ============================================================================
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];     // last line index (16-bit)
static uint8_t  pc_stride_conf[PC_TABLE_SIZE];   // 2-bit conceptual
static uint8_t  pc_forward_run[PC_TABLE_SIZE];   // 3-bit conceptual

#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE];            // 4-bit conceptual

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

// Stream detection: detect forward +1/+2 runs with confidence and run-length
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
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
    }

    pc_last_line[idx] = line;
    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

// ============================================================================
// Per-line metadata (conceptual bit-pack; stored as bytes here for clarity)
// ============================================================================
static uint8_t rrpv[LLC_SETS][LLC_WAYS];            // 3-bit conceptual
static uint8_t line_hits[LLC_SETS][LLC_WAYS];       // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];     // 1-bit conceptual

// ============================================================================
// Helpers
// ============================================================================
static inline bool is_writeback(uint32_t type) { return (type == WRITEBACK); }
static inline bool is_prefetch(uint32_t type) { return (type == PREFETCH); }

static inline void bump_confidence_toward_global(uint32_t set) {
    if (SAMPLED_SET(set)) return; // leaders are forced; don't bias them
    if (prefer_mode_B) {
        if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
    } else {
        if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
    }
}

static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;
    // Leader accounting: score = hits - misses
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }
    // Global decision each period
    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        // drift follower sets toward global preference
        // (lazy drift: nudge on next access to each set)
        // Reset scores to avoid overflow
        leaderA_score = 0;
        leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// SRRIP-style victim search (optionally prefer stream/quarantine in Mode B)
static inline int find_rrpv_max_with_pref(uint32_t set, bool prefer_stream_quarantine) {
    int cand = -1;
    for (int32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == RRPV_MAX) {
            if (!prefer_stream_quarantine) return i;
            // Prefer lines marked as stream or quarantined prefetch
            if (line_pref_q[set][i] || line_stream_tag[set][i]) return i;
            if (cand == -1) cand = i; // fallback if none are tagged
        }
    }
    return cand;
}

static inline uint32_t pick_victim_rrip(uint32_t set, const BLOCK *current_set) {
    // Try to find invalid way
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) return way;
    }
    // Look for max RRPV
    int v = find_rrpv_max_with_pref(set, false);
    if (v >= 0) return (uint32_t)v;

    // Otherwise, choose the largest RRPV
    uint8_t maxr = 0;
    int lru_victim = -1;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] >= maxr) { maxr = rrpv[set][i]; lru_victim = i; }
    }
    assert(lru_victim != -1);
    return (uint32_t)lru_victim;
}

static inline uint32_t mode_B_victim(uint32_t set, const BLOCK *current_set) {
    // Prefer evicting quarantined prefetch or streaming-tagged lines at tail
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) return way;
    }
    int v = find_rrpv_max_with_pref(set, true);
    if (v >= 0) return (uint32_t)v;

    // Else fall back to RRIP choice
    return pick_victim_rrip(set, current_set);
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type, const BLOCK *current_set) {
    if (is_writeback(type)) return false; // never bypass writebacks
    // If invalid way exists, do not bypass (allocate instead)
    for (uint32_t way = 0; way < LLC_WAYS; way++) if (!current_set[way].valid) return false;

    uint64_t line = (paddr >> 6);
    bool stream = detect_stream(PC, line);
    bool hot = freq_is_hot(PC);

    // Bypass only for LFU-cold, streaming/scanning demand requests
    if (!is_prefetch(type) && stream && !hot) return true;
    return false;
}

// ============================================================================
// Mode insert/update policies
// ============================================================================
static inline void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type) {
    line_hits[set][way] = 0;
    line_stream_tag[set][way] = 0;
    if (is_prefetch(type)) {
        line_pref_q[set][way] = 1;
        rrpv[set][way] = RRPV_DISTANT; // quarantine prefetches
    } else {
        line_pref_q[set][way] = 0;
        if (hawkeye_predict_friendly(PC)) rrpv[set][way] = RRPV_PROTECT;
        else rrpv[set][way] = RRPV_DISTANT;
    }
}

static inline void mode_B_insertion(uint32_t set, uint32_t way, uint64_t PC, uint64_t paddr, uint32_t type) {
    line_hits[set][way] = 0;
    uint64_t line = (paddr >> 6);
    bool stream = detect_stream(PC, line);
    bool hot = freq_is_hot(PC);

    if (is_prefetch(type)) {
        line_pref_q[set][way] = 1;
        line_stream_tag[set][way] = stream ? 1 : 0;
        rrpv[set][way] = RRPV_DISTANT; // keep in quarantine
    } else {
        line_pref_q[set][way] = 0;
        line_stream_tag[set][way] = stream ? 1 : 0;
        // Admission policy:
        // - Streaming & LFU-cold => hard-tail insertion (pollution guard)
        // - Hot or non-stream => protect modestly to allow short reuse
        if (stream && !hot) rrpv[set][way] = RRPV_DISTANT;
        else rrpv[set][way] = RRPV_PROTECT;
    }
}

static inline void hawkeye_on_hit(uint32_t set, uint32_t way) {
    // Promote to MRU on hit
    if (rrpv[set][way] > RRPV_MRU) rrpv[set][way] = RRPV_MRU;
    if (line_pref_q[set][way]) line_pref_q[set][way] = 0; // leave quarantine after a hit
}

static inline void mode_B_on_hit(uint32_t set, uint32_t way) {
    if (line_hits[set][way] < 3) line_hits[set][way]++;
    // Rescue short-reuse streams after HITS_TO_PROMOTE demand hits
    if (line_stream_tag[set][way] && line_hits[set][way] >= HITS_TO_PROMOTE) {
        rrpv[set][way] = RRPV_MRU;
        line_stream_tag[set][way] = 0; // clear tag after rescue
    } else {
        // Light promotion
        if (rrpv[set][way] > 0) rrpv[set][way]--;
    }
    if (line_pref_q[set][way]) line_pref_q[set][way] = 0;
}

// ============================================================================
// ChampSim required interface
// ============================================================================
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
            line_hits[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
    }
    std::memset(SHCT, 0, sizeof(SHCT));

    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    std::memset(pc_freq, 0, sizeof(pc_freq));

    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;

    std::cout << "Initialize H+SSL replacement state" << std::endl;
}

// return value should be 0 ~ 15 or 16 (bypass)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Fast invalid check
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) return way;
    }

    bool use_mode_B = should_use_mode_B(set);

    // Mode B may bypass on LFU-cold streams (never on writebacks)
    if (use_mode_B && mode_B_should_bypass(set, PC, paddr, type, current_set)) {
        return LLC_WAYS; // bypass
    }

    // Victim selection
    if (use_mode_B) {
        return mode_B_victim(set, current_set);
    } else {
        return pick_victim_rrip(set, current_set); // baseline Hawkeye-like
    }
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
    // Hawkeye always trained (safe baseline)
    if (!is_writeback(type)) {
        hawkeye_train_on_access(PC, hit);
    }

    // Update selector accounting
    update_selector(set, hit);
    bump_confidence_toward_global(set);

    // Frequency tracker + periodic aging
    freq_on_access(PC);
    periodic_freq_age();

    // Ignore writebacks (no insertion/promotion policy applied)
    if (is_writeback(type)) return;

    bool use_mode_B = should_use_mode_B(set);

    // On hit: promote according to chosen mode
    if (hit) {
        if (use_mode_B) mode_B_on_hit(set, way);
        else hawkeye_on_hit(set, way);
        return;
    }

    // On miss: a fill happens at (set, way) unless bypassed
    if (way >= LLC_WAYS) {
        // bypassed in GetVictimInSet
        return;
    }

    // Insertion policy according to chosen mode
    if (use_mode_B) mode_B_insertion(set, way, PC, paddr, type);
    else hawkeye_insertion(set, way, PC, type);
}

void PrintStats() { }
void PrintStats_Heartbeat() { }