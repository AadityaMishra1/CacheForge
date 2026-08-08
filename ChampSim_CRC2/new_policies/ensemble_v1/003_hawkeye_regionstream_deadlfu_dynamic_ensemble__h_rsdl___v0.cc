#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include "../inc/champsim_crc2.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// 3-bit RRIP
static const uint8_t RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU

// ---------------------------------------------------------------------------
// Leader-Follower Sampling (64 sampled sets, symmetric bit-indexing)
// ---------------------------------------------------------------------------
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ---------------------------------------------------------------------------
// Selector tunables
// ---------------------------------------------------------------------------
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const int32_t EVAL_PERIOD = 4096;   // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 8;

// ---------------------------------------------------------------------------
// Mode B tunables (exposed for exploration)
// ---------------------------------------------------------------------------
// RegionStream detector
static const uint8_t RUN_LEN_TO_STREAM = 2; // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ   = 2; // confidence to declare stream

// TinyLFU
static const uint8_t FREQ_THRESHOLD = 6;     // "hot" threshold
static const uint32_t FREQ_AGE_PERIOD = 8192;

// Promotion gate (per-PC)
static const uint8_t HITS_TO_PROMOTE = 2;  // 2 demand hits to promote

// ---------------------------------------------------------------------------
// Global selector state
// ---------------------------------------------------------------------------
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];   // Per-set soft state (drifts)

// ---------------------------------------------------------------------------
// Mode A: Hawkeye-like SHCT (always trained)
// ---------------------------------------------------------------------------
#define SHCT_ENTRIES 2048
static uint8_t SHCT[SHCT_ENTRIES]; // 5-bit conceptual (0..31), stored in uint8_t
static const uint8_t SHCT_MAX = 31;
static const uint8_t SHCT_INC = 1;
static const uint8_t SHCT_DEC = 1;
static const uint8_t SHCT_FRIENDLY_TH = 8;

static inline uint32_t shct_index(uint64_t pc) {
    return (uint32_t)pc & (SHCT_ENTRIES - 1);
}
static inline bool hawkeye_predict_friendly(uint64_t PC) {
    return SHCT[shct_index(PC)] >= SHCT_FRIENDLY_TH;
}
static inline void hawkeye_train(uint64_t PC, uint8_t hit) {
    uint32_t idx = shct_index(PC);
    if (hit) {
        if (SHCT[idx] < SHCT_MAX) SHCT[idx] += SHCT_INC;
    } else {
        if (SHCT[idx] > 0) SHCT[idx] -= SHCT_DEC;
    }
}

// ---------------------------------------------------------------------------
// Shared per-line state (RRIP) + tiny stream tag (1 bit per line, packed)
// ---------------------------------------------------------------------------
static uint8_t rrpv[LLC_SETS][LLC_WAYS]; // 3-bit conceptual

static const uint32_t N_LINES = LLC_SETS * LLC_WAYS;
static uint8_t stream_bit[(N_LINES + 7) / 8]; // 1 bit per line

static inline uint32_t line_index(uint32_t set, uint32_t way) {
    return set * LLC_WAYS + way;
}
static inline uint8_t get_stream_tag(uint32_t set, uint32_t way) {
    uint32_t idx = line_index(set, way);
    return (stream_bit[idx >> 3] >> (idx & 7)) & 1u;
}
static inline void set_stream_tag(uint32_t set, uint32_t way, uint8_t val) {
    uint32_t idx = line_index(set, way);
    uint8_t mask = (1u << (idx & 7));
    if (val) stream_bit[idx >> 3] |= mask;
    else     stream_bit[idx >> 3] &= ~mask;
}

// ---------------------------------------------------------------------------
// Mode B: RegionStream + TinyLFU + per-PC promotion gate
// ---------------------------------------------------------------------------
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE]; // track last 16-bit line within region
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3 (2-bit conceptual)
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7 (3-bit conceptual)

static uint8_t  pc_freq[PC_TABLE_SIZE];     // 4-bit conceptual hotness counter
static uint8_t  pc_promote[PC_TABLE_SIZE];  // 2-bit conceptual multi-hit promote

static inline uint32_t pc_tab_index(uint64_t PC) { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }

static inline void periodic_age_pc_tables() {
    if ((access_count % FREQ_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
            pc_freq[i] >>= 1;                 // TinyLFU aging
            pc_promote[i] >>= 1;              // soften phase memory
            if (pc_stride_conf[i]) pc_stride_conf[i]--; // mild decay
            // forward_run naturally resets on non-forward
        }
    }
}

static inline void lfu_on_access(uint64_t PC) {
    uint32_t idx = pc_tab_index(PC);
    if (pc_freq[idx] < 15) pc_freq[idx]++;
}

static inline uint8_t lfu_is_hot(uint64_t PC) {
    return pc_freq[pc_tab_index(PC)] >= FREQ_THRESHOLD;
}

// Stream detection: detect +1/+2 forward runs with small confidence window.
// paddr_line is physical line number (paddr >> 6); we fold to 16 bits to act region-local.
static inline bool detect_region_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t line16 = (uint16_t)(paddr_line & 0xFFFFu);
    uint16_t last = pc_last_line[idx];

    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(line16 - last);
        forward = (diff == 1) || (diff == 2);
    }

    if (forward) {
        if (pc_forward_run[idx] < 7) pc_forward_run[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_forward_run[idx] = 0;
    }

    pc_last_line[idx] = line16;
    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

static inline void promote_on_hits(uint64_t PC) {
    uint32_t idx = pc_tab_index(PC);
    if (pc_promote[idx] < 3) pc_promote[idx]++;
}
static inline bool should_promote_pc(uint64_t PC) {
    return pc_promote[pc_tab_index(PC)] >= HITS_TO_PROMOTE;
}

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------
static void update_selector(uint32_t set, uint8_t hit) {
    int delta = hit ? 1 : -1;

    if (SAMPLED_SET(set)) {
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    if ((access_count % EVAL_PERIOD) == 0) {
        bool new_pref_B = (leaderB_score > (leaderA_score + BIAS));
        prefer_mode_B = new_pref_B;

        // drift follower confidence toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (SAMPLED_SET(s)) continue; // don't overwrite leaders
            if (prefer_mode_B) {
                if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
            } else {
                if (mode_confidence[s] > -CONF_MAX) mode_confidence[s]--;
            }
        }

        // slowly decay scores to avoid overflow and allow phase change
        leaderA_score >>= 1;
        leaderB_score >>= 1;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Force Mode A
    if (LEADER_B(set)) return true;   // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline bool is_writeback(uint32_t type) { return type == WRITEBACK; }
static inline bool is_prefetch(uint32_t type) { return type == PREFETCH; }

// ---------------------------------------------------------------------------
// Mode-specific victim selection (both reuse shared RRIP state)
// ---------------------------------------------------------------------------
static uint32_t victim_find_rrip(uint32_t set, const BLOCK *current_set) {
    // First, prefer invalid ways
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (!current_set[i].valid) return i;
    }
    // Then, standard SRRIP: look for RRPV_MAX; else age and retry
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] == RRPV_MAX) return i;
        }
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < RRPV_MAX) rrpv[set][i]++;
        }
    }
}

// Mode A: Hawkeye-like (SHCT-guided) insertion/promotion
static inline void hawkeye_on_hit(uint32_t set, uint32_t way) {
    rrpv[set][way] = 0;
}
static inline void hawkeye_on_fill(uint32_t set, uint32_t way, uint64_t PC, uint32_t type) {
    // Prefetches quarantined deeper
    if (is_prefetch(type)) {
        rrpv[set][way] = RRPV_MAX;
        set_stream_tag(set, way, 0);
        return;
    }
    bool friendly = hawkeye_predict_friendly(PC);
    rrpv[set][way] = friendly ? RRPV_PROTECT : RRPV_DISTANT;
    set_stream_tag(set, way, 0);
}

// Mode B: RegionStream + TinyLFU (+ per-PC promotion)
static inline void modeB_on_hit(uint32_t set, uint32_t way, uint64_t PC) {
    promote_on_hits(PC);
    if (get_stream_tag(set, way) && should_promote_pc(PC)) {
        rrpv[set][way] = 0;            // rescue short reuse
        set_stream_tag(set, way, 0);   // clear stream tag after promotion
    } else {
        // Normal hit promotion
        rrpv[set][way] = 0;
    }
}

static inline void modeB_on_fill(uint32_t set, uint32_t way, uint64_t PC, uint64_t paddr, uint32_t type) {
    uint64_t line = paddr >> 6;
    bool is_stream = detect_region_stream(PC, line);
    lfu_on_access(PC);
    bool hot = lfu_is_hot(PC);

    if (is_prefetch(type)) {
        // quarantine prefetches to avoid pollution
        rrpv[set][way] = RRPV_MAX;
        set_stream_tag(set, way, 0);   // track streams on demand only
        return;
    }

    if (is_stream && !hot) {
        // stream/scan quarantine at tail; tag so a later hit can be rescued
        rrpv[set][way] = RRPV_MAX;
        set_stream_tag(set, way, 1);
    } else if (hot) {
        // frequent PC: protect
        rrpv[set][way] = RRPV_PROTECT;
        set_stream_tag(set, way, 0);
    } else {
        // default distant insertion
        rrpv[set][way] = RRPV_DISTANT;
        set_stream_tag(set, way, 0);
    }
}

// ---------------------------------------------------------------------------
// ChampSim entry points
// ---------------------------------------------------------------------------

// Initialize replacement state
void InitReplacementState() {
    // RRIP state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
        }
        mode_confidence[s] = 0;
    }
    // Selector scores
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
    access_count = 0;

    // SHCT
    std::memset(SHCT, 0, sizeof(SHCT));

    // Mode B tables
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
        pc_freq[i] = 0;
        pc_promote[i] = 0;
    }
    std::memset(stream_bit, 0, sizeof(stream_bit));

    // Optional banner
    // std::cout << "Init H+RSDL (Hawkeye-like + RegionStream-DeadLFU) state\n";
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
    bool use_mode_B = should_use_mode_B(set);
    (void)cpu; (void)PC; (void)paddr; (void)type;
    // Both modes share RRIP victim search; Mode B may have different insertion later.
    return victim_find_rrip(set, current_set);
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

    // Never bypass on writebacks; still train and return
    if (is_writeback(type)) {
        // writebacks typically not counted as hits; keep RRIP modestly protected
        rrpv[set][way] = (rrpv[set][way] > 0) ? (rrpv[set][way] - 1) : 0;
        return;
    }

    // Selector bookkeeping
    update_selector(set, hit);

    // Always train Mode A SHCT
    hawkeye_train(PC, hit);

    // Periodic Mode B table aging
    periodic_age_pc_tables();

    bool use_mode_B = should_use_mode_B(set);

    if (hit) {
        // On hit: apply chosen mode's promotion
        if (use_mode_B) modeB_on_hit(set, way, PC);
        else            hawkeye_on_hit(set, way);
    } else {
        // On fill: apply chosen mode's insertion
        if (use_mode_B) modeB_on_fill(set, way, PC, paddr, type);
        else            hawkeye_on_fill(set, way, PC, type);
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank
}