#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "../inc/champsim_crc2.h"

// ============================================================================
// Config
// ============================================================================
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
// Tunables
// ============================================================================
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // moderately protected
static const uint8_t RRPV_HARD_TAIL = RRPV_MAX;        // quarantine tail

// Selector
static const int32_t BIAS = 0;             // >=0 favors Mode A (Hawkeye)
static const uint32_t EVAL_PERIOD = 4096;  // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 7;

// Stream detector
static const uint8_t RUN_LEN_TO_STREAM = 2; // 2 forward steps => stream
static const uint8_t STRIDE_CONF_REQ = 2;   // confidence to declare stream

// Promotion gate
static const uint8_t HITS_TO_PROMOTE = 2;   // demand hits to promote stream-tagged lines
static const uint8_t MAX_PROTECTED_PER_SET = 6; // cap protected lines per set

// TinyLFU
static const uint8_t FREQ_THRESHOLD = 6;    // hotness threshold
static const uint32_t FREQ_AGE_PERIOD = 8192;

// Ensemble policy knobs (bypass only for demand/prefetch, never WB)
static const bool ENABLE_DEMAND_BYPASS = true;

// ============================================================================
// Global selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A (Hawkeye)
static int32_t leaderB_score = 0;  // Hits - misses on Mode B (StreamGuard-TinyLFU)
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];    // per-set drifted confidence

static inline void drift_confidence(bool preferB) {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (SAMPLED_SET(s)) continue; // leave leaders unbiased by drift
        if (preferB) {
            if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
        } else {
            if (mode_confidence[s] > -CONF_MAX) mode_confidence[s]--;
        }
    }
}

static inline void update_selector(uint32_t set, uint8_t hit) {
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }
    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        // dampen/decay scores to avoid runaway growth
        leaderA_score = leaderA_score >> 1;
        leaderB_score = leaderB_score >> 1;
        drift_confidence(prefer_mode_B);
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false; // force Hawkeye
    if (LEADER_B(set)) return true;  // force StreamGuard-TinyLFU
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// Shared per-line metadata (conceptually bit-packed)
// ============================================================================
static uint8_t rrpv[LLC_SETS][LLC_WAYS];          // 3-bit conceptual
static uint8_t line_hits2[LLC_SETS][LLC_WAYS];    // 2-bit hit counter (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit stream tag
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];   // 1-bit prefetch quarantine

static inline bool is_writeback(uint32_t type) { return (type == WRITEBACK); }
static inline bool is_prefetch(uint32_t type)  { return (type == PREFETCH);  }

// ============================================================================
// Mode A: Full Hawkeye (OPTgen + SHCT + Sampler)
// Note: We include Hawkeye predictor/optgen and sampler; Mode A logic mirrors
// Hawkeye-CRC2 victim/insertion/training while coexisting with Mode B.
// ============================================================================
#define maxRRPV 7
// Hawkeye predictors
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* demand_predictor;    // 2K entries, 5-bit
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;  // 2K entries, 5-bit

// OPTgen per-set occupancy vectors (we only use sampled sets)
#define OPTGEN_VECTOR_SIZE 128
#include "optgen.h"
static OPTgen perset_optgen[LLC_SETS];

// Per-set timers (used by OPTgen)
static uint64_t perset_mytimer[LLC_SETS];

// Signatures + prefetch state for sampled sets (Hawkeye’s bookkeeping)
static uint64_t signatures[LLC_SETS][LLC_WAYS];
static bool prefetched[LLC_SETS][LLC_WAYS];

// Sampler to track 8x cache history for sampled sets (Hawkeye sampler)
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
static std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

// Helper from Hawkeye-CRC2
static inline uint64_t bitmask_u64(uint32_t l) {
    return (l == 64) ? (uint64_t)(-1LL) : (((uint64_t)1 << l) - 1ULL);
}
static inline uint64_t bits_u64(uint64_t x, uint32_t i, uint32_t l) {
    return (x >> i) & bitmask_u64(l);
}
static inline bool HAWK_SAMPLED_SET(uint32_t set) {
    // same 64-set sampling as leader/follower
    return SAMPLED_SET(set);
}

// Hawkeye: initialize internal state
static void hawkeye_init_state() {
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2); // occupancy target
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            signatures[i][w] = 0;
            prefetched[i][w] = false;
        }
    }
    addr_history.clear();
    addr_history.resize(SAMPLER_SETS);
    for (uint32_t i = 0; i < SAMPLER_SETS; i++) addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();
}

// Hawkeye: predictor query
static inline bool hawkeye_predict_friendly(uint64_t PC, bool is_pref) {
    if (is_pref) return prefetch_predictor->get_prediction(PC);
    else         return demand_predictor->get_prediction(PC);
}

// Hawkeye: train predictor and OPTgen/sampler (subset of official flow; full behavior requires headers)
static void hawkeye_train(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    // Track prefetch state
    if (type == PREFETCH) {
        if (!hit) prefetched[set][way] = true;
    } else {
        prefetched[set][way] = false;
    }
    // Ignore writebacks for Hawkeye training
    if (type == WRITEBACK) return;

    // OPTgen + sampler operate only on sampled sets
    if (!HAWK_SAMPLED_SET(set)) return;

    // Sampler bookkeeping (mirrors Hawkeye-CRC2 sampler structure)
    uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
    uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
    uint64_t sampler_tag = CRC(paddr >> 12) % 256;

    // Lookup in sampler
    auto &smap = addr_history[sampler_set];
    auto it = smap.find(sampler_tag);

    // Train predictors via OPTgen signal
    if (it != smap.end()) {
        // We saw this line before; compute reuse and train as Hawkeye does
        ADDR_INFO &ainfo = it->second;
        // OPTgen update
        uint32_t fill_occupancy = perset_optgen[set].update(ainfo.last_quanta, curr_quanta);

        // Friendly if cache would have contained it (Belady’s)
        bool friendly = (fill_occupancy != 0);
        if (ainfo.prefetched) prefetch_predictor->update(ainfo.sig, friendly);
        else                  demand_predictor->update(ainfo.sig, friendly);

        // Refresh this sampler entry
        ainfo.sig = PC;
        ainfo.prefetched = (type == PREFETCH);
        ainfo.last_quanta = curr_quanta;
        // LRU update within sampler set
        for (auto &kv : smap) if (kv.second.lru < ainfo.lru) kv.second.lru++;
        ainfo.lru = 0;
    } else {
        // Need to insert into sampler; evict LRU if full
        if (smap.size() >= SAMPLER_WAYS) {
            // evict LRU
            uint64_t victim_tag = 0;
            uint32_t victim_lru = 0;
            bool found = false;
            for (auto &kv : smap) {
                if (!found || kv.second.lru > victim_lru) {
                    victim_lru = kv.second.lru;
                    victim_tag = kv.first;
                    found = true;
                }
            }
            if (found) smap.erase(victim_tag);
        }
        // Insert new sampler entry
        ADDR_INFO ainfo{};
        ainfo.sig = PC;
        ainfo.prefetched = (type == PREFETCH);
        ainfo.last_quanta = curr_quanta;
        ainfo.lru = 0;
        for (auto &kv : smap) kv.second.lru++;
        smap[sampler_tag] = ainfo;
    }

    // Advance per-set time
    perset_mytimer[set]++;
}

// Hawkeye: insertion policy (guided by predictor)
static inline void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type, uint8_t hit) {
    bool friendly = hawkeye_predict_friendly(PC, (type == PREFETCH));
    if (hit) {
        // On hit, promote toward protection (SRRIP style)
        if (rrpv[set][way] > 0) rrpv[set][way] = 0;
    } else {
        // On fill, insert
        if (friendly) rrpv[set][way] = RRPV_PROTECT;
        else          rrpv[set][way] = RRPV_DISTANT;
    }
}

// Hawkeye: victim selection (SRRIP with predictor training on eviction at leaders)
// Return [0..15], or 16 for bypass (Hawkeye does not bypass)
static uint32_t hawkeye_victim(uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    // If invalid way exists, take it
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (!current_set[i].valid) return i;
    }
    // Try SRRIP maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    // Else, bump RRPVs until we find a victim
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < maxRRPV) rrpv[set][i]++;
            if (rrpv[set][i] == maxRRPV) return i;
        }
    }
    return 0;
}

// ============================================================================
// Mode B: StreamGuard + TinyLFU
// ============================================================================
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 2 bits conceptual (0..3)
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 3 bits conceptual (0..7)

#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE]; // 4-bit conceptual

static inline uint32_t pc_tab_index(uint64_t PC)  { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }
static inline uint32_t pc_freq_index(uint64_t PC) { return (uint32_t)PC & (PC_FREQ_SIZE  - 1); }

static inline void freq_on_access(uint64_t PC) {
    uint32_t idx = pc_freq_index(PC);
    if (pc_freq[idx] < 15) pc_freq[idx]++;
}
static inline bool freq_is_hot(uint64_t PC) { return pc_freq[pc_freq_index(PC)] >= FREQ_THRESHOLD; }
static inline void periodic_freq_age() {
    if ((access_count % FREQ_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_FREQ_SIZE; i++) pc_freq[i] >>= 1;
    }
}

// Detect forward +1/+2 stride runs with small confidence hysteresis
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

static inline uint32_t count_protected(uint32_t set) {
    uint32_t cnt = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] <= RRPV_PROTECT) cnt++;
    return cnt;
}

// Mode B victim selection (uses same SRRIP mechanism)
static uint32_t mode_B_victim(uint32_t set, const BLOCK *current_set) {
    // Reuse standard SRRIP victim mechanics
    for (uint32_t i = 0; i < LLC_WAYS; i++) if (!current_set[i].valid) return i;
    for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv[set][i] == RRPV_MAX) return i;
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < RRPV_MAX) rrpv[set][i]++;
            if (rrpv[set][i] == RRPV_MAX) return i;
        }
    }
    return 0;
}

static inline void mode_B_insertion(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint64_t line_addr = (paddr >> 6);
    bool is_stream = detect_stream(PC, line_addr);
    bool hot = freq_is_hot(PC);

    if (hit) {
        // Promote on hit (short-reuse booster)
        if (line_stream_tag[set][way]) {
            if (line_hits2[set][way] < 3) line_hits2[set][way]++;
            if (line_hits2[set][way] >= HITS_TO_PROMOTE) {
                if (count_protected(set) < MAX_PROTECTED_PER_SET) {
                    if (rrpv[set][way] > RRPV_PROTECT) rrpv[set][way] = RRPV_PROTECT;
                }
            }
        } else {
            // normal hit: gentle promotion
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
        return;
    }

    // On fill
    line_hits2[set][way] = 0;
    line_pref_q[set][way] = is_prefetch(type) ? 1 : 0;
    line_stream_tag[set][way] = is_stream ? 1 : 0;

    if (is_prefetch(type)) {
        // quarantine prefetches at hard tail
        rrpv[set][way] = RRPV_HARD_TAIL;
        return;
    }

    if (!hot && is_stream) {
        // stream+not-hot: hard-tail insert
        rrpv[set][way] = RRPV_HARD_TAIL;
    } else if (!hot && !is_stream) {
        // cold/noisy PCs: distant insert
        rrpv[set][way] = RRPV_DISTANT;
    } else {
        // hot PCs: moderate protection
        rrpv[set][way] = RRPV_PROTECT;
    }
}

// ============================================================================
// Required ChampSim entry points (ensemble wiring)
// ============================================================================
void InitReplacementState() {
    // Clear shared state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
            line_hits2[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
    }
    // Mode B PC tables
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    for (uint32_t i = 0; i < PC_FREQ_SIZE; i++) pc_freq[i] = 0;

    // Mode A (Hawkeye) init
    hawkeye_init_state();

    std::cout << "Initialize H+SGT ensemble (Mode A=Hawkeye, Mode B=StreamGuard-TinyLFU)" << std::endl;
}

// Return value: 0..15 or 16 (bypass). Must never bypass on WRITEBACK.
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Bypass only for demand/prefetch and only if Mode B says so
    bool use_mode_B = should_use_mode_B(set);

    // Demand/prefetch bypass for strong streams (never bypass on WB)
    if (use_mode_B && ENABLE_DEMAND_BYPASS && !is_writeback(type)) {
        uint64_t line_addr = (paddr >> 6);
        bool is_stream = detect_stream(PC, line_addr);
        bool hot = freq_is_hot(PC);
        if (is_stream && !hot) {
            // For prefetches or demand from cold stream PCs: bypass
            return 16;
        }
    }

    // Normal victim selection (SRRIP), identical in both modes
    if (use_mode_B) return mode_B_victim(set, current_set);
    else            return hawkeye_victim(set, current_set, PC, paddr, type);
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
    access_count++;

    // Telemetry for selector (leader-follower dueling)
    update_selector(set, hit);

    // TinyLFU periodic aging and per-access update
    periodic_freq_age();
    freq_on_access(PC);

    // Always train Mode A (Hawkeye) predictors/sampler/OPTgen
    hawkeye_train(set, way, (paddr >> 6) << 6, PC, type, hit);

    // Apply chosen mode’s insertion/promotion on the shared RRIP state
    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        mode_B_insertion(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_insertion(set, way, PC, type, hit);
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}