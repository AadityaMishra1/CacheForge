#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include "../inc/champsim_crc2.h"

// ============================================================================
// Configuration
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
// Tunables (knobs for exploration)
// - STREAM_WINDOW: +1/+2 forward-run acceptance
// - RUN_LEN_TO_STREAM: min consecutive strides to tag stream
// - STRIDE_CONF_REQ: stride confidence to declare stream
// - HITS_TO_PROMOTE: demand hits to escape quarantine
// - FREQ_THRESHOLD: TinyLFU admission threshold
// - BYPASS_STREAMS: allow demand bypass on proven streams
// - PREF_Q_DEPTH: quarantine depth for prefetches/streams (RRPV)
// - EVAL_PERIOD: selector epoch length
// ============================================================================
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU

// Selector tunables
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096;  // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 8;

// Stream detector tunables
static const uint8_t STREAM_WINDOW = 2;       // accept +1/+2 strides
static const uint8_t RUN_LEN_TO_STREAM = 2;   // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ = 2;     // confidence to declare stream
static const bool BYPASS_STREAMS = true;      // allow bypass for streams

// Promotion gate tunables
static const uint8_t HITS_TO_PROMOTE = 2;     // demand hits to promote stream-tagged lines

// TinyLFU tunables
static const uint8_t FREQ_THRESHOLD = 6;      // PC must reach this to be "hot"
static const uint32_t FREQ_AGE_PERIOD = 8192; // LFU aging period

// Prefetch quarantine
static const uint8_t PREF_Q_DEPTH = RRPV_DISTANT; // deeper is more quarantine

// ============================================================================
// Selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];   // Per-set soft state (drifts)

// ============================================================================
// Shared per-line state (3-bit RRIP reused by both modes; Hawkeye owns policy)
// ============================================================================
static uint8_t rrpv[LLC_SETS][LLC_WAYS];          // 3-bit conceptual

// ============================================================================
// Mode A: Full Hawkeye (CRC2) implementation
//   - OPTgen sampling per sampled set
//   - Sampler (8-way) to track reuse
//   - PC-based SHCT for friendly/averse classification
//   - SRRIP-based victim/insertion as in Hawkeye
// NOTE: Logic unchanged; we only wrap calls to integrate with the selector.
// ============================================================================

#define maxRRPV RRPV_MAX

// Per-set timers (only 64 sampled sets effectively used)
#define TIMER_SIZE 1024
uint64_t perset_mytimer[LLC_SETS];

// Signatures + prefetched flags per line (used for training on sampled sets)
uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched[LLC_SETS][LLC_WAYS];

#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
HAWKEYE_PC_PREDICTOR* demand_predictor;    // Predictor for demand
HAWKEYE_PC_PREDICTOR* prefetch_predictor;  // Predictor for prefetch

#define OPTGEN_VECTOR_SIZE 128
#include "optgen.h"
OPTgen perset_optgen[LLC_SETS]; // per-set occupancy vectors; used for sampled sets

#include <math.h>
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define HAWK_SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

// Sampler to track 8x cache history for sampled sets
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE/SAMPLER_WAYS)

typedef struct
{
    uint64_t last_quanta;
    uint64_t signature;
    bool    used;
    uint8_t lru;
} ADDR_INFO;

std::vector<std::map<uint64_t, ADDR_INFO> > addr_history; // Sampler

static inline bool is_writeback(uint32_t type) { return type == WRITEBACK; }
static inline bool is_prefetch(uint32_t type) { return type == PREFETCH; }
static inline bool is_demand(uint32_t type) { return (type != PREFETCH) && (type != WRITEBACK); }

// Hawkeye helpers (original behavior preserved)
static void hawk_replace_addr_history_element(unsigned int sampler_set)
{
    uint64_t lru_addr = 0;
    for (auto it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); ++it) {
        if ((it->second).lru == (SAMPLER_WAYS-1)) {
            lru_addr = it->first;
            break;
        }
    }
    addr_history[sampler_set].erase(lru_addr);
}

static void hawk_update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru)
{
    for (auto it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); ++it) {
        if ((it->second).lru < curr_lru) {
            (it->second).lru++;
            assert((it->second).lru < SAMPLER_WAYS);
        }
    }
}

// Initialize replacement state
void InitReplacementState()
{
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        mode_confidence[i] = 0;
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2); // occupancy target (ISCA'16)
        for (uint32_t j = 0; j < LLC_WAYS; j++) {
            rrpv[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
    }

    addr_history.resize(SAMPLER_SETS);
    for (int i = 0; i < SAMPLER_SETS; i++) addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // Initialize Mode B structures (below)
    // Initialize PC tables for stream detection and LFU
    for (int i = 0; i < 1024; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
        pc_freq[i] = 0;
    }
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            line_hits[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
    }

    std::cout << "Initialize H+SSQ state (Hawkeye + StreamScan-LFU)" << std::endl;
}

// Hawkeye victim selection (unchanged)
static uint32_t hawkeye_victim(uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{
    // First, choose invalid if exists
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    // look for the maxRRPV line
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    // If cannot find maxRRPV, evict the oldest cache-friendly line (largest RRIP)
    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rrip) {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }

    assert(lru_victim != -1);
    // The predictor is trained negatively on LRU evictions in sampled sets
    if (HAWK_SAMPLED_SET(set)) {
        if (prefetched[set][lru_victim])
            prefetch_predictor->decrement(signatures[set][lru_victim]);
        else
            demand_predictor->decrement(signatures[set][lru_victim]);
    }
    return (uint32_t)lru_victim;
}

// Hawkeye insertion/promotion (unchanged)
static inline void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type)
{
    // Predict friendliness
    uint64_t sig = signatures[set][way];
    bool friend_pred = false;
    if (is_prefetch(type)) friend_pred = (prefetch_predictor->get_prediction(sig) >= 0);
    else                    friend_pred = (demand_predictor->get_prediction(sig) >= 0);

    if (friend_pred) {
        // Insert closer to MRU
        rrpv[set][way] = RRPV_PROTECT;
    } else {
        // Insert far from MRU
        rrpv[set][way] = maxRRPV;
    }
}

static inline void hawkeye_promotion(uint32_t set, uint32_t way)
{
    // On hit, bring closer to MRU
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}

// Called on every cache hit and cache fill (Hawkeye training)
static void hawkeye_train(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    // Align line
    paddr = (paddr >> 6) << 6;

    // Prefetch mark
    if (type == PREFETCH) {
        if (!hit) prefetched[set][way] = true;
    } else {
        prefetched[set][way] = false;
    }

    // Ignore writebacks
    if (type == WRITEBACK) return;

    // If we are sampling, OPTgen sees accesses from sampled sets
    if (HAWK_SAMPLED_SET(set)) {
        // current timestep
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;
        assert(sampler_set < SAMPLER_SETS);

        auto &set_map = addr_history[sampler_set];
        auto it = set_map.find(sampler_tag);

        // Previously used line?
        if (it != set_map.end()) {
            // A re-reference, demand accesses only train right ends
            if (is_demand(type)) {
                // feed OPTgen: true = hit in OPT at target occupancy
                bool hit_in_opt = perset_optgen[set].should_cache(curr_quanta, (it->second).last_quanta);
                uint64_t sig = (it->second).signature;
                if (hit_in_opt) {
                    // friendly
                    demand_predictor->increment(sig);
                } else {
                    // averse
                    demand_predictor->decrement(sig);
                }
                // update addr_history used bit
                (it->second).used = true;
            }
            // update LRU for sampler
            (it->second).last_quanta = curr_quanta;
            hawk_update_addr_history_lru(sampler_set, (it->second).lru);
            (it->second).lru = 0;
        } else {
            // Miss in sampler set: allocate if room else evict LRU entry
            if (set_map.size() >= SAMPLER_WAYS) hawk_replace_addr_history_element(sampler_set);
            ADDR_INFO new_info;
            new_info.last_quanta = curr_quanta;
            new_info.signature = CRC(PC) % SHCT_SIZE;
            new_info.used = false;
            new_info.lru = 0;
            // age other entries
            hawk_update_addr_history_lru(sampler_set, 0);
            set_map[sampler_tag] = new_info;
        }

        // advance timer quanta for this set
        perset_mytimer[set]++;
    }

    // On hit, promote
    if (hit) {
        hawkeye_promotion(set, way);
    } else {
        // On fill, set line signature for later negative training on eviction in sampled sets
        signatures[set][way] = CRC(PC) % SHCT_SIZE;
        // Insertion based on predictor
        hawkeye_insertion(set, way, PC, type);
    }
}

// ============================================================================
// Mode B: StreamScan-LFU Quarantine
//   - Detect +1/+2 forward stride runs (PC-local)
//   - Quarantine streams/prefetches with deep RRPV insertion; optional bypass
//   - TinyLFU admission on PC to block cold/noisy PCs
//   - 2-hit promotion escapes quarantine for short reuse (e.g., zeusmp)
// ============================================================================
#define PC_TABLE_SIZE 1024
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7

#define PC_FREQ_SIZE 1024
static uint8_t pc_freq[PC_FREQ_SIZE]; // 0..15 conceptual

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

// Per-line lightweight Mode B metadata (packed conceptually; stored as bytes)
static uint8_t line_hits[LLC_SETS][LLC_WAYS];       // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];     // 1-bit conceptual

// Mode B victim selection: SRRIP-like (respect invalids; then RRPV==max; else age once)
static uint32_t mode_B_victim(uint32_t set, const BLOCK *current_set)
{
    // Invalid way first
    for (uint32_t way = 0; way < LLC_WAYS; way++)
        if (current_set[way].valid == 0) return way;

    // Find any maxRRPV
    for (uint32_t way = 0; way < LLC_WAYS; way++)
        if (rrpv[set][way] == RRPV_MAX) return way;

    // Age once and return a max
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (rrpv[set][way] < RRPV_MAX) rrpv[set][way]++;
    }
    for (uint32_t way = 0; way < LLC_WAYS; way++)
        if (rrpv[set][way] == RRPV_MAX) return way;

    // Fallback
    return 0;
}

// Mode B insertion policy with quarantine and LFU admission; may decide bypass
static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type, bool stream_detected)
{
    if (is_writeback(type)) return false; // never bypass writebacks
    if (!BYPASS_STREAMS) return false;
    // Bypass only demand fills on proven forward streams when PC not hot
    if (is_demand(type) && stream_detected && !freq_is_hot(PC)) return true;
    return false;
}

static inline void mode_B_insertion(uint32_t set, uint32_t way, uint64_t PC, uint64_t paddr, uint32_t type, bool stream_detected)
{
    // Admission using TinyLFU: cold PCs get tail insertion
    bool hot = freq_is_hot(PC);

    // Quarantine for prefetches and streams
    if (is_prefetch(type) || stream_detected) {
        rrpv[set][way] = PREF_Q_DEPTH; // deep insertion
        line_pref_q[set][way] = 1;
        line_stream_tag[set][way] = stream_detected ? 1 : 0;
        line_hits[set][way] = 0;
        return;
    }

    // Demand path
    line_pref_q[set][way] = 0;
    line_stream_tag[set][way] = stream_detected ? 1 : 0;
    line_hits[set][way] = 0;

    if (hot) {
        // Hot PC: protect moderately
        rrpv[set][way] = RRPV_PROTECT;
    } else {
        // Cold PC: tail or distant insert
        rrpv[set][way] = RRPV_MAX;
    }
}

static inline void mode_B_promotion(uint32_t set, uint32_t way)
{
    // Count demand hits; a 2-hit gate escapes quarantine/stream tag
    if (line_hits[set][way] < 3) line_hits[set][way]++;
    if (line_hits[set][way] >= HITS_TO_PROMOTE) {
        line_pref_q[set][way] = 0;
        line_stream_tag[set][way] = 0;
    }
    // Promote toward MRU modestly
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}

// ============================================================================
// Selector logic
// ============================================================================
static void update_selector(uint32_t set, uint8_t hit)
{
    access_count++;

    // Update LFU counts and age periodically (hotness tracking is global)
    // Note: freq_on_access is called by caller with actual PC
    if ((access_count % FREQ_AGE_PERIOD) == 0) periodic_freq_age();

    // Update leader scores on sampled sets
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    // Epoch decision
    if ((access_count % EVAL_PERIOD) == 0) {
        bool new_prefer_B = (leaderB_score > (leaderA_score + BIAS));
        prefer_mode_B = new_prefer_B;

        // Drift per-set confidence toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (prefer_mode_B) {
                if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
            } else {
                if (mode_confidence[s] > 0) mode_confidence[s]--;
            }
        }

        // mild decay to avoid runaway
        leaderA_score = leaderA_score / 2;
        leaderB_score = leaderB_score / 2;
    }
}

static inline bool should_use_mode_B(uint32_t set)
{
    if (LEADER_A(set)) return false;  // force Mode A
    if (LEADER_B(set)) return true;   // force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// Glue: Victim selection & update
// ============================================================================
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Default to Hawkeye if Mode B is not preferred
    bool use_mode_B = should_use_mode_B(set);

    // Mode B may request bypass (return 16) on demand streaming
    if (use_mode_B) {
        uint64_t line_num = (paddr >> 6);
        bool stream = detect_stream(PC, line_num);
        // demand bypass only if allowed and not writeback
        if (mode_B_should_bypass(set, PC, paddr, type, stream))
            return 16u; // bypass
        return mode_B_victim(set, current_set);
    } else {
        return hawkeye_victim(set, current_set, PC, paddr, type);
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
    // Track frequency on every access (used by Mode B)
    if (!is_writeback(type)) freq_on_access(PC);

    // Update selector stats (leader scores)
    update_selector(set, hit);

    // Always train Mode A (Hawkeye), regardless of chosen mode
    hawkeye_train(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Apply chosen mode’s insertion/promotion
    bool use_mode_B = should_use_mode_B(set);

    if (hit) {
        // On hit: both modes promote via their rules
        if (use_mode_B) {
            mode_B_promotion(set, way);
        } else {
            hawkeye_promotion(set, way);
        }
        return;
    }

    // Miss path (fill)
    // Never bypass on writeback (already filtered at GetVictimInSet)
    if (is_writeback(type)) {
        // writeback lines: conservative insertion
        rrpv[set][way] = RRPV_DISTANT;
        line_pref_q[set][way] = 0;
        line_stream_tag[set][way] = 0;
        line_hits[set][way] = 0;
        return;
    }

    if (use_mode_B) {
        uint64_t line_num = (paddr >> 6);
        bool stream = detect_stream(PC, line_num);
        mode_B_insertion(set, way, PC, paddr, type, stream);
    } else {
        // Hawkeye insertion is already invoked inside hawkeye_train on miss,
        // but we re-apply to ensure state if framework calls Update after fill.
        hawkeye_insertion(set, way, PC, type);
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}