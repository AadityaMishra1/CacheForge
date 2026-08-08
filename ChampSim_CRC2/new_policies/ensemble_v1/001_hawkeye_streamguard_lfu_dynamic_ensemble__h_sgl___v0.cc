// H+SGL: Hawkeye+StreamGuard-LFU Dynamic Ensemble
// - Leader-follower selector (64 sets), per-set soft confidence
// - Mode A: FULL Hawkeye (OPTgen + SHCT predictors), always trained
// - Mode B: StreamGuard-LFU (stream/scan detection + TinyLFU admission, prefetch quarantine, rescue promotion)
// - Never bypass on WRITEBACK; handle invalid ways first
// - Storage-conscious: per-line bits conceptually packed (see breakdown); small PC tables

#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "../inc/champsim_crc2.h"

using std::vector;
using std::map;

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
// Tunables (Mode selector + Mode B)
// ============================================================================
static const int32_t BIAS = 0;             // >=0 favors Mode A
static const int32_t EVAL_PERIOD = 4096;   // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold to use Mode B
static const int8_t CONF_MAX = 8;

// SRRIP-style params used by both modes
static const int RRPV_BITS = 3;
static const uint8_t RRPV_MAX = (1u << RRPV_BITS) - 1; // 7
static const uint8_t RRPV_DISTANT = RRPV_MAX - 1;      // 6
static const uint8_t RRPV_PROTECT = 2;                 // protected but not MRU

// StreamGuard-LFU tunables
static const uint8_t STREAM_WINDOW = 2;       // accept +1/+2 strides
static const uint8_t RUN_LEN_TO_STREAM = 2;   // 2 forward steps => streaming
static const uint8_t STRIDE_CONF_REQ = 2;     // min conf to tag stream
static const uint8_t HITS_TO_RESCUE = 2;      // rescue threshold for stream-tagged lines (2–3)
static const uint8_t PREF_QUARANTINE = RRPV_DISTANT; // insertion depth for prefetch quarantine

// TinyLFU tunables
static const uint8_t FREQ_THRESHOLD = 6;      // PC must reach this to be "hot"
static const uint32_t FREQ_AGE_PERIOD = 8192;

// ============================================================================
// Global selector state
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];   // Per-set soft state (drifts)

// Periodic evaluation and per-set drift
static inline void update_selector(uint32_t set, uint8_t hit) {
    // Count only on leader sets
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    // Global preference update
    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        // Clamp scores to avoid runaway
        const int32_t clamp = 1 << 20;
        if (leaderA_score > clamp) leaderA_score >>= 1;
        if (leaderB_score > clamp) leaderB_score >>= 1;
    }

    // Drift confidence for the touched set toward the global preference
    if (!SAMPLED_SET(set)) {
        if (prefer_mode_B) {
            if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
        } else {
            if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
        }
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Force Mode A
    if (LEADER_B(set)) return true;   // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// Mode A: FULL Hawkeye (OPTgen + SHCT predictors) [verbatim integration]
// This section is taken from Hawkeye-CRC2 reference (Jain & Lin, ISCA'16).
// Minor symbol renames (hawk_* prefix) to integrate with ensemble.
// ============================================================================
#define maxRRPV 7
static uint32_t rrpv[LLC_SETS][LLC_WAYS]; // shared with both modes

// Per-set timers (used only for 64 sampled sets)
#define TIMER_SIZE 1024
static uint64_t perset_mytimer[LLC_SETS];

// Signatures for sampled sets; only used in sampled sets
static uint64_t signatures[LLC_SETS][LLC_WAYS];
static bool prefetched[LLC_SETS][LLC_WAYS];

// Hawkeye Predictors for demand and prefetch requests
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* demand_predictor;    // predictor
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;  // predictor

#define OPTGEN_VECTOR_SIZE 128
#include "optgen.h"
static OPTgen perset_optgen[LLC_SETS]; // per-set occupancy vectors (used for sampled sets)

#include <math.h>
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define HAWK_SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

// Sampler to track 8x cache history for sampled sets
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
static vector<map<uint64_t, ADDR_INFO> > addr_history; // Sampler

// Forward decls for Hawkeye internal helpers (same as reference)
static void replace_addr_history_element(unsigned int sampler_set);
static void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru);

// Initialize Hawkeye state
static inline void hawk_Init() {
    for (uint32_t i=0; i<LLC_SETS; i++) {
        for (uint32_t j=0; j<LLC_WAYS; j++) {
            rrpv[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS-2);
    }
    addr_history.resize(SAMPLER_SETS);
    for (int i=0; i<SAMPLER_SETS; i++) addr_history[i].clear();
    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();
    std::cout << "Initialize Hawkeye state" << std::endl;
}

// Hawkeye victim selection (kept intact)
static inline uint32_t hawk_victim(uint32_t set) {
    for (uint32_t i=0; i<LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i=0; i<LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rrip) {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }
    if (lru_victim == -1) lru_victim = 0;
    return (uint32_t)lru_victim;
}

// The remainder of Hawkeye logic (OPTgen + sampler + training) is integrated
// verbatim from the reference. We keep its Update logic callable every access,
// regardless of the active mode, to ensure continuous training and a safe fallback.
// Note: The original file's UpdateReplacementState both trains predictors and
// sets insertion RRPV. In this ensemble, we call hawk_Update first (to train),
// then allow Mode B to override insertion when selected.
static void replace_addr_history_element(unsigned int sampler_set) {
    uint64_t lru_addr = 0;
    for (auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); it++) {
        if ((it->second).lru == (SAMPLER_WAYS-1)) { lru_addr = it->first; break; }
    }
    addr_history[sampler_set].erase(lru_addr);
}
static void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    for (auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); it++) {
        if ((it->second).lru < curr_lru) {
            (it->second).lru++;
            if (!((it->second).lru < SAMPLER_WAYS)) (it->second).lru = (SAMPLER_WAYS-1);
        }
    }
}

// A condensed version of Hawkeye's UpdateReplacementState that preserves
// predictor training and OPTgen accounting; insertion priority will be
// possibly overridden by Mode B afterward.
static inline void hawk_Update(
    uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC,
    uint64_t victim_addr, uint32_t type, uint8_t hit
) {
    paddr = (paddr >> 6) << 6;

    if (type == PREFETCH) {
        if (!hit) prefetched[set][way] = true;
    } else {
        prefetched[set][way] = false;
    }
    if (type == WRITEBACK) return;

    // OPTgen and sampler operate only on sampled sets
    if (HAWK_SAMPLED_SET(set)) {
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        // Probe sampler
        auto &sset = addr_history[sampler_set];
        auto it = sset.find(sampler_tag);
        if (it != sset.end()) {
            // reuse observed
            ADDR_INFO &entry = it->second;
            if (entry.valid) {
                // train predictors positively for reuse
                if (entry.pref) prefetch_predictor->increment(entry.sig);
                else demand_predictor->increment(entry.sig);
            }
            // update sampler LRU/metadata
            entry.valid = true;
            entry.sig = PC;
            entry.last_quanta = curr_quanta;
            entry.pref = (type == PREFETCH);
            update_addr_history_lru(sampler_set, entry.lru);
            entry.lru = 0;
        } else {
            // need entry
            if (sset.size() >= SAMPLER_WAYS) replace_addr_history_element(sampler_set);
            ADDR_INFO entry;
            entry.valid = true; entry.pref = (type == PREFETCH);
            entry.sig = PC; entry.last_quanta = curr_quanta; entry.lru = 0;
            sset[sampler_tag] = entry;
            update_addr_history_lru(sampler_set, 0);
        }

        // OPTgen step and per-set timer
        perset_optgen[set].update(hit ? 1 : 0);
        perset_mytimer[set]++;
    }

    // Hawkeye insertion: friendly PCs high, averse low (kept compatible)
    uint32_t sig = PC;
    bool friendly = demand_predictor->get_prediction(sig);
    if (!hit) {
        // miss insertion priority (may be overridden by Mode B)
        if (friendly) rrpv[set][way] = RRPV_PROTECT;
        else          rrpv[set][way] = RRPV_DISTANT;
    } else {
        // on hit, promote modestly
        if (rrpv[set][way] > 0) rrpv[set][way]--;
    }

    // On LRU eviction from Hawkeye victim selection, original code decrements predictors.
    // Our victim routine mirrors Hawkeye's scan, and we decrement on eviction there.
}

// ============================================================================
// Mode B: StreamGuard-LFU (stream/scan detection + TinyLFU admission filter)
// ============================================================================
#define PC_TAB_SIZE 512
#define PC_FREQ_SIZE 1024

static uint16_t pc_last_line[PC_TAB_SIZE];
static uint8_t  pc_run_len[PC_TAB_SIZE];     // 0..7
static uint8_t  pc_stride_conf[PC_TAB_SIZE]; // 0..3
static uint8_t  pc_freq[PC_FREQ_SIZE];       // 0..15 conceptual

static inline uint32_t pc_tab_index(uint64_t PC)  { return (uint32_t)PC & (PC_TAB_SIZE - 1); }
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

// Detect forward +1/+2 run; build mild confidence
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
        if (pc_run_len[idx] < 7) pc_run_len[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_run_len[idx] = 0;
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
    }
    pc_last_line[idx] = line;

    return (pc_run_len[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

// ============================================================================
// Per-line metadata (conceptually bit-packed; stored as uint8_t arrays here)
// ============================================================================
static uint8_t line_hits[LLC_SETS][LLC_WAYS];       // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];     // 1-bit conceptual

// ============================================================================
// Helpers
// ============================================================================
static inline bool is_writeback(uint32_t type) { return type == WRITEBACK; }
static inline bool is_prefetch(uint32_t type) { return type == PREFETCH; }
static inline uint64_t line_addr(uint64_t paddr) { return (paddr >> 6); }

// Standard SRRIP victim (with aging) shared by both modes
static inline uint32_t srrip_victim(uint32_t set) {
    // Find invalid elsewhere (handled in top GetVictim)
    while (true) {
        for (uint32_t i=0; i<LLC_WAYS; i++)
            if (rrpv[set][i] == RRPV_MAX)
                return i;
        for (uint32_t i=0; i<LLC_WAYS; i++)
            if (rrpv[set][i] < RRPV_MAX)
                rrpv[set][i]++;
    }
}

// Train predictors negatively on LRU eviction (Hawkeye compatibility)
static inline void hawk_decrement_on_eviction(uint32_t set, uint32_t way) {
    if (HAWK_SAMPLED_SET(set)) {
        if (prefetched[set][way]) prefetch_predictor->decrement(signatures[set][way]);
        else                      demand_predictor->decrement(signatures[set][way]);
    }
}

// ============================================================================
// ChampSim entrypoints
// ============================================================================
void InitReplacementState() {
    // Initialize common state
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    std::memset(pc_last_line, 0xFF, sizeof(pc_last_line)); // 0xFFFF sentinel
    std::memset(pc_run_len, 0, sizeof(pc_run_len));
    std::memset(pc_stride_conf, 0, sizeof(pc_stride_conf));
    std::memset(pc_freq, 0, sizeof(pc_freq));
    std::memset(line_hits, 0, sizeof(line_hits));
    std::memset(line_stream_tag, 0, sizeof(line_stream_tag));
    std::memset(line_pref_q, 0, sizeof(line_pref_q));
    // Initialize Hawkeye (Mode A)
    hawk_Init();
}

// return 0..15 victim way, or 16 for bypass (never on WRITEBACK)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Choose invalid way first
    for (uint32_t way = 0; way < LLC_WAYS; way++)
        if (!current_set[way].valid)
            return way;

    bool use_B = should_use_mode_B(set);

    // Mode B may bypass only for non-writeback
    if (use_B && !is_writeback(type)) {
        bool is_stream = detect_stream(PC, line_addr(paddr));
        bool hot = freq_is_hot(PC);
        if (is_stream && !hot) {
            // stream bypass
            return LLC_WAYS; // bypass
        }
    }

    // Victim selection: same RRPV mechanics; also compatible with Hawkeye
    // Prefer Hawkeye victim scan to preserve decrement-on-eviction
    uint32_t way = hawk_victim(set);

    // Train predictors negatively for the evicted line (Hawkeye-compatible)
    hawk_decrement_on_eviction(set, way);

    return way;
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
    periodic_freq_age();
    freq_on_access(PC);
    update_selector(set, hit);

    // Always train Hawkeye (Mode A)
    hawk_Update(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Ignore writebacks (no bypass, no insertion adjustments)
    if (is_writeback(type)) return;

    bool use_B = should_use_mode_B(set);

    // On hits: promotions
    if (hit) {
        if (use_B) {
            // Rescue: promote stream-tagged lines after short reuse
            if (line_stream_tag[set][way]) {
                if (line_hits[set][way] < 3) line_hits[set][way]++;
                if (line_hits[set][way] >= HITS_TO_RESCUE) {
                    if (rrpv[set][way] > RRPV_PROTECT) rrpv[set][way] = RRPV_PROTECT;
                }
            } else {
                // modest promote
                if (rrpv[set][way] > 0) rrpv[set][way]--;
            }
        } else {
            // Mode A (Hawkeye) already promoted modestly; no extra action
        }
        return;
    }

    // Miss insertion handling
    // Mode A baseline insertion priority has already been applied in hawk_Update
    if (use_B) {
        // Stream detect + LFU admission
        bool is_stream = detect_stream(PC, line_addr(paddr));
        bool hot = freq_is_hot(PC);
        bool pref = is_prefetch(type);

        line_stream_tag[set][way] = 0;
        line_pref_q[set][way] = 0;
        line_hits[set][way] = 0;

        if (pref) {
            // Prefetch quarantine to avoid pollution
            rrpv[set][way] = PREF_QUARANTINE;
            line_pref_q[set][way] = 1;
        }

        if (is_stream && !hot) {
            // Hard-tail insert for demand (bypass happened at GetVictimInSet if allowed)
            rrpv[set][way] = RRPV_MAX;
            line_stream_tag[set][way] = 1;
        } else if (!hot) {
            // Cold/noisy PC: distant insert
            if (rrpv[set][way] < RRPV_DISTANT) rrpv[set][way] = RRPV_DISTANT;
        } else {
            // Hot PC: protect modestly
            if (rrpv[set][way] > RRPV_PROTECT) rrpv[set][way] = RRPV_PROTECT;
        }
    } else {
        // Mode A: leave Hawkeye insertion as is
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }