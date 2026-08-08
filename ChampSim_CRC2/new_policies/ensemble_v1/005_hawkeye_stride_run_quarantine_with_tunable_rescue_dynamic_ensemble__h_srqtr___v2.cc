#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>  // CRITICAL: for log2()
#include "../inc/champsim_crc2.h"  // CRITICAL: for BLOCK, PREFETCH, WRITEBACK

using namespace std;

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16

// ============================================================================
// MANDATORY SELECTOR PARAMETERS (DO NOT CHANGE - HARD-CODED REQUIREMENTS)
// ============================================================================
#define LEADER_SETS_A 32
#define LEADER_SETS_B 32

static const int32_t BIAS = 50;        // REQUIRED: Mode B needs +50 hit advantage
static const int8_t THRESHOLD = 6;      // REQUIRED: 75% confidence (6/8) to switch
static const int8_t CONF_MAX = 8;       // REQUIRED: Max confidence level
static bool prefer_mode_B = false;      // REQUIRED: Default to Hawkeye

// Leader set macros (first 32 = Leader A, next 32 = Leader B)
#define LEADER_A(set) ((set) < LEADER_SETS_A)
#define LEADER_B(set) ((set) >= LEADER_SETS_A && (set) < (LEADER_SETS_A + LEADER_SETS_B))

// Selector state (per-set confidence tracking)
static int8_t mode_confidence[LLC_SETS];  // Range: [0, CONF_MAX]
static int32_t leaderA_score = 0;         // Hits on Leader A sets
static int32_t leaderB_score = 0;         // Hits on Leader B sets

static inline bool should_use_mode_B(uint32_t set) {
    // LEADER SETS NEVER SWITCH (critical for clean measurements)
    if (LEADER_A(set)) return false;  // Leader A → Always Mode A (Hawkeye)
    if (LEADER_B(set)) return true;   // Leader B → Always Mode B

    // FOLLOWER SETS: Require STRONG evidence before switching to Mode B
    int32_t score_advantage = leaderB_score - leaderA_score;

    // Both conditions MUST be met to enable Mode B on followers:
    // 1. Mode B has significant advantage (>BIAS) on leader sets
    // 2. Per-set confidence is high (≥THRESHOLD)
    return (score_advantage > BIAS) && (mode_confidence[set] >= THRESHOLD);
}

static void update_selector(uint32_t set, uint8_t hit) {
    // Track leader set performance
    if (LEADER_A(set) && hit) leaderA_score++;
    if (LEADER_B(set) && hit) leaderB_score++;

    // Update global preference
    prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));

    // Update per-set confidence
    if (hit) {
        if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
    } else {
        if (mode_confidence[set] > 0) mode_confidence[set]--;
    }
}

// ============================================================================
// MODE A: FULL HAWKEYE IMPLEMENTATION (BASELINE - DO NOT MODIFY)
// ============================================================================
// This is the COMPLETE Hawkeye algorithm with:
// - OPTgen (Belady-optimal reuse prediction)
// - SHCT predictor (PC-based learning)
// - 8x cache history sampler (2800 entries)
// - Per-set timers and occupancy tracking
// Total: ~300 lines of code - COPY ALL OF IT

// ============================================================================
// CRITICAL: SEPARATE STATE FOR MODE A AND MODE B (PREVENTS CORRUPTION)
// ============================================================================
// Phase 1: Relaxed budget to guarantee baseline performance
// Phase 2: Will compress/optimize to meet 64 KB constraint
#define maxRRPV 7
uint32_t rrpv_A[LLC_SETS][LLC_WAYS];  // Mode A (Hawkeye) private state - NEVER touched by Mode B
uint32_t rrpv_B[LLC_SETS][LLC_WAYS];  // Mode B private state - NEVER touched by Hawkeye

// Legacy alias for Hawkeye code (points to rrpv_A)
#define rrpv rrpv_A

// Per-set timers (64 sampled sets × 10 bits = 80 bytes)
#define TIMER_SIZE 1024
uint64_t perset_mytimer[LLC_SETS];

// Signatures for sampled sets (64 sets × 16 ways × 12 bits = 1.5KB)
uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched[LLC_SETS][LLC_WAYS];

// Hawkeye PC Predictor (2048 entries × 5-bit counter = 1.25KB)
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)

// CRC hash function
uint64_t CRC(uint64_t _blockAddress) {
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal = _blockAddress;
    for(unsigned int i = 0; i < 32; i++)
        _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
    return _returnVal;
}

// HAWKEYE_PC_PREDICTOR class
class HAWKEYE_PC_PREDICTOR {
    map<uint64_t, short unsigned int> SHCT;
public:
    void increment(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if(SHCT.find(signature) == SHCT.end())
            SHCT[signature] = (1+MAX_SHCT)/2;
        SHCT[signature] = (SHCT[signature] < MAX_SHCT) ? (SHCT[signature]+1) : MAX_SHCT;
    }

    void decrement(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if(SHCT.find(signature) == SHCT.end())
            SHCT[signature] = (1+MAX_SHCT)/2;
        if(SHCT[signature] != 0)
            SHCT[signature] = SHCT[signature]-1;
    }

    bool get_prediction(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if(SHCT.find(signature) != SHCT.end() && SHCT[signature] < ((MAX_SHCT+1)/2))
            return false;
        return true;
    }
};

// ADDR_INFO structure for sampler
struct ADDR_INFO {
    uint64_t addr;
    uint32_t last_quanta;
    uint64_t PC;
    bool prefetched;
    uint32_t lru;

    void init(unsigned int curr_quanta) {
        last_quanta = 0;
        PC = 0;
        prefetched = false;
        lru = 0;
    }

    void update(unsigned int curr_quanta, uint64_t _pc, bool prediction) {
        last_quanta = curr_quanta;
        PC = _pc;
    }

    void mark_prefetch() {
        prefetched = true;
    }
};

// OPTgen structure (Belady-optimal tracking)
#define OPTGEN_VECTOR_SIZE 128
struct OPTgen {
    vector<unsigned int> liveness_history;
    uint64_t num_cache;
    uint64_t num_dont_cache;
    uint64_t access;
    uint64_t CACHE_SIZE;

    void init(uint64_t size) {
        num_cache = 0;
        num_dont_cache = 0;
        access = 0;
        CACHE_SIZE = size;
        liveness_history.resize(OPTGEN_VECTOR_SIZE, 0);
    }

    void add_access(uint64_t curr_quanta) {
        access++;
        liveness_history[curr_quanta] = 0;
    }

    void add_prefetch(uint64_t curr_quanta) {
        liveness_history[curr_quanta] = 0;
    }

    bool should_cache(uint64_t curr_quanta, uint64_t last_quanta) {
        bool is_cache = true;
        unsigned int i = last_quanta;
        while (i != curr_quanta) {
            if(liveness_history[i] >= CACHE_SIZE) {
                is_cache = false;
                break;
            }
            i = (i+1) % liveness_history.size();
        }

        if (is_cache) {
            i = last_quanta;
            while (i != curr_quanta) {
                liveness_history[i]++;
                i = (i+1) % liveness_history.size();
            }
        }

        if (is_cache) num_cache++;
        else num_dont_cache++;

        return is_cache;
    }

    uint64_t get_num_opt_hits() {
        return num_cache;
    }
};

// Global Hawkeye structures
HAWKEYE_PC_PREDICTOR* demand_predictor;
HAWKEYE_PC_PREDICTOR* prefetch_predictor;
OPTgen perset_optgen[LLC_SETS];

// Sampler (2800 entries × 4 bytes = 11.2KB)
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE/SAMPLER_WAYS)
vector<map<uint64_t, ADDR_INFO>> addr_history;

// Sampling macros
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0, 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

// Sampler helper functions
void replace_addr_history_element(unsigned int sampler_set) {
    uint64_t lru_addr = 0;
    for(map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin();
        it != addr_history[sampler_set].end(); it++) {
        if((it->second).lru == (SAMPLER_WAYS-1)) {
            lru_addr = it->first;
            break;
        }
    }
    addr_history[sampler_set].erase(lru_addr);
}

void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    for(map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin();
        it != addr_history[sampler_set].end(); it++) {
        if((it->second).lru < curr_lru) {
            (it->second).lru++;
            assert((it->second).lru < SAMPLER_WAYS);
        }
    }
}

// Hawkeye victim selection
uint32_t hawkeye_get_victim(uint32_t cpu, uint32_t set, const BLOCK *current_set) {
    // Look for maxRRPV line
    for (uint32_t i=0; i<LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    // Evict oldest cache-friendly line
    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i=0; i<LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rrip) {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }

    assert(lru_victim != -1);
    // Train predictor negatively on LRU evictions
    if(SAMPLED_SET(set)) {
        if(prefetched[set][lru_victim])
            prefetch_predictor->decrement(signatures[set][lru_victim]);
        else
            demand_predictor->decrement(signatures[set][lru_victim]);
    }
    return lru_victim;
}

// Hawkeye update replacement state
void hawkeye_update_replacement_state(uint32_t cpu, uint32_t set, uint32_t way,
                                     uint64_t paddr, uint64_t PC,
                                     uint64_t victim_addr, uint32_t type, uint8_t hit) {
    paddr = (paddr >> 6) << 6;

    if(type == PREFETCH) {
        if (!hit)
            prefetched[set][way] = true;
    } else
        prefetched[set][way] = false;

    // Ignore writebacks
    if (type == WRITEBACK)
        return;

    // If sampling, OPTgen sees accesses from sampled sets
    if(SAMPLED_SET(set)) {
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;
        assert(sampler_set < SAMPLER_SETS);

        // Line used before
        if((addr_history[sampler_set].find(sampler_tag) != addr_history[sampler_set].end()) && (type != PREFETCH)) {
            unsigned int curr_timer = perset_mytimer[set];
            if(curr_timer < addr_history[sampler_set][sampler_tag].last_quanta)
                curr_timer = curr_timer + TIMER_SIZE;
            bool wrap = ((curr_timer - addr_history[sampler_set][sampler_tag].last_quanta) > OPTGEN_VECTOR_SIZE);
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;

            if(!wrap && perset_optgen[set].should_cache(curr_quanta, last_quanta)) {
                if(addr_history[sampler_set][sampler_tag].prefetched)
                    prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                else
                    demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
            } else {
                if(addr_history[sampler_set][sampler_tag].prefetched)
                    prefetch_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
                else
                    demand_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
            }
            perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);
            addr_history[sampler_set][sampler_tag].prefetched = false;
        }
        // First time seeing this line
        else if(addr_history[sampler_set].find(sampler_tag) == addr_history[sampler_set].end()) {
            if(addr_history[sampler_set].size() == SAMPLER_WAYS)
                replace_addr_history_element(sampler_set);

            assert(addr_history[sampler_set].size() < SAMPLER_WAYS);
            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            if(type == PREFETCH) {
                addr_history[sampler_set][sampler_tag].mark_prefetch();
                perset_optgen[set].add_prefetch(curr_quanta);
            } else
                perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, SAMPLER_WAYS-1);
        }
        // Prefetch case
        else {
            assert(addr_history[sampler_set].find(sampler_tag) != addr_history[sampler_set].end());
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;
            if (perset_mytimer[set] - addr_history[sampler_set][sampler_tag].last_quanta < 5*NUM_CORE) {
                if(perset_optgen[set].should_cache(curr_quanta, last_quanta)) {
                    if(addr_history[sampler_set][sampler_tag].prefetched)
                        prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                    else
                        demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                }
            }
            addr_history[sampler_set][sampler_tag].mark_prefetch();
            perset_optgen[set].add_prefetch(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);
        }

        // Get Hawkeye's prediction
        bool new_prediction = demand_predictor->get_prediction(PC);
        if (type == PREFETCH)
            new_prediction = prefetch_predictor->get_prediction(PC);
        addr_history[sampler_set][sampler_tag].update(perset_mytimer[set], PC, new_prediction);
        addr_history[sampler_set][sampler_tag].lru = 0;
        perset_mytimer[set] = (perset_mytimer[set]+1) % TIMER_SIZE;
    }

    bool new_prediction = demand_predictor->get_prediction(PC);
    if (type == PREFETCH)
        new_prediction = prefetch_predictor->get_prediction(PC);

    signatures[set][way] = PC;

    // Set RRIP values and age cache-friendly lines
    if(!new_prediction)
        rrpv[set][way] = maxRRPV;
    else {
        rrpv[set][way] = 0;
        if(!hit) {
            bool saturated = false;
            for(uint32_t i=0; i<LLC_WAYS; i++)
                if (rrpv[set][i] == maxRRPV-1)
                    saturated = true;

            // Age all cache-friendly lines
            for(uint32_t i=0; i<LLC_WAYS; i++) {
                if (!saturated && rrpv[set][i] < maxRRPV-1)
                    rrpv[set][i]++;
            }
        }
        rrpv[set][way] = 0;
    }
}

// Hawkeye initialization
void hawkeye_init_replacement_state() {
    for (int i=0; i<LLC_SETS; i++) {
        for (int j=0; j<LLC_WAYS; j++) {
            rrpv[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS-2);
    }

    addr_history.resize(SAMPLER_SETS);
    for (int i=0; i<SAMPLER_SETS; i++)
        addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();
}
// ============================================================================
// END MODE A (HAWKEYE)
// ============================================================================


// ============================================================================
// MODE B: Stride-Run Quarantine with Tunable Rescue (SRQTR)
// - ±1/±2 forward stride run detector (window=4, threshold=2)
// - 75% soft-bypass (hard-tail insert) on streams; prefetch quarantine depth=3
// - Gated rescue: promote after 2 demand hits for stream PCs
// - TinyLFU (256x4b) PC coldness filter; demotion rate=2
// NOTE: Mode B uses ONLY rrpv_B state.
// ============================================================================

#define PC_TABLE_SIZE 256

// Packed-as-bytes metadata (conceptually 1–4 bits each; tiny footprint)
static uint16_t pc_last_line[PC_TABLE_SIZE];   // last line idx (10–16 bits used)
static uint8_t  pc_run_len[PC_TABLE_SIZE];     // run length (cap at 4)
static uint8_t  pc_hot[PC_TABLE_SIZE];         // 4-bit TinyLFU freq (stored in 8-bit)
static uint8_t  pc_rescue_hits[PC_TABLE_SIZE]; // 2-bit rescue counter
static uint8_t  pc_stream_tag[PC_TABLE_SIZE];  // 1-bit: PC currently classified as streaming

// Tunables (guided by surrogate)
static const uint8_t STREAM_WINDOW = 4;
static const uint8_t RUN_DETECT_THRESHOLD = 2;
static const uint8_t BYPASS_NUM = 3;   // 75% soft-bypass → tail insert at max
static const uint8_t BYPASS_DEN = 4;
static const uint8_t PREFETCH_QUARANTINE_DEPTH = 3;
static const uint8_t HITS_TO_PROMOTE = 2;
static const uint8_t DEMOTION_RATE = 2;      // every 2 demand fills, age set once
static const uint8_t HOT_THRESHOLD = 6;      // TinyLFU threshold for "hot" PCs

static uint64_t modeB_access_ctr = 0;
static uint32_t demote_tick = 0;

static inline uint32_t pc_index(uint64_t PC) {
    return (uint32_t)(PC) & (PC_TABLE_SIZE - 1);
}

static inline uint16_t line_index(uint64_t paddr) {
    return (uint16_t)((paddr >> 6) & 0xFFFF);
}

static inline bool is_forward_stride(uint16_t curr, uint16_t last) {
    // detect +1 or +2 with wrap-safe unsigned math
    return (curr == (uint16_t)(last + 1)) || (curr == (uint16_t)(last + 2));
}

static inline bool stream_detector_update(uint64_t PC, uint64_t paddr, uint32_t& idx_out) {
    uint32_t idx = pc_index(PC);
    idx_out = idx;
    uint16_t curr = line_index(paddr);
    uint16_t last = pc_last_line[idx];
    bool forward = false;

    if (last != 0xFFFF && is_forward_stride(curr, last)) {
        if (pc_run_len[idx] < STREAM_WINDOW) pc_run_len[idx]++;
        forward = true;
    } else {
        pc_run_len[idx] = 0;
    }
    pc_last_line[idx] = curr;

    // TinyLFU update (+ decay)
    if (pc_hot[idx] < 15) pc_hot[idx]++;
    modeB_access_ctr++;
    if ((modeB_access_ctr & 0xFFF) == 0) { // periodic light decay every 4096 accesses
        for (int i = 0; i < PC_TABLE_SIZE; i++) {
            if (pc_hot[i] > 0) pc_hot[i]--;
        }
    }

    // streaming classification if forward run length >= threshold
    bool is_stream = forward && (pc_run_len[idx] >= RUN_DETECT_THRESHOLD);
    if (is_stream) {
        pc_stream_tag[idx] = 1;
        pc_rescue_hits[idx] = 0;
    } else if (pc_run_len[idx] == 0) {
        pc_stream_tag[idx] = 0; // leave stream mode when pattern breaks
    }
    return is_stream;
}

static void mode_B_init_replacement_state() {
    for (int i = 0; i < LLC_SETS; i++) {
        for (int j = 0; j < LLC_WAYS; j++) {
            rrpv_B[i][j] = maxRRPV;
        }
    }
    for (int i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFF;
        pc_run_len[i] = 0;
        pc_hot[i] = 0;
        pc_rescue_hits[i] = 0;
        pc_stream_tag[i] = 0;
    }
    modeB_access_ctr = 0;
    demote_tick = 0;
}

static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // RRIP victim selection with on-demand aging until a maxRRPV appears
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv_B[set][i] == maxRRPV) return i;
        }
        // age all lines
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv_B[set][i] < maxRRPV) rrpv_B[set][i]++;
        }
    }
}

static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    uint32_t idx = 0;
    bool stream = stream_detector_update(PC, paddr, idx);

    // WRITEBACKs must never bypass; just place near tail conservatively
    if (type == WRITEBACK) {
        if (hit) {
            if (rrpv_B[set][way] > 0) rrpv_B[set][way]--;
            return;
        }
        rrpv_B[set][way] = maxRRPV - 1;
        return;
    }

    // On HIT: gated rescue for stream PCs; otherwise light promotion
    if (hit) {
        // Count rescue hits only for stream-tagged PCs
        if (pc_stream_tag[idx] && pc_rescue_hits[idx] < 3) pc_rescue_hits[idx]++;
        if (pc_stream_tag[idx] && pc_rescue_hits[idx] >= HITS_TO_PROMOTE) {
            rrpv_B[set][way] = 0; // promote to head after enough hits
            pc_stream_tag[idx] = 0; // stop forcing stream behavior after rescue
        } else {
            if (rrpv_B[set][way] > 0) rrpv_B[set][way]--; // mild promotion
        }
        return;
    }

    // On MISS (insertion policy)
    // Prefetch quarantine: insert at deep tail for first few touches
    if (type == PREFETCH) {
        // deepen quarantine by inserting at max repeatedly; gated by depth using run_len as a proxy
        uint8_t depth = (pc_run_len[idx] > PREFETCH_QUARANTINE_DEPTH) ? PREFETCH_QUARANTINE_DEPTH : pc_run_len[idx];
        rrpv_B[set][way] = maxRRPV; // hard-tail insert
        // soft aging of the set to accelerate eviction of quarantined lines
        if (((++demote_tick) % DEMOTION_RATE) == 0) {
            for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_B[set][i] < maxRRPV) rrpv_B[set][i]++;
        }
        (void)depth; // depth is implicit via repeated quarantined installs
        return;
    }

    // Demand insertion
    bool is_cold_pc = (pc_hot[idx] < HOT_THRESHOLD);

    if (stream && pc_rescue_hits[idx] < HITS_TO_PROMOTE) {
        // 75% soft-bypass (deterministic): 3/4 at max, 1/4 at max-1
        uint64_t h = CRC(paddr) & (BYPASS_DEN - 1);
        rrpv_B[set][way] = (h < BYPASS_NUM) ? maxRRPV : (maxRRPV - 1);

        // gentle set-wide demotion at configured rate
        if (((++demote_tick) % DEMOTION_RATE) == 0) {
            for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_B[set][i] < maxRRPV) rrpv_B[set][i]++;
        }
    } else {
        // Non-stream or rescued stream: admit based on PC hotness
        if (is_cold_pc) {
            rrpv_B[set][way] = maxRRPV - 1; // near-tail for noisy PCs
        } else {
            rrpv_B[set][way] = 2; // mid-head to allow short reuse (helps zeusmp)
        }
    }
}

// ============================================================================
// POLICY GLUE: Initialization, Victim Selection, State Update, Stats
// ============================================================================

void InitReplacementState() {
    // Initialize Mode A (Hawkeye baseline)
    hawkeye_init_replacement_state();
    // Initialize Mode B (SRQTR)
    mode_B_init_replacement_state();
    // Initialize selector state
    memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Prefer invalid ways
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        return mode_B_get_victim(set, current_set);
    } else {
        return hawkeye_get_victim(cpu, set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Update selector first (leaders feed the scores)
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates to preserve baseline guarantee
    if (use_mode_B) {
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() {
    // Intentionally blank
}

void PrintStats_Heartbeat() {
    // Intentionally blank
}