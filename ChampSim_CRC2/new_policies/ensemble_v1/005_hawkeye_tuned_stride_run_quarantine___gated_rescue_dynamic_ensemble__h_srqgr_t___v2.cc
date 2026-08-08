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
static const int8_t THRESHOLD = 6;     // REQUIRED: 75% confidence (6/8) to switch
static const int8_t CONF_MAX = 8;      // REQUIRED: Max confidence level
static bool prefer_mode_B = false;     // REQUIRED: Default to Hawkeye

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
// MODE B: Tuned Stride-Run Quarantine & Gated Rescue (SRQGR-T)
// ============================================================================
static inline uint32_t pc_hash(uint64_t PC) { return (uint32_t)((PC ^ (PC >> 2) ^ (PC >> 5)) & 0x1FF); } // 512-1
#define PC_TABLE_SIZE 512

// Tiny, per-PC state
static uint16_t pc_last_line[PC_TABLE_SIZE]; // last line index (16b)
static uint8_t  pc_runlen[PC_TABLE_SIZE];    // forward run length (0..3)
static uint8_t  pc_hits[PC_TABLE_SIZE];      // demand hits for rescue gate (0..3)
static uint8_t  pc_lfu[PC_TABLE_SIZE];       // 4b LFU (0..15), stored in 8b
static uint8_t  pc_prefq[PC_TABLE_SIZE];     // prefetch quarantine depth (0..3)

// Tunables (guided by feedback)
static const uint8_t RUN_REQUIRED = 2;        // two forward steps to tag as stream
static const uint8_t HITS_TO_PROMOTE = 2;     // two demand hits to rescue
static const uint8_t LFU_THRESHOLD = 8;       // admit only if warm/hot
static const uint8_t PREF_Q_DEPTH = 3;        // quarantine depth for prefetch
static const uint8_t MAX_RUNLEN = 3;          // saturate

// Initialize Mode B state
static void mode_B_init_replacement_state() {
    for (int i = 0; i < LLC_SETS; i++)
        for (int j = 0; j < LLC_WAYS; j++)
            rrpv_B[i][j] = maxRRPV;
    for (int i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFF;
        pc_runlen[i] = 0;
        pc_hits[i] = 0;
        pc_lfu[i] = 0;
        pc_prefq[i] = 0;
    }
}

// Detect ±1/±2 forward stride-run and update per-PC state.
// Returns true if the PC is currently considered a stream.
static inline bool mode_B_update_stride(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_hash(PC);
    uint16_t line = (uint16_t)((paddr >> 6) & 0xFFFF);
    if (pc_last_line[idx] != 0xFFFF) {
        int32_t delta = (int32_t)line - (int32_t)pc_last_line[idx];
        if (delta == 1 || delta == 2) {
            if (pc_runlen[idx] < MAX_RUNLEN) pc_runlen[idx]++;
        } else {
            pc_runlen[idx] = 0;
        }
    }
    pc_last_line[idx] = line;
    return (pc_runlen[idx] >= RUN_REQUIRED);
}

static inline void mode_B_touch_prefetch(uint64_t PC) {
    uint32_t idx = pc_hash(PC);
    pc_prefq[idx] = PREF_Q_DEPTH; // quarantine future insertions for this PC
}

static inline void mode_B_on_access(uint64_t PC, uint8_t is_demand, uint8_t is_hit) {
    uint32_t idx = pc_hash(PC);
    // TinyLFU: increment on any access, saturate at 15
    if (pc_lfu[idx] < 15) pc_lfu[idx]++;
    // Mild demotion on noise: on demand miss, penalize noisy PCs slightly
    if (is_demand && !is_hit && pc_lfu[idx] > 0) pc_lfu[idx]--;
    // Drain quarantine on demand accesses
    if (is_demand && pc_prefq[idx] > 0) pc_prefq[idx]--;
    // Rescue gating: count demand hits while stream-tagged
    if (is_demand && is_hit && pc_hits[idx] < 3) pc_hits[idx]++;
}

static inline void mode_B_reset_rescue(uint64_t PC) {
    uint32_t idx = pc_hash(PC);
    pc_hits[idx] = 0;
}

// Mode B victim selection (SRRIP-style)
static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // First try to find a line with max RRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv_B[set][i] == maxRRPV) return i;

    // If none, increment all RRPVs (age) until a victim appears
    for (int round = 0; round < maxRRPV; round++) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv_B[set][i] < maxRRPV) rrpv_B[set][i]++;
        }
        for (uint32_t i = 0; i < LLC_WAYS; i++)
            if (rrpv_B[set][i] == maxRRPV) return i;
    }

    // Fallback: choose the way with the highest RRPV
    uint32_t victim = 0, best = 0;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] >= best) { best = rrpv_B[set][i]; victim = i; }
    }
    return victim;
}

static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    // Update PC-side metadata first
    bool is_pref = (type == PREFETCH);
    bool is_wb   = (type == WRITEBACK);
    bool is_dmd  = (!is_pref && !is_wb);

    // Update stride/run detector with this access
    bool is_stream = mode_B_update_stride(PC, paddr);

    if (is_pref) {
        mode_B_touch_prefetch(PC); // quarantine this PC's future inserts
    }

    mode_B_on_access(PC, is_dmd, hit);

    // RRPV update on hits
    if (hit) {
        if (rrpv_B[set][way] > 0) rrpv_B[set][way]--;
        return;
    }

    // Miss/Fill: choose insertion policy
    uint32_t ins_rrpv = 2; // default mid insertion for friendly cases

    // WRITEBACK must not bypass
    if (is_wb) {
        ins_rrpv = (maxRRPV - 1);
    } else if (is_pref) {
        // Quarantine prefetch fills aggressively
        ins_rrpv = maxRRPV;
    } else {
        // Demand access: combine stream tagging, quarantine, LFU, and rescue
        uint32_t idx = pc_hash(PC);
        bool in_quarantine = (pc_prefq[idx] > 0);
        bool rescued = (pc_hits[idx] >= HITS_TO_PROMOTE);

        if (is_stream || in_quarantine) {
            if (!rescued) {
                ins_rrpv = maxRRPV;        // hard tail insert / soft-bypass
            } else {
                ins_rrpv = 2;              // gated rescue: allow moderate protection
            }
        } else if (pc_lfu[idx] < LFU_THRESHOLD) {
            ins_rrpv = maxRRPV;            // cold/noisy PCs: avoid polluting
        } else {
            ins_rrpv = 2;                  // warm/hot demand
        }

        // If we insert as stream-protected, reset rescue counter so it must re-earn
        if (ins_rrpv == maxRRPV) mode_B_reset_rescue(PC);
    }

    rrpv_B[set][way] = ins_rrpv;
}

// ============================================================================
// TOP-LEVEL WRAPPER: Init / Victim / Update / Stats
// ============================================================================
void InitReplacementState() {
    // Mode A: Hawkeye baseline
    hawkeye_init_replacement_state();

    // Mode B: SRQGR-T
    mode_B_init_replacement_state();

    // Selector init
    memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Always honor invalid ways first
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        return mode_B_get_victim(set, current_set);              // Mode B victim
    } else {
        return hawkeye_get_victim(cpu, set, current_set);        // Hawkeye victim
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Update leader/follower selector first
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates: only one mode touches its state per access
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