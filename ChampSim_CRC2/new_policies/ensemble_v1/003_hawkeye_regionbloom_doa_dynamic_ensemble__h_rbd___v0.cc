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
// MODE B: RegionBloom-DoA (RBD) - STRICT STATE ISOLATION (USE rrpv_B ONLY)
// ============================================================================
#define PC_TABLE_SIZE 512
static uint8_t pc_doastate[PC_TABLE_SIZE];  // low2: DoA state {0=cold,1=warm,2=hot,3=sticky}; high4: hit freq

// Region run/scan tracker (direct-mapped table keyed by 1KB region)
#define REGION_TABLE_ENTRIES 512
static uint32_t region_tag[REGION_TABLE_ENTRIES];     // tag of region id
static uint8_t  region_last_off[REGION_TABLE_ENTRIES]; // last 1KB-line offset (0..15)
static uint8_t  region_run_len[REGION_TABLE_ENTRIES];  // 0..7 small run length

// Rolling Bloom for streaming regions (keeps regions in "scan quarantine" briefly)
#define BLOOM_BITS 4096
static uint64_t bloom_bits[BLOOM_BITS/64];
static uint64_t bloom_access_ctr = 0;
static const uint64_t BLOOM_RESET_PERIOD = 1ULL << 18; // periodic decay (~256K accesses)

// Per-line short-burst rescue counter (2 bits per line packed into a byte here)
static uint8_t burst_hits_B[LLC_SETS][LLC_WAYS];

static inline void bloom_clear() {
    memset(bloom_bits, 0, sizeof(bloom_bits));
}
static inline uint32_t bloom_h1(uint64_t x) { return (uint32_t)(CRC(x) & (BLOOM_BITS-1)); }
static inline uint32_t bloom_h2(uint64_t x) { return (uint32_t)(((x * 11400714819323198485ull) >> 17) & (BLOOM_BITS-1)); }
static inline void bloom_set_region(uint64_t region_id) {
    uint32_t b1 = bloom_h1(region_id), b2 = bloom_h2(region_id);
    bloom_bits[b1 >> 6] |= (1ull << (b1 & 63));
    bloom_bits[b2 >> 6] |= (1ull << (b2 & 63));
}
static inline bool bloom_test_region(uint64_t region_id) {
    uint32_t b1 = bloom_h1(region_id), b2 = bloom_h2(region_id);
    bool t1 = (bloom_bits[b1 >> 6] >> (b1 & 63)) & 1ull;
    bool t2 = (bloom_bits[b2 >> 6] >> (b2 & 63)) & 1ull;
    return t1 & t2;
}

static void mode_B_init_replacement_state() {
    for (int i = 0; i < LLC_SETS; i++) {
        for (int j = 0; j < LLC_WAYS; j++) {
            rrpv_B[i][j] = maxRRPV;
            burst_hits_B[i][j] = 0;
        }
    }
    memset(pc_doastate, 0, sizeof(pc_doastate));
    memset(region_tag, 0xFF, sizeof(region_tag));
    memset(region_last_off, 0, sizeof(region_last_off));
    memset(region_run_len, 0, sizeof(region_run_len));
    bloom_clear();
    bloom_access_ctr = 0;
}

static inline uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // Find immediate maxRRPV victim
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] == maxRRPV) return i;
    }
    // Otherwise evict the line with largest RRPV_B (tie -> highest index as in RRIP)
    uint32_t maxv = 0, vic = 0;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] >= maxv) { maxv = rrpv_B[set][i]; vic = i; }
    }
    return vic;
}

static inline void pc_doastate_on_hit(uint64_t PC) {
    uint32_t idx = (uint32_t)(PC & (PC_TABLE_SIZE - 1));
    uint8_t s = pc_doastate[idx] & 0x3;
    uint8_t f = pc_doastate[idx] >> 4;
    if (f < 15) f++;
    if (s < 3 && f >= 2) s = 2;        // become hot after 2 hits
    if (f >= 8) s = 3;                 // sticky hot after enough hits
    pc_doastate[idx] = (uint8_t)((f << 4) | (s & 0x3));
}
static inline void pc_doastate_on_miss(uint64_t PC) {
    uint32_t idx = (uint32_t)(PC & (PC_TABLE_SIZE - 1));
    uint8_t s = pc_doastate[idx] & 0x3;
    uint8_t f = pc_doastate[idx] >> 4;
    if (f > 0) f--;                    // gentle decay
    if (f == 0 && s > 0) s--;          // cool down
    pc_doastate[idx] = (uint8_t)((f << 4) | (s & 0x3));
}
static inline bool pc_is_hot(uint64_t PC) {
    uint32_t idx = (uint32_t)(PC & (PC_TABLE_SIZE - 1));
    uint8_t s = pc_doastate[idx] & 0x3;
    return (s >= 2);
}

// Region run detector on 1KB region
static inline bool region_stream_detect(uint64_t paddr) {
    uint64_t region_id = paddr >> 10; // 1KB region
    uint32_t idx = (uint32_t)(region_id & (REGION_TABLE_ENTRIES - 1));
    uint32_t tag = (uint32_t)(region_id >> 9);
    uint8_t off = (uint8_t)((paddr >> 6) & 0xF); // 16 lines per 1KB

    bool stream = false;
    if (region_tag[idx] == tag) {
        uint8_t last = region_last_off[idx];
        if (off == (uint8_t)(last + 1) || off == (uint8_t)(last + 2)) {
            if (region_run_len[idx] < 7) region_run_len[idx]++;
            if (region_run_len[idx] >= 2) stream = true;
        } else {
            // reset/soft-decay on non-forward step
            if (region_run_len[idx] > 0) region_run_len[idx]--;
        }
        region_last_off[idx] = off;
    } else {
        // new region allocation
        region_tag[idx] = tag;
        region_last_off[idx] = off;
        region_run_len[idx] = 1;
        stream = false;
    }

    // Rolling Bloom quarantine: remember streaming regions for a short epoch
    if (stream) bloom_set_region(region_id);
    if (bloom_access_ctr++ % BLOOM_RESET_PERIOD == 0) bloom_clear();
    if (!stream && bloom_test_region(region_id)) stream = true;

    return stream;
}

static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    // Demand hits: promote and train PC hotness
    if (hit) {
        if (rrpv_B[set][way] > 0) rrpv_B[set][way]--;
        if (burst_hits_B[set][way] < 3) burst_hits_B[set][way]++;
        if (burst_hits_B[set][way] >= 2) rrpv_B[set][way] = 0; // burst rescue
        if (type != PREFETCH) pc_doastate_on_hit(PC);
        return;
    }

    // Miss/fill path
    if (type == WRITEBACK) {
        // Never bypass on writeback; just leave state untouched
        return;
    }

    // Update classifiers on miss
    if (type != PREFETCH) pc_doastate_on_miss(PC);

    bool is_stream = region_stream_detect(paddr);
    bool is_hot_pc = pc_is_hot(PC);

    // Insertion policy (no bypass): quarantine prefetches and streams
    uint32_t ins_rrpv = maxRRPV; // deepest by default for quarantine
    if (type != PREFETCH) {
        if (!is_stream) {
            ins_rrpv = is_hot_pc ? 0 : (maxRRPV - 1);  // hot -> MRU, cold -> near-tail
        } else {
            ins_rrpv = maxRRPV; // stream/scans tail-insert
        }
    }
    rrpv_B[set][way] = ins_rrpv;
    burst_hits_B[set][way] = 0;

    // Gentle aging of friendly lines in set when inserting MRU (balance occupancy)
    if (ins_rrpv == 0) {
        bool saturated = false;
        for (uint32_t i = 0; i < LLC_WAYS; i++)
            if (rrpv_B[set][i] == maxRRPV - 1) saturated = true;
        if (!saturated) {
            for (uint32_t i = 0; i < LLC_WAYS; i++)
                if (rrpv_B[set][i] < maxRRPV - 1) rrpv_B[set][i]++;
        }
        rrpv_B[set][way] = 0;
    }
}

// ============================================================================
// TOP-LEVEL CHAMPSIM HOOKS
// ============================================================================
void InitReplacementState() {
    hawkeye_init_replacement_state();   // Initialize Mode A (baseline)
    mode_B_init_replacement_state();    // Initialize Mode B state
    memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Check for invalid ways first (always fill if space exists)
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        return mode_B_get_victim(set, current_set);  // Mode B victim
    } else {
        return hawkeye_get_victim(cpu, set, current_set);  // Mode A victim
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                           uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                           uint32_t type, uint8_t hit) {
    // Always update selector (leaders measure cleanly due to strict gating)
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates: update only the chosen mode's state (no contamination)
    if (use_mode_B) {
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }