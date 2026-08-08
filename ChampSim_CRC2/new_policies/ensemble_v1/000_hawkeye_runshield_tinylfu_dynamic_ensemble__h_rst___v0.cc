#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>  // for log2()
#include "../inc/champsim_crc2.h"  // for BLOCK, PREFETCH, WRITEBACK

using namespace std;

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16

// ============================================================================
// MANDATORY SELECTOR (Leader-Follower, strict gating)
// ============================================================================
#define LEADER_SETS_A 32
#define LEADER_SETS_B 32

static const int32_t BIAS = 50;       // Mode B requires +50 hits advantage
static const int8_t THRESHOLD = 6;    // Per-set confidence gate (≥6/8)
static const int8_t CONF_MAX = 8;
static bool prefer_mode_B = false;

#define LEADER_A(set) ((set) < LEADER_SETS_A)
#define LEADER_B(set) ((set) >= LEADER_SETS_A && (set) < (LEADER_SETS_A + LEADER_SETS_B))

static int8_t mode_confidence[LLC_SETS];  // [0..8]
static int32_t leaderA_score = 0;         // count hits on Leader A sets
static int32_t leaderB_score = 0;         // count hits on Leader B sets

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Always Hawkeye on Leader A
    if (LEADER_B(set)) return true;   // Always Mode B on Leader B
    int32_t score_advantage = leaderB_score - leaderA_score;
    return (score_advantage > BIAS) && (mode_confidence[set] >= THRESHOLD);
}

static void update_selector(uint32_t set, uint8_t hit) {
    if (LEADER_A(set) && hit) leaderA_score++;
    if (LEADER_B(set) && hit) leaderB_score++;

    prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));

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
// MODE B: RunShield-TinyLFU (separate state; uses only rrpv_B)
// ============================================================================
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 2-bit used, stored in 8-bit
static uint8_t  pc_run_len[PC_TABLE_SIZE];     // 3-bit used, stored in 8-bit
static uint8_t  pc_freq[PC_TABLE_SIZE];        // 4-bit TinyLFU counters
static uint8_t  pc_hit_streak[PC_TABLE_SIZE];  // 2-bit multi-hit rescue
static uint8_t  set_pressure[LLC_SETS];        // per-set thrash pressure

static uint64_t modeB_access_ctr = 0;
static const uint32_t DECAY_PERIOD = 4096;

static inline uint32_t pc_index(uint64_t PC) { return (uint32_t)(CRC(PC) & (PC_TABLE_SIZE - 1)); }

static inline bool detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint32_t line = (uint32_t)((paddr >> 6) & 0xFFFF); // compact line id
    bool is_stream = false;

    if (pc_last_line[idx] != 0xFFFF) {
        int32_t delta = (int32_t)line - (int32_t)pc_last_line[idx];
        bool fwd = (delta == 1) || (delta == 2) || (delta == 4);
        if (fwd) {
            if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
            if (pc_run_len[idx] < 7) pc_run_len[idx]++;
        } else {
            // decay confidence/run length on pattern break
            if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
            if (pc_run_len[idx] > 0) pc_run_len[idx]--;
        }
    } else {
        pc_stride_conf[idx] = 0;
        pc_run_len[idx] = 0;
    }

    pc_last_line[idx] = (uint16_t)line;
    is_stream = (pc_stride_conf[idx] >= 2) && (pc_run_len[idx] >= 2);
    return is_stream;
}

static inline bool tiny_lfu_admit(uint64_t PC) {
    uint32_t idx = pc_index(PC);
    if (pc_freq[idx] < 15) pc_freq[idx]++; // saturating 4-bit

    // periodic decay
    modeB_access_ctr++;
    if ((modeB_access_ctr & (DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) pc_freq[i] >>= 1;
        for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) if (pc_hit_streak[i] > 0) pc_hit_streak[i]--; // soft decay streak
    }

    return (pc_freq[idx] >= 8); // hot threshold
}

static void mode_B_init_replacement_state() {
    for (int s = 0; s < LLC_SETS; s++) {
        for (int w = 0; w < LLC_WAYS; w++) rrpv_B[s][w] = maxRRPV;
        set_pressure[s] = 0;
    }
    for (int i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFF;
        pc_stride_conf[i] = 0;
        pc_run_len[i] = 0;
        pc_freq[i] = 0;
        pc_hit_streak[i] = 0;
    }
}

static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // Prefer any line at maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] == maxRRPV) return i;
    }
    // Else pick the largest RRPV (oldest among ties)
    uint32_t victim = 0, best = 0;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] >= best) { best = rrpv_B[set][i]; victim = i; }
    }
    return victim;
}

static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    if (type == WRITEBACK) return; // never bypass or change on writebacks

    uint32_t pidx = pc_index(PC);
    bool is_stream = detect_stream(PC, paddr);
    bool hot_pc = tiny_lfu_admit(PC);

    // update per-set pressure
    if (hit) {
        if (set_pressure[set] > 0) set_pressure[set]--;
        if (pc_hit_streak[pidx] < 3) pc_hit_streak[pidx]++;  // toward rescue
        // promotion on hit
        if (rrpv_B[set][way] > 0) {
            // aggressive promotion if already showed reuse
            if (pc_hit_streak[pidx] >= 2) rrpv_B[set][way] = 0;
            else rrpv_B[set][way] = (rrpv_B[set][way] >= 2) ? (rrpv_B[set][way] - 2) : 0;
        }
        return;
    } else {
        if (set_pressure[set] < 15) set_pressure[set]++;
        // on fill/insertion
        uint32_t ins_rrpv = maxRRPV; // default: tail (quarantine)
        if (type == PREFETCH) {
            ins_rrpv = maxRRPV; // quarantine prefetches
        } else {
            // demand: decide based on stream, hotness, pressure, rescue
            if (pc_hit_streak[pidx] >= 2 && !is_stream) {
                ins_rrpv = 0; // multi-hit rescue: insert at head
            } else if (is_stream || (set_pressure[set] >= 8) || !hot_pc) {
                ins_rrpv = maxRRPV; // quarantine likely scans/noisy PCs
            } else {
                ins_rrpv = 2; // moderately friendly
            }
        }
        rrpv_B[set][way] = ins_rrpv;
        // slight aging of others on new friendly fill (anti-bullying)
        if (ins_rrpv <= 2) {
            for (uint32_t i = 0; i < LLC_WAYS; i++) {
                if (i == way) continue;
                if (rrpv_B[set][i] < maxRRPV) rrpv_B[set][i]++;
            }
        }
    }
}

// ============================================================================
// WRAPPER: Replacement interface (exclusive updates)
// ============================================================================
void InitReplacementState() {
    hawkeye_init_replacement_state();  // Mode A init
    mode_B_init_replacement_state();   // Mode B init
    // Selector init
    memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0; leaderB_score = 0; prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Honor invalid ways first
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }
    bool use_B = should_use_mode_B(set);
    if (use_B) return mode_B_get_victim(set, current_set);
    return hawkeye_get_victim(cpu, set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Update selector on every event
    update_selector(set, hit);

    bool use_B = should_use_mode_B(set);
    if (use_B) {
        // Mode B only
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    } else {
        // Mode A only
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }