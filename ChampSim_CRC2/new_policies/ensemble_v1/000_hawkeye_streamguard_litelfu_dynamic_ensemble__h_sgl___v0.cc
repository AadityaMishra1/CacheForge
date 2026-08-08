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
// MODE A: FULL HAWKEYE IMPLEMENTATION (BASELINE - DO NOT MODIFY)
// ============================================================================

// Hawkeye data structures
#define maxRRPV 7
uint32_t rrpv[LLC_SETS][LLC_WAYS];

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
        uint32_t sampler_set2 = (paddr >> 6) % SAMPLER_SETS; (void)sampler_set2; // silence unused if optimized
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
// DYNAMIC ENSEMBLE: Selector + Mode B (StreamGuard-LiteLFU)
// ============================================================================

// Leader–Follower: reuse Hawkeye's SAMPLED_SET; split by set parity
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// Selector tunables
static const int32_t BIAS = 0;            // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096; // accesses per global decision
static const int8_t THRESHOLD = 2;        // per-set confidence threshold
static const int8_t CONF_MAX = 7;

// Global selector state
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A leaders
static int32_t leaderB_score = 0;  // Hits - misses on Mode B leaders
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS]; // Soft per-set drift toward global pref

// Mode B: tunables
static const uint8_t RRPV_DISTANT = maxRRPV - 1; // 6
static const uint8_t RRPV_PROTECT = 2;           // protected but not MRU

// Stream detector
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3 (2-bit conceptual)
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7 (3-bit conceptual)
static const uint8_t RUN_LEN_TO_STREAM = 2;    // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ = 2;      // min confidence

// TinyLFU PC frequency (4-bit conceptual)
#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE];
static const uint8_t FREQ_THRESHOLD = 6;
static const uint32_t FREQ_AGE_PERIOD = 8192;

// Multi-hit rescue
static const uint8_t HITS_TO_PROMOTE = 2;  // rapid promotion for short reuse

// Per-line lightweight metadata (conceptually bit-packed)
static uint8_t line_hits[LLC_SETS][LLC_WAYS];       // 2-bit conceptual (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit conceptual
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];     // 1-bit conceptual

// Helpers
static inline uint32_t pc_tab_index(uint64_t PC) { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }
static inline uint32_t pc_freq_index(uint64_t PC) { return (uint32_t)PC & (PC_FREQ_SIZE - 1); }
static inline uint64_t line_addr(uint64_t paddr) { return (paddr >> 6); }

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

// Predict stream without updating state (used in victim/bypass decision)
static inline bool predict_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t last = pc_last_line[idx];
    if (last == 0xFFFFu) return false;
    int32_t diff = (int32_t)((uint16_t)paddr_line) - (int32_t)last;
    bool forward = (diff == 1) || (diff == 2);
    uint8_t frun = pc_forward_run[idx];
    uint8_t conf = pc_stride_conf[idx];
    if (forward) {
        frun = (frun < 7) ? (frun + 1) : 7;
        conf = (conf < 3) ? (conf + 1) : 3;
    } else {
        frun = 0;
        conf = (conf > 0) ? (uint8_t)(conf - 1) : 0;
    }
    return (frun >= RUN_LEN_TO_STREAM) && (conf >= STRIDE_CONF_REQ);
}

// Update stream detector state (called on UpdateReplacementState)
static inline void update_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t last = pc_last_line[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        int32_t diff = (int32_t)((uint16_t)paddr_line) - (int32_t)last;
        forward = (diff == 1) || (diff == 2);
    }
    if (forward) {
        if (pc_forward_run[idx] < 7) pc_forward_run[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_forward_run[idx] = 0;
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
    }
    pc_last_line[idx] = (uint16_t)(paddr_line & 0xFFFFu);
}

// Selector update
static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;

    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    if ((access_count % EVAL_PERIOD) == 0) {
        bool new_pref = (leaderB_score > (leaderA_score + BIAS));
        prefer_mode_B = new_pref;
        // gentle decay to avoid runaway
        leaderA_score >>= 1;
        leaderB_score >>= 1;
    }

    // Drift this set toward global preference
    if (!SAMPLED_SET(set)) {
        if (prefer_mode_B) {
            if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
        } else {
            if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
        }
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// Mode B victim (with optional bypass; never bypass WRITEBACK)
static inline uint32_t mode_B_get_victim(uint32_t set, const BLOCK* current_set,
                                         uint64_t PC, uint64_t paddr, uint32_t type) {
    uint64_t la = line_addr(paddr);
    bool predicted_stream = predict_stream(PC, la);
    bool hot_pc = freq_is_hot(PC);

    if ((type != WRITEBACK) && predicted_stream && !hot_pc) {
        // streaming from cold PC: bypass to avoid pollution
        return 16; // bypass
    }

    // SRRIP victim on shared RRIP state
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV) return i;

    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++)
            if (rrpv[set][i] == maxRRPV) return i;
        for (uint32_t i = 0; i < LLC_WAYS; i++)
            if (rrpv[set][i] < maxRRPV) rrpv[set][i]++;
    }
}

// Mode B state update (overrides Hawkeye insertion/promotion as needed)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr,
                                 uint64_t PC, uint32_t type, uint8_t hit) {
    uint64_t la = line_addr(paddr);

    // PC telemetry
    freq_on_access(PC);
    periodic_freq_age();
    update_stream(PC, la);

    // WRITEBACK: never bypass; modest priority; do not count as demand reuse
    if (type == WRITEBACK) {
        rrpv[set][way] = RRPV_DISTANT;
        return;
    }

    // Prefetch quarantine: always tail
    if (type == PREFETCH) {
        line_pref_q[set][way] = 1;
        rrpv[set][way] = maxRRPV;
        return;
    }

    // Demand access
    bool hot = freq_is_hot(PC);
    bool is_stream = predict_stream(PC, la);

    if (!hit) {
        // New fill
        line_hits[set][way] = 0;
        line_stream_tag[set][way] = is_stream ? 1 : 0;
        line_pref_q[set][way] = 0;

        if (is_stream && !hot) {
            // admitted (not bypassed) but quarantined at tail
            rrpv[set][way] = maxRRPV;
        } else if (hot) {
            rrpv[set][way] = RRPV_PROTECT; // protect promising lines
        } else {
            rrpv[set][way] = RRPV_DISTANT; // distant for cold/noisy PCs
        }
    } else {
        // Hit handling
        if (line_pref_q[set][way]) {
            // demand hit on prefetched line: clear quarantine and rescue quickly
            line_pref_q[set][way] = 0;
            rrpv[set][way] = (rrpv[set][way] > 0) ? (rrpv[set][way] - 1) : 0;
        }

        if (line_hits[set][way] < 3) line_hits[set][way]++;

        // Multi-hit rescue: once short reuse appears, promote aggressively
        if (line_hits[set][way] >= HITS_TO_PROMOTE) {
            rrpv[set][way] = (rrpv[set][way] > 1) ? (rrpv[set][way] - 2) : 0;
        } else {
            // gentle SRRIP-style promotion
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
    }
}

// ============================================================================
// Top-level ChampSim API
// ============================================================================

void InitReplacementState() {
    // Mode A
    hawkeye_init_replacement_state();

    // Mode B init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            line_hits[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    memset(pc_freq, 0, sizeof(pc_freq));

    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // First, honor invalid ways
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_B = should_use_mode_B(set);

    if (use_B) {
        uint32_t v = mode_B_get_victim(set, current_set, PC, paddr, type);
        if (v == 16) return 16; // bypass (never used for writeback due to guard)
        return v;
    } else {
        return hawkeye_get_victim(cpu, set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                           uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                           uint32_t type, uint8_t hit) {
    // Update selector accounting first
    update_selector(set, hit);

    // Always train Hawkeye (predictors, sampler, and its RRIP state)
    hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Apply Mode B overrides if selected for this set/epoch
    if (should_use_mode_B(set)) {
        mode_B_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }