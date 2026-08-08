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
// LEADER-FOLLOWER SAMPLING (selector-only; avoid clash with Hawkeye macro)
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET_SEL(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A_SEL(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B_SEL(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 1u); }

// ============================================================================
// MODE SELECTOR
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];  // Per-set soft confidence

// Selector tunables (telemetry knobs)
// BIAS >=0 favors Hawkeye; EVAL_PERIOD is epoch; THRESHOLD gate for followers
static const int32_t BIAS = 2;
static const uint32_t EVAL_PERIOD = 4096;
static const int8_t THRESHOLD = 2;
static const int8_t CONF_MAX = 8;

static inline void update_selector(uint32_t set, uint8_t hit) {
    access_count++;

    if (SAMPLED_SET_SEL(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A_SEL(set)) leaderA_score += delta;
        if (LEADER_B_SEL(set)) leaderB_score += delta;
    }

    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        leaderA_score = 0;
        leaderB_score = 0;
    }

    if (!SAMPLED_SET_SEL(set)) {
        if (prefer_mode_B) {
            if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
        } else {
            if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
        }
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A_SEL(set)) return false;  // Force Mode A on A leaders
    if (LEADER_B_SEL(set)) return true;   // Force Mode B on B leaders
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

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
// MODE B: StrideScan-LiteLFU (complements Hawkeye on streams/scans + hot bursts)
// ============================================================================

// Small PC tables (power of two)
#define PC_TABLE_SIZE 512
#define PC_INDEX(PC) (static_cast<uint32_t>((PC) & (PC_TABLE_SIZE - 1)))

// Stride/scan detector state
static uint16_t pc_last_line[PC_TABLE_SIZE];   // last line index (16-bit)
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 2-3 bits used (0..3)
static uint8_t  pc_run_len[PC_TABLE_SIZE];     // 3 bits (0..7)
// TinyLFU PC hotness (4-bit conceptual, stored in 8-bit)
static uint8_t  pc_freq[PC_TABLE_SIZE];        // 0..15; decay periodically

// Per-line rescue: count hits to trigger rapid promotion (2-bit conceptual)
static uint8_t  rescue_hits[LLC_SETS][LLC_WAYS]; // use lower 2 bits

// Mode B tunables (telemetry knobs)
// Stream: +1/+2 forward run length and confidence thresholds
static const uint8_t STREAM_CONF_TH = 2;
static const uint8_t RUNLEN_TH = 3;
// Frequency gate
static const uint8_t FREQ_HOT_TH = 8;
static const uint32_t FREQ_DECAY_PERIOD = 4096;
// Insertion RRIP levels
static const uint8_t RRPV_TAIL = maxRRPV;
static const uint8_t RRPV_NEAR_TAIL = maxRRPV - 1;
static const uint8_t RRPV_MED = 2;
static const uint8_t RESCUE_HITS_TO_PROMOTE = 2;

// Predict stream/scan without mutating state (for GetVictim decisions)
static inline bool predict_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = PC_INDEX(PC);
    uint32_t line = (uint32_t)((paddr >> 6) & 0xFFFFu);
    uint16_t last = pc_last_line[idx];
    if (last == 0xFFFFu) return false;
    int32_t delta = (int32_t)line - (int32_t)last;
    bool forward_small = (delta == 1) || (delta == 2);
    bool strong = (pc_stride_conf[idx] >= STREAM_CONF_TH) && (pc_run_len[idx] >= RUNLEN_TH);
    return forward_small && strong;
}

static inline bool is_hot_pc(uint64_t PC) {
    return pc_freq[PC_INDEX(PC)] >= FREQ_HOT_TH;
}

static inline bool mode_B_should_bypass(uint64_t PC, uint64_t paddr, uint32_t type) {
    if (type == WRITEBACK) return false; // never bypass on writeback
    bool streamy = predict_stream(PC, paddr);
    bool hot = is_hot_pc(PC);
    return (streamy && !hot);
}

// Mode B victim: do NOT touch Hawkeye predictors
static inline uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // Prefer any line at maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    // Otherwise, evict line with highest RRPV (ties to last)
    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rrip) {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }
    assert(lru_victim != -1);
    return (uint32_t)lru_victim;
}

// Update PC stream stats and TinyLFU
static inline void mode_B_update_pc_tables(uint64_t PC, uint64_t paddr) {
    uint32_t idx = PC_INDEX(PC);
    uint32_t line = (uint32_t)((paddr >> 6) & 0xFFFFu);
    uint16_t last = pc_last_line[idx];

    // Stride/run update
    if (last != 0xFFFFu) {
        int32_t delta = (int32_t)line - (int32_t)last;
        if (delta == 1 || delta == 2) {
            if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
            if (pc_run_len[idx] < 7) pc_run_len[idx]++;
        } else {
            // Break in stream: decay run/conf
            if (pc_run_len[idx] > 0) pc_run_len[idx]--;
            if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
        }
    }
    pc_last_line[idx] = (uint16_t)line;

    // TinyLFU increment (saturating 4-bit conceptual)
    if (pc_freq[idx] < 15) pc_freq[idx]++;
    // Periodic decay
    if ((access_count % FREQ_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) pc_freq[i] >>= 1;
    }
}

// Mode B replacement state update (exclusive path)
static inline void mode_B_update_replacement_state(
    uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit)
{
    // Ignore writebacks (no allocation policy changes)
    if (type == WRITEBACK) return;

    // Update PC stats (stream detector + LFU)
    mode_B_update_pc_tables(PC, paddr);

    if (hit) {
        // Multi-hit rescue: quickly promote short-reuse lines
        uint8_t cnt = (rescue_hits[set][way] & 0x3);
        if (cnt < 3) rescue_hits[set][way] = cnt + 1;
        if (rescue_hits[set][way] >= RESCUE_HITS_TO_PROMOTE) {
            rrpv[set][way] = 0; // MRU
        } else {
            // Mild promotion
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
        return;
    }

    // Miss/fill insertion policy
    rescue_hits[set][way] = 0;

    bool streamy = predict_stream(PC, paddr);
    bool hot = is_hot_pc(PC);

    if (type == PREFETCH) {
        // Quarantine prefetches: keep short-lived unless proven hot
        rrpv[set][way] = RRPV_TAIL;
        return;
    }

    if (streamy && !hot) {
        // Scan/stream pollution control
        rrpv[set][way] = RRPV_TAIL; // hard tail insert (bypass handled in GetVictim)
    } else if (hot) {
        // Hot demands get moderate insertion
        rrpv[set][way] = RRPV_MED;
    } else {
        // Default: near tail to be conservative
        rrpv[set][way] = RRPV_NEAR_TAIL;
    }
}

// ============================================================================
// TOP-LEVEL ENSEMBLE HOOKS
// ============================================================================
void InitReplacementState() {
    // Mode A (Hawkeye)
    hawkeye_init_replacement_state();

    // Mode B init
    memset(pc_last_line, 0xFF, sizeof(pc_last_line)); // set to 0xFFFF
    memset(pc_stride_conf, 0, sizeof(pc_stride_conf));
    memset(pc_run_len, 0, sizeof(pc_run_len));
    memset(pc_freq, 0, sizeof(pc_freq));
    memset(rescue_hits, 0, sizeof(rescue_hits));

    // Selector init
    memset(mode_confidence, 0, sizeof(mode_confidence));
    access_count = 0;
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Check for invalid ways first
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        // Optional bypass for streaming/scanning demands (never on writebacks)
        if (mode_B_should_bypass(PC, paddr, type))
            return LLC_WAYS; // bypass
        return mode_B_get_victim(set, current_set);
    } else {
        return hawkeye_get_victim(cpu, set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Always update selector first
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates: update only the chosen mode
    if (use_mode_B) {
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }