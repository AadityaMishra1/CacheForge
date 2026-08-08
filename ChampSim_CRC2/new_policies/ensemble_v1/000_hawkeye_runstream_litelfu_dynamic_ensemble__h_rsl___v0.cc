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
// LEADER-FOLLOWER SAMPLING (separate from Hawkeye's internal sampler)
// ============================================================================
#define LLC_SET_BITS 11
static inline bool LF_SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LF_LEADER_A(uint32_t set) { return LF_SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LF_LEADER_B(uint32_t set) { return LF_SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// MODE SELECTOR (leader-follower)
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;  // Hits - misses on Mode A
static int32_t leaderB_score = 0;  // Hits - misses on Mode B
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];  // Per-set confidence

// Tunables
static const int32_t BIAS = 0;            // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096; // accesses between decisions
static const int8_t CONF_MAX = 7;         // per-set confidence bounds [-7..+7]
static const int8_t THRESHOLD = 2;        // threshold for follower to use Mode B

static inline void update_selector(uint32_t set, uint8_t hit, bool used_mode_B) {
    access_count++;

    if (LF_SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LF_LEADER_A(set)) leaderA_score += delta;
        if (LF_LEADER_B(set)) leaderB_score += delta;
    }

    // periodic aging and decision
    if ((access_count % EVAL_PERIOD) == 0) {
        bool new_prefer_B = (leaderB_score > (leaderA_score + BIAS));
        prefer_mode_B = new_prefer_B;

        // drift the current set's confidence toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (prefer_mode_B) {
                if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
            } else {
                if (mode_confidence[s] > -CONF_MAX) mode_confidence[s]--;
            }
        }

        // decay scores to avoid overflow and allow phase tracking
        leaderA_score >>= 1;
        leaderB_score >>= 1;
    }

    // Local reinforcement: whichever mode we used, gently nudge this set
    if (!LF_SAMPLED_SET(set)) {
        if (used_mode_B) {
            if (mode_confidence[set] < CONF_MAX) mode_confidence[set]++;
        } else {
            if (mode_confidence[set] > -CONF_MAX) mode_confidence[set]--;
        }
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LF_LEADER_A(set)) return false;  // Force Mode A
    if (LF_LEADER_B(set)) return true;   // Force Mode B
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
// MODE B: RunStream-LiteLFU (stream/scan detector + TinyLFU admission)
// ============================================================================

// RRPV bits (conceptual 3-bit)
static const uint8_t RRPV_B_MAX = 7;
static const uint8_t RRPV_B_DISTANT = 6;
static const uint8_t RRPV_B_PROTECT = 2;

// Per-line metadata (packed in accounting; stored as bytes here)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];
static uint8_t line_hits_B[LLC_SETS][LLC_WAYS];     // 2-bit conceptual (0..3)
static uint8_t stream_tag_B[LLC_SETS][LLC_WAYS];    // 1-bit
static uint8_t pref_q_B[LLC_SETS][LLC_WAYS];        // 1-bit

// PC-based forward-run detector (+1/+2 strides) and TinyLFU (small)
#define PC_TAB_SIZE 512
static uint16_t pc_last_line[PC_TAB_SIZE];   // 16-bit line index
static uint8_t  pc_stride_conf[PC_TAB_SIZE]; // 2-bit (0..3)
static uint8_t  pc_run_len[PC_TAB_SIZE];     // 3-bit (0..7)

#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE];        // 4-bit conceptual (0..15)

// Tunables
static const uint8_t RUN_TO_STREAM = 2;      // 2 forward steps => stream
static const uint8_t STRIDE_CONF_REQ = 2;    // confidence to declare stream
static const uint8_t HITS_TO_PROMOTE = 2;    // rescue short reuse
static const uint8_t HOT_PC_THRESH = 6;      // TinyLFU admission
static const uint32_t FREQ_AGE_PERIOD = 8192;

// Helpers
static inline uint32_t pc_idx(uint64_t PC) { return ((uint32_t)PC) & (PC_TAB_SIZE - 1); }
static inline uint32_t pc_fidx(uint64_t PC) { return ((uint32_t)PC) & (PC_FREQ_SIZE - 1); }

static inline void lfu_on_access(uint64_t PC) {
    uint32_t i = pc_fidx(PC);
    if (pc_freq[i] < 15) pc_freq[i]++;
    if ((access_count % FREQ_AGE_PERIOD) == 0) {
        for (uint32_t j = 0; j < PC_FREQ_SIZE; j++) pc_freq[j] >>= 1;
    }
}
static inline bool lfu_is_hot(uint64_t PC) {
    return pc_freq[pc_fidx(PC)] >= HOT_PC_THRESH;
}

static inline bool detect_forward_run(uint64_t PC, uint64_t line_addr) {
    uint32_t i = pc_idx(PC);
    uint16_t line = (uint16_t)(line_addr & 0xFFFFu);
    uint16_t last = pc_last_line[i];

    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(line - last);
        forward = (diff == 1) || (diff == 2);
    }

    if (forward) {
        if (pc_run_len[i] < 7) pc_run_len[i]++;
        if (pc_stride_conf[i] < 3) pc_stride_conf[i]++;
    } else {
        pc_run_len[i] = 0;
        if (pc_stride_conf[i] > 0) pc_stride_conf[i]--;
    }
    pc_last_line[i] = line;

    return (pc_run_len[i] >= RUN_TO_STREAM) && (pc_stride_conf[i] >= STRIDE_CONF_REQ);
}

// Mode B victim
static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set, uint64_t /*PC*/, uint64_t /*paddr*/, uint32_t type) {
    // Never bypass on writeback; we also avoid bypass for simplicity and predictability
    // 1) Find invalid
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }
    // 2) Evict any line at max RRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] == RRPV_B_MAX) return i;
    }
    // 3) Increment RRPVs and retry (classic RRIP walk)
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] < RRPV_B_MAX) rrpv_B[set][i]++;
    }
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_B[set][i] == RRPV_B_MAX) return i;
    }
    // Fallback
    return 0;
}

// Mode B update
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    // Ignore writebacks (no policy state change)
    if (type == WRITEBACK) return;

    uint64_t line_addr = (paddr >> 6);

    // Frequency and stream updates always on access
    lfu_on_access(PC);
    bool is_stream = detect_forward_run(PC, line_addr);
    bool hot_pc = lfu_is_hot(PC);

    if (hit) {
        // Promotion on demand hit; prefetch hits still quarantined unless repeated
        if (line_hits_B[set][way] < 3) line_hits_B[set][way]++;
        if (line_hits_B[set][way] >= HITS_TO_PROMOTE) {
            rrpv_B[set][way] = 0;           // protect
            stream_tag_B[set][way] = 0;     // clear stream tag after proving reuse
            pref_q_B[set][way] = 0;
        } else {
            // light promotion
            if (rrpv_B[set][way] > 0) rrpv_B[set][way]--;
        }
        return;
    }

    // Miss/fill path: set per-line metadata and insertion priority
    stream_tag_B[set][way] = is_stream ? 1 : 0;
    pref_q_B[set][way]     = (type == PREFETCH) ? 1 : 0;
    line_hits_B[set][way]  = 0;

    if (type == PREFETCH) {
        // quarantine prefetches
        rrpv_B[set][way] = RRPV_B_MAX;
        return;
    }

    // Demand insertion
    if (is_stream && !hot_pc) {
        // hard-tail insert to keep scans from polluting
        rrpv_B[set][way] = RRPV_B_MAX;
    } else if (hot_pc) {
        // hot PC: moderate protection
        rrpv_B[set][way] = RRPV_B_PROTECT;
    } else {
        // neutral: near-tail insert
        rrpv_B[set][way] = RRPV_B_DISTANT;
    }
}

// ============================================================================
// ENSEMBLE GLUE
// ============================================================================

void InitReplacementState() {
    // Mode A (Hawkeye) init
    hawkeye_init_replacement_state();

    // Mode B init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv_B[s][w] = RRPV_B_MAX;
            line_hits_B[s][w] = 0;
            stream_tag_B[s][w] = 0;
            pref_q_B[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TAB_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_run_len[i] = 0;
    }
    memset(pc_freq, 0, sizeof(pc_freq));

    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Check invalid ways first (policy-independent)
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        return mode_B_get_victim(set, current_set, PC, paddr, type);
    } else {
        return hawkeye_get_victim(cpu, set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                           uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                           uint32_t type, uint8_t hit) {
    bool use_mode_B = should_use_mode_B(set);

    // Update selector first (score leaders using observed hit/miss)
    update_selector(set, hit, use_mode_B);

    // EXCLUSIVE updates: only the chosen mode updates its state
    if (use_mode_B) {
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }