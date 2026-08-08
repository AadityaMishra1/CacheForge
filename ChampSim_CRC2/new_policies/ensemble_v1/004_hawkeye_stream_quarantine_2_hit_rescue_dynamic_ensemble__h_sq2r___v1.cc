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
// Mode B stores its state separately (packed, below)

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
// MODE B: STREAM-QUARANTINE + 2-HIT RESCUE (H+SQ2R) -- SEPARATE STATE
// ============================================================================

// Packed 3-bit RRIP state for Mode B (TOTAL_LINES * 3 bits)
#define TOTAL_LINES (LLC_SETS * LLC_WAYS)
static uint8_t rrpvB_bits[(TOTAL_LINES*3 + 7)/8];

// Helpers to access packed 3-bit values
static inline uint32_t line_index(uint32_t set, uint32_t way) {
    return set * LLC_WAYS + way;
}
static inline uint8_t rrpvB_get(uint32_t set, uint32_t way) {
    uint32_t idx = line_index(set, way);
    uint32_t bitpos = idx * 3;
    uint32_t byte = bitpos >> 3;
    uint32_t offset = bitpos & 7;
    uint32_t val = (rrpvB_bits[byte] >> offset) & 0x7u;
    if (offset > 5) {
        // spans into next byte
        uint32_t rem = 8 - offset;
        uint32_t hi = rrpvB_bits[byte+1] & ((1u << (3 - rem)) - 1u);
        val |= (hi << rem);
    }
    return (uint8_t)val;
}
static inline void rrpvB_set(uint32_t set, uint32_t way, uint8_t val) {
    if (val > maxRRPV) val = maxRRPV;
    uint32_t idx = line_index(set, way);
    uint32_t bitpos = idx * 3;
    uint32_t byte = bitpos >> 3;
    uint32_t offset = bitpos & 7;
    // clear
    uint8_t mask = (uint8_t)(0x7u << offset);
    rrpvB_bits[byte] = (uint8_t)((rrpvB_bits[byte] & ~mask) | ((val << offset) & mask));
    if (offset > 5) {
        // spans into next byte
        uint32_t rem = 8 - offset;
        uint8_t mask2 = (uint8_t)((1u << (3 - rem)) - 1u);
        rrpvB_bits[byte+1] = (uint8_t)((rrpvB_bits[byte+1] & ~mask2) | ((val >> rem) & mask2));
    }
}

// Tiny PC stride/run detector and promotion gate
#define PC_TABLE_SIZE 256
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t pc_run[PC_TABLE_SIZE];          // 2-bit run length (0-3)
static uint8_t pc_stride_conf[PC_TABLE_SIZE];  // 2-bit confidence (0-3)
static uint8_t pc_promote_hits[PC_TABLE_SIZE]; // 2-bit hits toward promotion (0-3)

// TinyLFU-style PC coldness filter (4-bit counters)
#define PC_FREQ_SIZE 256
static uint8_t pc_freq[PC_FREQ_SIZE];
static uint64_t pc_epoch_counter = 0;

// Tunables (retuned per feedback)
static const uint8_t STREAM_RUN_TRIGGER = 2;        // need 2 forward steps
static const uint8_t QUARANTINE_DEMAND_DEPTH = 6;   // streaming demand insert
static const uint8_t QUARANTINE_PREFETCH_DEPTH = 7; // deeper for prefetches
static const uint8_t BASE_INSERT_DEPTH = 2;         // non-stream base RRIP
static const uint8_t HOT_INSERT_DEPTH = 1;          // for hot PCs
static const uint8_t COLD_THRESH = 2;               // TinyLFU gate
static const uint8_t HITS_TO_PROMOTE = 2;           // 2-hit rescue
static const uint32_t DECAY_INTERVAL = 8192;        // periodic decay for PC freq
static const uint8_t STREAM_EXTRA_TAIL_MASK = 0x1;  // pseudo-bypass: 50% deeper

static inline uint32_t pc_idx(uint64_t PC) { return (uint32_t)(PC) & (PC_TABLE_SIZE - 1); }
static inline uint32_t pc_freq_idx(uint64_t PC) { return (uint32_t)(PC) & (PC_FREQ_SIZE - 1); }

static inline void pc_freq_touch(uint64_t PC) {
    uint32_t i = pc_freq_idx(PC);
    if (pc_freq[i] < 15) pc_freq[i]++;
    pc_epoch_counter++;
    if ((pc_epoch_counter & (DECAY_INTERVAL - 1)) == 0) {
        // periodic halve to forget cold PCs
        for (uint32_t j = 0; j < PC_FREQ_SIZE; j++) pc_freq[j] >>= 1;
    }
}
static inline bool pc_is_cold(uint64_t PC) {
    return pc_freq[pc_freq_idx(PC)] < COLD_THRESH;
}

static inline bool detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t i = pc_idx(PC);
    uint16_t line = (uint16_t)((paddr >> 6) & 0xFFFFu);
    bool is_stream = false;

    if (pc_last_line[i] != 0xFFFFu) {
        int32_t delta = (int32_t)((int32_t)line - (int32_t)pc_last_line[i]);
        if (delta == 1 || delta == 2) {
            if (pc_run[i] < 3) pc_run[i]++;
            if (pc_stride_conf[i] < 3) pc_stride_conf[i]++;
        } else {
            // reset run, mild decay of confidence
            pc_run[i] = 0;
            if (pc_stride_conf[i] > 0) pc_stride_conf[i]--;
        }
    } else {
        pc_run[i] = 0;
    }

    pc_last_line[i] = line;
    is_stream = (pc_run[i] >= STREAM_RUN_TRIGGER) && (pc_stride_conf[i] >= 1);
    return is_stream;
}

// Mode B init
static void mode_B_init_replacement_state() {
    memset(rrpvB_bits, 0xFF, sizeof(rrpvB_bits)); // set all 3-bit fields to 7 (safe)
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_run[i] = 0;
        pc_stride_conf[i] = 0;
        pc_promote_hits[i] = 0;
    }
    memset(pc_freq, 0, sizeof(pc_freq));
    pc_epoch_counter = 0;
}

// Mode B victim selection (SRRIP-style search with aging)
static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // first try: any line at maxRRPV
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (rrpvB_get(set, way) == maxRRPV) return way;
    }
    // age until someone reaches max
    for (int round = 0; round < 8; round++) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            uint8_t v = rrpvB_get(set, way);
            if (v < maxRRPV) rrpvB_set(set, way, v + 1);
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpvB_get(set, way) == maxRRPV) return way;
        }
    }
    // fallback: return the last way
    return LLC_WAYS - 1;
}

// Mode B update
static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    // Never bypass on writeback; do not disturb state
    if (type == WRITEBACK) return;

    pc_freq_touch(PC);
    bool is_stream = detect_stream(PC, paddr);
    uint32_t pi = pc_idx(PC);

    if (hit) {
        // On hits: apply gated promotion
        if (is_stream) {
            if (pc_promote_hits[pi] + 1 < HITS_TO_PROMOTE) {
                pc_promote_hits[pi]++;
                // gentle promotion: decrement by 1 if possible
                uint8_t v = rrpvB_get(set, way);
                if (v > 0) rrpvB_set(set, way, v - 1);
            } else {
                // reached threshold: full promote
                pc_promote_hits[pi] = HITS_TO_PROMOTE;
                rrpvB_set(set, way, 0);
            }
        } else {
            // non-stream: promote aggressively
            rrpvB_set(set, way, 0);
        }
        return;
    }

    // Miss/insert path: choose insertion depth
    uint8_t depth = BASE_INSERT_DEPTH;

    if (type == PREFETCH) {
        depth = QUARANTINE_PREFETCH_DEPTH;  // deepest quarantine
        pc_promote_hits[pi] = 0;
    } else if (is_stream) {
        // pseudo-bypass: ~50% go to max tail to hard-quarantine
        if ((CRC(PC) & STREAM_EXTRA_TAIL_MASK) != 0) depth = QUARANTINE_PREFETCH_DEPTH;
        else depth = QUARANTINE_DEMAND_DEPTH;
        pc_promote_hits[pi] = 0;
    } else if (pc_is_cold(PC)) {
        // cold/noisy PCs: deeper insert to reduce pollution
        depth = QUARANTINE_DEMAND_DEPTH; // biased tail
        pc_promote_hits[pi] = 0;
    } else {
        // hot PC: slightly more optimistic insert
        depth = HOT_INSERT_DEPTH;
        pc_promote_hits[pi] = 0;
    }

    rrpvB_set(set, way, depth);
}

// ============================================================================
// TOP-LEVEL REPLACEMENT INTERFACE
// ============================================================================

// Initialize replacement state
void InitReplacementState() {
    hawkeye_init_replacement_state();  // Initialize Mode A (baseline)
    mode_B_init_replacement_state();   // Initialize Mode B (separate state)

    // Initialize selector
    memset(mode_confidence, 0, sizeof(mode_confidence));
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Handle invalid ways first
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

// Update replacement state
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
    // Always update selector bookkeeping
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates: update ONLY the chosen mode
    if (use_mode_B) {
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    } else {
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank
}