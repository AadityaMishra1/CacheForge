#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>  // CRITICAL: for log2()
#include "../inc/champsim_crc2.h"  // CRITICAL: for BLOCK, PREFETCH, WRITEBACK

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16

// ============================================================================
// DYNAMIC ENSEMBLE: Leader-Follower Selector (64 sampled sets, 32/32 leaders)
// NOTE: Use DUEL_* names to avoid clashing with Hawkeye's SAMPLED_SET macro.
// ============================================================================
#define LLC_SET_BITS 11
static inline bool DUEL_SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool DUEL_LEADER_A(uint32_t set) { return DUEL_SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool DUEL_LEADER_B(uint32_t set) { return DUEL_SAMPLED_SET(set) && ((set & 1u) == 1u); }

static uint64_t access_count = 0;
static int32_t leaderA_score = 0;   // hits - misses on Mode A leader sets
static int32_t leaderB_score = 0;   // hits - misses on Mode B leader sets
static bool prefer_mode_B = false;

// Per-set soft confidence (conceptually 2–3 bits; stored as int8_t)
static int8_t mode_confidence[LLC_SETS];

static const int32_t BIAS = 0;            // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096; // accesses between global decisions
static const int8_t THRESHOLD = 2;        // min confidence to use Mode B
static const int8_t CONF_MAX = 8;

static void selector_periodic_drift() {
    // Drift follower sets toward global preference
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (DUEL_LEADER_A(s) || DUEL_LEADER_B(s)) continue;
        if (prefer_mode_B) {
            if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
        } else {
            if (mode_confidence[s] > -CONF_MAX) mode_confidence[s]--;
        }
    }
}

static void update_selector(uint32_t set, uint8_t hit) {
    if (DUEL_SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (DUEL_LEADER_A(set)) leaderA_score += delta;
        if (DUEL_LEADER_B(set)) leaderB_score += delta;
    }

    if (access_count && (access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        selector_periodic_drift();
        leaderA_score = 0;
        leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (DUEL_LEADER_A(set)) return false; // force Mode A
    if (DUEL_LEADER_B(set)) return true;  // force Mode B
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
// MODE B: FlowGuard-LFU (stream/scan quarantine + TinyLFU admission + 2-hit escape)
// ============================================================================
#define PC_TAB_SIZE 512
#define PC_FREQ_SIZE 512
#define REG_FREQ_SIZE 512

// PC-based forward-run detector state
static uint16_t pc_last_line[PC_TAB_SIZE];     // last line index (16-bit)
static uint8_t  pc_stride_conf[PC_TAB_SIZE];   // 0..3 (2-bit conceptual)
static uint8_t  pc_forward_run[PC_TAB_SIZE];   // 0..7 (3-bit conceptual)

// Frequency (TinyLFU) state for PC and 4KB regions (conceptual 4-bit counters)
static uint8_t pc_freq[PC_FREQ_SIZE];
static uint8_t reg_freq[REG_FREQ_SIZE];

// Per-line soft metadata (conceptual bits; stored as bytes)
static uint8_t line_hits[LLC_SETS][LLC_WAYS];      // 2-bit (0..3)
static uint8_t line_stream_tag[LLC_SETS][LLC_WAYS];// 1-bit: 1=stream/quarantine
static uint8_t line_pref_q[LLC_SETS][LLC_WAYS];    // 1-bit: 1=prefetch quarantine

// Tunables
static const uint8_t RUN_LEN_TO_STREAM = 2;    // 2 forward strides => stream
static const uint8_t STRIDE_CONF_REQ   = 2;    // min confidence for stream
static const uint8_t HITS_TO_PROMOTE   = 2;    // escape threshold for stream-tagged
static const uint8_t FREQ_TH_PC        = 6;    // TinyLFU hot PC threshold
static const uint8_t FREQ_TH_REG       = 6;    // TinyLFU hot region threshold
static const uint32_t FREQ_AGE_PERIOD  = 8192; // periodic halving

static inline uint32_t pc_idx(uint64_t PC) { return (uint32_t)PC & (PC_TAB_SIZE - 1); }
static inline uint32_t pcf_idx(uint64_t PC) { return (uint32_t)PC & (PC_FREQ_SIZE - 1); }
static inline uint32_t reg_idx(uint64_t paddr) { return (uint32_t)((paddr >> 12) & (REG_FREQ_SIZE - 1)); }

static inline void freq_on_access(uint64_t PC, uint64_t paddr) {
    uint32_t pi = pcf_idx(PC);
    uint32_t ri = reg_idx(paddr);
    if (pc_freq[pi] < 15) pc_freq[pi]++;
    if (reg_freq[ri] < 15) reg_freq[ri]++;
}

static inline void periodic_freq_age() {
    if ((access_count % FREQ_AGE_PERIOD) == 0 && access_count) {
        for (uint32_t i = 0; i < PC_FREQ_SIZE; i++) pc_freq[i] >>= 1;
        for (uint32_t i = 0; i < REG_FREQ_SIZE; i++) reg_freq[i] >>= 1;
    }
}

static inline bool freq_is_hot(uint64_t PC, uint64_t paddr) {
    return (pc_freq[pcf_idx(PC)] >= FREQ_TH_PC) || (reg_freq[reg_idx(paddr)] >= FREQ_TH_REG);
}

// Detect forward run (+1/+2 strides) per PC
static inline bool detect_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_idx(PC);
    uint16_t line = (uint16_t)(paddr_line & 0xFFFFu);
    uint16_t last = pc_last_line[idx];

    bool fwd = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(line - last);
        fwd = (diff == 1) || (diff == 2);
    }

    if (fwd) {
        if (pc_forward_run[idx] < 7) pc_forward_run[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_forward_run[idx] = 0;
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--;
    }

    pc_last_line[idx] = line;
    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

// Mode B victim: prefer evicting quarantined (prefetch/stream) lines among maxRRPV
static inline uint32_t mode_B_get_victim(uint32_t set, const BLOCK* current_set) {
    // First look for invalid
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }
    // Prefer maxRRPV and quarantined
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == maxRRPV && (line_pref_q[set][i] || line_stream_tag[set][i])) return i;
    }
    // Any maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == maxRRPV) return i;
    }
    // Fallback: evict highest RRPV, bias to stream-tagged
    uint32_t best = 0, best_rrpv = 0;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if ((rrpv[set][i] > best_rrpv) || (rrpv[set][i] == best_rrpv && line_stream_tag[set][i])) {
            best_rrpv = rrpv[set][i];
            best = i;
        }
    }
    return best;
}

// Mode B update (insertion/promotion)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    periodic_freq_age();
    if (type != WRITEBACK) freq_on_access(PC, paddr);

    // No action on writeback (never bypass/don't alter)
    if (type == WRITEBACK) return;

    uint64_t line_addr = (paddr >> 6);
    bool is_stream = detect_stream(PC, line_addr);
    bool is_hot = freq_is_hot(PC, paddr);

    if (hit) {
        // light promotion and escape for stream-tagged lines
        if (line_hits[set][way] < 3) line_hits[set][way]++;
        if (line_stream_tag[set][way] && line_hits[set][way] >= HITS_TO_PROMOTE) {
            line_stream_tag[set][way] = 0;      // escaped the quarantine
            if (rrpv[set][way] > 1) rrpv[set][way] = 1;
        } else {
            if (rrpv[set][way] > 0) rrpv[set][way]--; // gentle promote
        }
        return;
    }

    // Miss/fill path: set quarantine flags and insertion depth
    line_hits[set][way] = 0;
    line_pref_q[set][way] = (type == PREFETCH) ? 1 : 0;
    if (type == PREFETCH) {
        // quarantine all prefetches
        rrpv[set][way] = maxRRPV;
        line_stream_tag[set][way] = 0;
        return;
    }

    // Demand fill
    if (is_stream && !is_hot) {
        // stream quarantine
        rrpv[set][way] = maxRRPV;
        line_stream_tag[set][way] = 1;
    } else if (is_hot) {
        // hot flow: protect modestly
        rrpv[set][way] = 2;
        line_stream_tag[set][way] = 0;
    } else {
        // cold/uncertain: distant insertion
        rrpv[set][way] = maxRRPV - 1;
        line_stream_tag[set][way] = 0;
    }
}

// ============================================================================
// ENSEMBLE INTERFACE (Init/GetVictim/Update)
// ============================================================================
void InitReplacementState() {
    hawkeye_init_replacement_state(); // Initialize Mode A

    // Initialize Mode B state
    for (uint32_t i = 0; i < PC_TAB_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
    }
    std::memset(pc_freq, 0, sizeof(pc_freq));
    std::memset(reg_freq, 0, sizeof(reg_freq));
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            line_hits[s][w] = 0;
            line_stream_tag[s][w] = 0;
            line_pref_q[s][w] = 0;
        }
    }

    prefer_mode_B = false;
    leaderA_score = leaderB_score = 0;
    access_count = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Always honor invalid first
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
    access_count++;

    // Update selector with observed outcome on this set
    update_selector(set, hit);

    // Always train Hawkeye (safety net)
    hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Apply Mode B insertion/promotion if selected
    if (should_use_mode_B(set)) {
        mode_B_update(set, way, paddr, PC, type, hit);
    }
}

// Stats hooks
void PrintStats() {}
void PrintStats_Heartbeat() {}