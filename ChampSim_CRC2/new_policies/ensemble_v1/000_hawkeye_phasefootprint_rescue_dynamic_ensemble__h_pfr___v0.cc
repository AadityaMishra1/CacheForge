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
// LEADER-FOLLOWER SAMPLING (selector-only; avoid name clash with Hawkeye)
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

// Selector tunables
static const int32_t BIAS = 2;            // Non-negative, slightly favors Hawkeye
static const uint32_t EVAL_PERIOD = 4096; // accesses between global decisions
static const int8_t THRESHOLD = 2;        // per-set confidence threshold
static const int8_t CONF_MAX = 8;

// Periodic evaluation and per-set drift toward global preference
static inline void update_selector(uint32_t set, uint8_t hit) {
    access_count++;

    // Score leaders
    if (SAMPLED_SET_SEL(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A_SEL(set)) leaderA_score += delta;
        if (LEADER_B_SEL(set)) leaderB_score += delta;
    }

    // Every EVAL_PERIOD accesses: decide global preference, reset scores
    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Drift this set's confidence slightly toward current global preference
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
// MODE B: PhaseFootprint-Rescue (PC temporal footprint + short-reuse rescue)
// ============================================================================
// Tiny per-PC state
#define PC_TABLE_SIZE 1024
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint16_t pc_last2_line[PC_TABLE_SIZE];
static uint8_t  pc_unique[PC_TABLE_SIZE];   // decayed "unique-line" proxy
static uint8_t  pc_dhits[PC_TABLE_SIZE];    // decayed demand-hit counter

// Tunables for Mode B
static const uint32_t PC_AGE_PERIOD = 8192; // aging period
static const uint8_t  FP_TH = 8;            // footprint threshold to call it scan/streamy
static const uint8_t  DHIT_TH = 2;          // low reuse threshold
static const uint8_t  RESCUE_HITS = 2;      // promote after >=2 recent demand hits
static const uint8_t  INSERT_PROTECT = 2;   // friendly insertion depth

static inline uint32_t pc_index(uint64_t PC) { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }

static inline void pc_periodic_age() {
    if ((access_count % PC_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
            pc_unique[i] >>= 1;
            pc_dhits[i] >>= 1;
        }
    }
}

// Update per-PC footprint estimate and hit counters
static inline void modeB_pc_update(uint64_t PC, uint64_t line_addr, uint8_t hit, uint32_t type) {
    uint32_t idx = pc_index(PC);
    uint16_t line16 = (uint16_t)(line_addr & 0xFFFFu);

    // "Unique" proxy: count when current line differs from last two low-16 ids
    if (line16 != pc_last_line[idx] && line16 != pc_last2_line[idx]) {
        if (pc_unique[idx] < 255) pc_unique[idx]++;
        pc_last2_line[idx] = pc_last_line[idx];
        pc_last_line[idx]  = line16;
    } else {
        // Update recency order
        if (line16 != pc_last_line[idx]) {
            pc_last2_line[idx] = pc_last_line[idx];
            pc_last_line[idx]  = line16;
        }
    }

    // Count demand hits only
    if (hit && type != PREFETCH && type != WRITEBACK) {
        if (pc_dhits[idx] < 255) pc_dhits[idx]++;
    }

    pc_periodic_age();
}

static inline bool modeB_pc_is_scanlike(uint64_t PC) {
    uint32_t idx = pc_index(PC);
    return (pc_unique[idx] >= FP_TH) && (pc_dhits[idx] <= DHIT_TH);
}

// Mode B victim selection: use RRIP counters (shared) with simple maxRRPV preference
static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    // Try to find a maxRRPV line
    for (uint32_t i=0; i<LLC_WAYS; i++) {
        if (rrpv[set][i] == maxRRPV) return i;
    }
    // Else pick the line with largest RRPV (ties -> highest index)
    uint32_t max_rr = 0;
    int32_t victim = -1;
    for (uint32_t i=0; i<LLC_WAYS; i++) {
        if (rrpv[set][i] >= max_rr) {
            max_rr = rrpv[set][i];
            victim = i;
        }
    }
    if (victim < 0) victim = 0;
    return (uint32_t)victim;
}

// Mode B update: insertion/promotion policy only (no Hawkeye predictor updates)
static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint64_t victim_addr, uint32_t type, uint8_t hit) {
    // Ignore writebacks entirely
    if (type == WRITEBACK) return;

    uint64_t line_addr = (paddr >> 6);
    // Always update PC-level telemetry
    modeB_pc_update(PC, line_addr, hit, type);

    // Promotion on hits
    if (hit) {
        // If scan-like PC but has accumulated enough recent demand hits, rescue it
        if (!modeB_pc_is_scanlike(PC) || (pc_dhits[pc_index(PC)] >= RESCUE_HITS)) {
            rrpv[set][way] = 0; // strong protect
        } else {
            // Mild promotion for scan-like PCs without rescue: keep them vulnerable
            if (rrpv[set][way] > 1) rrpv[set][way]--;
        }
        return;
    }

    // On fills/insertions (type is demand or prefetch)
    // Prefetch quarantine: aggressive tail insert
    if (type == PREFETCH) {
        rrpv[set][way] = maxRRPV; // quarantine prefetches
        return;
    }

    // Demand miss insert
    if (modeB_pc_is_scanlike(PC) && (pc_dhits[pc_index(PC)] < RESCUE_HITS)) {
        // Scan/streamy PC with low reuse ⇒ tail insert (pollution control)
        rrpv[set][way] = maxRRPV;
    } else {
        // Friendly/likely-reuse ⇒ protected insertion
        rrpv[set][way] = (INSERT_PROTECT <= maxRRPV) ? INSERT_PROTECT : (maxRRPV - 1);
    }
}

// ============================================================================
// TOP-LEVEL INTERFACE
// ============================================================================
void InitReplacementState() {
    // Initialize Hawkeye (Mode A)
    hawkeye_init_replacement_state();

    // Initialize selector and Mode B state
    memset(mode_confidence, 0, sizeof(mode_confidence));
    access_count = 0;
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;

    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i]  = 0xFFFFu;
        pc_last2_line[i] = 0xFFFFu;
        pc_unique[i]     = 0;
        pc_dhits[i]      = 0;
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Check for invalid way first
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
    // Update selector accounting first
    update_selector(set, hit);

    // Decide which mode to apply (exclusive)
    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        // Mode B ONLY: do not touch Hawkeye state here
        mode_B_update_replacement_state(set, way, paddr, PC, victim_addr, type, hit);
    } else {
        // Mode A ONLY: full Hawkeye update
        hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);
    }
}

void PrintStats() { /* intentionally blank */ }
void PrintStats_Heartbeat() { /* intentionally blank */ }