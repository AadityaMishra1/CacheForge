#include <vector>
#include <map>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>
using namespace std;

#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ============================================================================
// Leader-Follower Sampling (64 sampled sets) for Mode selection
// NOTE: Use distinct names to avoid collision with Hawkeye's internal macro.
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET_SEL(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 1u); }

// ============================================================================
// Global selector state and tunables
// ============================================================================
static uint64_t access_count = 0;
static int32_t leaderA_score = 0;   // Hits - misses on Mode A leader sets
static int32_t leaderB_score = 0;   // Hits - misses on Mode B leader sets
static bool prefer_mode_B = false;

static const int32_t BIAS = 0;             // >=0 favors Mode A
static const uint32_t EVAL_PERIOD = 4096;  // accesses between global decisions
static const int8_t THRESHOLD = 2;         // per-set confidence threshold
static const int8_t CONF_MAX = 8;
static int8_t mode_confidence[LLC_SETS];   // per-set drift state

static void drift_confidence() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        int8_t c = mode_confidence[s];
        if (prefer_mode_B) {
            if (c < CONF_MAX) c++;
        } else {
            if (c > -CONF_MAX) c--;
        }
        mode_confidence[s] = c;
    }
}

static void update_selector(uint32_t set, uint8_t hit) {
    int delta = hit ? 1 : -1;
    if (SAMPLED_SET_SEL(set)) {
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }
    access_count++;
    if (access_count % EVAL_PERIOD == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));
        drift_confidence();
        // mild decay to avoid runaway
        leaderA_score /= 2;
        leaderB_score /= 2;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;  // Force Mode A
    if (LEADER_B(set)) return true;   // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// MODE A: FULL HAWKEYE IMPLEMENTATION (BASELINE - DO NOT MODIFY)
// ============================================================================
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
// END Hawkeye Mode A
// ============================================================================


// ============================================================================
// MODE B: StreamScan + TinyLFU + Ghost admission (H+SSG)
// - PC-based forward-run detector (+1/+2 strides) with run-length confidence
// - Prefetch quarantine and stream tail-insertion
// - TinyLFU per-PC frequency and 64-entry ghost-evict table for noisy PCs
// - Two-hit PC-based promotion to rescue short reuse
// - Light stream-pressure based extra aging in victim selection
// ============================================================================

// Tunables
static const uint8_t RUN_LEN_TO_STREAM = 2;  // two forward steps => stream/scan
static const uint8_t STRIDE_CONF_REQ   = 2;  // minimal confidence
static const uint8_t HITS_TO_PROMOTE   = 2;  // demand hits to push to MRU
static const uint8_t PROTECT_RRIP      = 2;  // protected insertion
static const uint8_t DISTANT_RRIP      = (maxRRPV - 1); // near-tail insertion (6)
static const uint8_t FREQ_THRESHOLD    = 6;  // TinyLFU threshold (4-bit conceptual)
static const uint32_t FREQ_AGE_PERIOD  = 8192;
static const uint32_t GHOST_AGE_PERIOD = 8192;
static const uint8_t GHOST_TH          = 3;  // noisy if >=3 recent evictions
static const uint8_t STREAM_PRESS_MAX  = 3;

// Per-set stream pressure (helps aggressive aging under scans)
static uint8_t stream_pressure[LLC_SETS];

// PC stride/stream detector tables
#define PC_TABLE_SIZE 512
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stride_conf[PC_TABLE_SIZE]; // 0..3
static uint8_t  pc_forward_run[PC_TABLE_SIZE]; // 0..7
static uint8_t  pc_recent_hits[PC_TABLE_SIZE]; // 2-bit conceptual

// TinyLFU frequency table
#define PC_FREQ_SIZE 512
static uint8_t pc_freq[PC_FREQ_SIZE]; // 4-bit conceptual

// Ghost-evict table (direct-mapped by hashed PC)
#define GHOST_SIZE 64
static uint16_t ghost_sig[GHOST_SIZE];
static uint8_t  ghost_cnt[GHOST_SIZE];

static inline uint32_t pc_tab_index(uint64_t PC) { return (uint32_t)PC & (PC_TABLE_SIZE - 1); }
static inline uint32_t pc_freq_index(uint64_t PC) { return (uint32_t)PC & (PC_FREQ_SIZE - 1); }
static inline uint32_t ghost_index(uint64_t PC) { return (uint32_t)(CRC(PC) & (GHOST_SIZE - 1)); }
static inline uint16_t ghost_sig_hash(uint64_t PC) { return (uint16_t)(CRC(PC) & 0xFFFFu); }

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

static inline void ghost_on_evict(uint64_t PC) {
    uint32_t gi = ghost_index(PC);
    uint16_t sig = ghost_sig_hash(PC);
    if (ghost_sig[gi] == sig) {
        if (ghost_cnt[gi] < 7) ghost_cnt[gi]++;
    } else {
        ghost_sig[gi] = sig;
        ghost_cnt[gi] = 1;
    }
}

static inline bool ghost_is_noisy(uint64_t PC) {
    uint32_t gi = ghost_index(PC);
    return (ghost_sig[gi] == ghost_sig_hash(PC)) && (ghost_cnt[gi] >= GHOST_TH);
}

static inline void periodic_ghost_age() {
    if ((access_count % GHOST_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < GHOST_SIZE; i++) ghost_cnt[i] >>= 1;
    }
}

// Stream detection: detect forward +1/+2 runs with confidence; update PC state
static inline bool detect_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = pc_tab_index(PC);
    uint16_t line = (uint16_t)(paddr_line & 0xFFFFu);
    uint16_t last = pc_last_line[idx];

    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(line - last);
        forward = (diff == 1) || (diff == 2);
    }

    if (forward) {
        if (pc_forward_run[idx] < 7) pc_forward_run[idx]++;
        if (pc_stride_conf[idx] < 3) pc_stride_conf[idx]++;
    } else {
        pc_forward_run[idx] = 0;
        if (pc_stride_conf[idx] > 0) pc_stride_conf[idx]--; // mild decay
    }

    pc_last_line[idx] = line;

    return (pc_forward_run[idx] >= RUN_LEN_TO_STREAM) && (pc_stride_conf[idx] >= STRIDE_CONF_REQ);
}

// Mode B victim selection (SRRIP with optional extra aging under stream pressure)
static uint32_t mode_B_get_victim(uint32_t set, const BLOCK *current_set) {
    while (true) {
        // Prefer maxRRPV
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] == maxRRPV)
                return i;
        }
        // Age all lines if none distant enough; add slight bias under pressure
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < maxRRPV) rrpv[set][i]++;
        }
        if (stream_pressure[set]) {
            // one extra aging step when scans detected to flush tail sooner
            for (uint32_t i = 0; i < LLC_WAYS; i++) {
                if (rrpv[set][i] < maxRRPV) rrpv[set][i]++;
            }
            if (stream_pressure[set] > 0) stream_pressure[set]--;
        }
    }
}

// Mode B insertion/update (applied after Hawkeye training to override insertion)
static void mode_B_update_replacement_state(uint32_t set, uint32_t way,
                                            uint64_t paddr, uint64_t PC,
                                            uint32_t type, uint8_t hit) {
    uint64_t line = (paddr >> 6);
    uint32_t pcidx = pc_tab_index(PC);

    // Update PC frequency and age periodic structures
    freq_on_access(PC);
    periodic_freq_age();
    periodic_ghost_age();

    if (hit) {
        // Two-hit PC-level promotion gate
        if (pc_recent_hits[pcidx] < 3) pc_recent_hits[pcidx]++;
        if (pc_recent_hits[pcidx] + 1 >= HITS_TO_PROMOTE) {
            rrpv[set][way] = 0; // promote to MRU
        } else {
            if (rrpv[set][way] > 0) rrpv[set][way]--; // gentle promotion
        }
        return;
    }

    // Miss/fill path: detect stream, decide insertion
    bool is_stream = detect_stream(PC, line);
    bool is_hot = freq_is_hot(PC);
    bool noisy = ghost_is_noisy(PC);

    // WRITEBACK: never bypass, modest priority near tail
    if (type == WRITEBACK) {
        rrpv[set][way] = DISTANT_RRIP;
        return;
    }

    // PREFETCH: quarantine to tail
    if (type == PREFETCH) {
        rrpv[set][way] = maxRRPV;
        return;
    }

    if (is_stream) {
        // Hard tail for streams/scans; also raise stream pressure for this set
        rrpv[set][way] = maxRRPV;
        if (stream_pressure[set] < STREAM_PRESS_MAX) stream_pressure[set]++;
        // Reset recent hits for streaming PC
        pc_recent_hits[pcidx] = 0;
        return;
    }

    // Demand from non-stream PC: gate by TinyLFU and ghost-noise
    if (is_hot && !noisy) {
        rrpv[set][way] = PROTECT_RRIP; // protected insertion for hot PCs
    } else {
        rrpv[set][way] = DISTANT_RRIP; // cold/noisy PCs near tail
    }

    // Mild decay of per-PC recent hit gate on fills
    if (pc_recent_hits[pcidx] > 0) pc_recent_hits[pcidx]--;
}

// ============================================================================
// Exported interface required by ChampSim
// ============================================================================

// Initialize replacement state
void InitReplacementState() {
    // Initialize Hawkeye (Mode A)
    hawkeye_init_replacement_state();

    // Initialize selector
    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    memset(mode_confidence, 0, sizeof(mode_confidence));

    // Initialize Mode B state
    memset(stream_pressure, 0, sizeof(stream_pressure));
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) {
        pc_last_line[i] = 0xFFFFu;
        pc_stride_conf[i] = 0;
        pc_forward_run[i] = 0;
        pc_recent_hits[i] = 0;
    }
    memset(pc_freq, 0, sizeof(pc_freq));
    for (uint32_t i = 0; i < GHOST_SIZE; i++) {
        ghost_sig[i] = 0xFFFFu;
        ghost_cnt[i] = 0;
    }
}

// Find victim in the set
uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set,
                         uint64_t PC, uint64_t paddr, uint32_t type) {
    // Always prefer invalid way if present
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (current_set[i].valid == 0) return i;
    }

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        // Mode B victim selection
        uint32_t vic = mode_B_get_victim(set, current_set);

        // Train Hawkeye predictor negatively on eviction for sampled sets (like Hawkeye)
        if (SAMPLED_SET(set)) {
            if (prefetched[set][vic])
                prefetch_predictor->decrement(signatures[set][vic]);
            else
                demand_predictor->decrement(signatures[set][vic]);
        }

        // Update ghost with victim's PC to mark noisy PCs
        ghost_on_evict(signatures[set][vic]);

        return vic;
    } else {
        // Mode A victim selection
        return hawkeye_get_victim(cpu, set, current_set);
    }
}

// Update replacement state
void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way,
                             uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                             uint32_t type, uint8_t hit) {
    // Update selector accounting first
    update_selector(set, hit);

    // Always train Hawkeye (Mode A) for safety net and comparison
    hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Apply Mode B adjustments if selected
    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        mode_B_update_replacement_state(set, way, paddr, PC, type, hit);
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally left blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally left blank
}