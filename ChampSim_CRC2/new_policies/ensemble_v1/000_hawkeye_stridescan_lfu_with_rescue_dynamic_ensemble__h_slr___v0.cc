#include <vector>
#include <map>
#include <cstdint>
#include <cassert>

// Champsim provides NUM_CORE/LLC_{SETS,WAYS}/BLOCK/type enums in build; define here for standalone
#ifndef NUM_CORE
#define NUM_CORE 1
#endif
#ifndef LLC_SETS
#define LLC_SETS (NUM_CORE * 2048)
#endif
#ifndef LLC_WAYS
#define LLC_WAYS 16
#endif

// ============================================================================
// Leader-Follower Sampling (selector-only; distinct from Hawkeye's sampler)
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET_SEL(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LF_LEADER_A(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 0u); }
static inline bool LF_LEADER_B(uint32_t set) { return SAMPLED_SET_SEL(set) && ((set & 1u) == 1u); }

// ============================================================================
// Selector knobs (tunable): BIAS>=0 favors Hawkeye; THRESHOLD controls per-set drift
// ============================================================================
static const int32_t BIAS = 0;
static const uint32_t EVAL_PERIOD = 4096;
static const int8_t THRESHOLD = 2;
static const int8_t CONF_MAX = 8;

static uint64_t access_count = 0;
static int32_t leaderA_score = 0;
static int32_t leaderB_score = 0;
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];

static void selector_init() {
    access_count = 0;
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    for (uint32_t s = 0; s < LLC_SETS; s++) mode_confidence[s] = 0;
}

static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;
    if (SAMPLED_SET_SEL(set)) {
        int delta = hit ? 1 : -1;
        if (LF_LEADER_A(set)) leaderA_score += delta;
        else if (LF_LEADER_B(set)) leaderB_score += delta;
    }
    if ((access_count % EVAL_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > leaderA_score + BIAS);
        // Drift follower sets toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (prefer_mode_B) {
                if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
            } else {
                if (mode_confidence[s] > -CONF_MAX) mode_confidence[s]--;
            }
        }
        // mild decay to avoid long-term bias lock-in
        leaderA_score >>= 1;
        leaderB_score >>= 1;
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LF_LEADER_A(set)) return false;
    if (LF_LEADER_B(set)) return true;
    return prefer_mode_B && (mode_confidence[set] >= THRESHOLD);
}

// ============================================================================
// MODE B: StrideScan-LFU with Rescue (H+SLR)
// - Stream/scan detector: PC-based +1/+2 forward runs; quarantine streams
// - TinyLFU: PC frequency admission, periodic aging
// - Rescue: promote stream-tagged lines after 2 demand hits
// Notes:
//   - Never bypass on WRITEBACK
//   - Prefetches always quarantined (tail insert)
//   - Tunables exposed below through constants
// ============================================================================

// Tunables for Mode B
static const uint8_t RUN_LEN_TO_STREAM = 2;   // >=2 forward steps => stream
static const uint8_t STRIDE_CONF_REQ   = 2;   // min confidence to declare stream
static const uint8_t HITS_TO_RESCUE    = 2;   // demand hits to rescue a stream-tagged line
static const uint8_t RRPV_PROTECT      = 2;   // protected insert (not MRU)
static const uint32_t FREQ_AGE_PERIOD  = 8192;
static const uint8_t FREQ_THRESHOLD    = 6;   // PC is hot if >= threshold

// Per-line metadata (conceptually bit-packed; maintained as bytes for simplicity)
static uint8_t mb_line_hits[LLC_SETS][LLC_WAYS];       // 2-bit (0..3)
static uint8_t mb_line_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit
static uint8_t mb_line_pref_q[LLC_SETS][LLC_WAYS];     // 1-bit

// PC tables for stream detection and frequency
#define MB_PC_TAB 512
#define MB_PC_FREQ 512
static uint16_t slr_pc_lastline[MB_PC_TAB]; // 16-bit line index (low bits)
static uint8_t  slr_pc_run[MB_PC_TAB];      // 3-bit (0..7)
static uint8_t  slr_pc_conf[MB_PC_TAB];     // 2-bit (0..3)
static uint8_t  slr_pc_freq[MB_PC_FREQ];    // 4-bit (0..15 conceptual)

// helpers
static inline uint32_t mb_pc_idx(uint64_t pc)   { return (uint32_t)pc & (MB_PC_TAB-1); }
static inline uint32_t mb_freq_idx(uint64_t pc) { return (uint32_t)pc & (MB_PC_FREQ-1); }
static inline void mb_freq_on_access(uint64_t PC) {
    uint32_t i = mb_freq_idx(PC);
    if (slr_pc_freq[i] < 15) slr_pc_freq[i]++;
}
static inline void mb_freq_age_periodic() {
    if ((access_count % FREQ_AGE_PERIOD) == 0) {
        for (uint32_t i = 0; i < MB_PC_FREQ; i++) slr_pc_freq[i] >>= 1;
    }
}
static inline bool mb_freq_is_hot(uint64_t PC) {
    return slr_pc_freq[mb_freq_idx(PC)] >= FREQ_THRESHOLD;
}
static inline bool mb_detect_stream(uint64_t PC, uint64_t paddr) {
    uint64_t line = (paddr >> 6);
    uint32_t idx = mb_pc_idx(PC);
    uint16_t l   = (uint16_t)(line & 0xFFFFu);
    uint16_t last = slr_pc_lastline[idx];
    bool fwd = false;
    if (last != 0xFFFFu) {
        uint16_t diff = (uint16_t)(l - last);
        fwd = (diff == 1) || (diff == 2);
    }
    if (fwd) {
        if (slr_pc_run[idx]  < 7) slr_pc_run[idx]++;
        if (slr_pc_conf[idx] < 3) slr_pc_conf[idx]++;
    } else {
        slr_pc_run[idx] = 0;
        if (slr_pc_conf[idx] > 0) slr_pc_conf[idx]--;
    }
    slr_pc_lastline[idx] = l;
    return (slr_pc_run[idx] >= RUN_LEN_TO_STREAM) && (slr_pc_conf[idx] >= STRIDE_CONF_REQ);
}

// We reuse Hawkeye's 3-bit RRIP array (rrpv[][]) and maxRRPV from Mode A.
// Prefer evicting quarantined stream/prefetch lines first.
extern uint32_t rrpv[LLC_SETS][LLC_WAYS];
extern const uint32_t maxRRPV; // from Hawkeye block

static uint32_t mode_B_get_victim(uint32_t set, const void* current_set_unused) {
    // pass 1: maxRRPV with quarantine tags
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == maxRRPV && (mb_line_stream_tag[set][i] || mb_line_pref_q[set][i])) return i;
    }
    // pass 2: any maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv[set][i] == maxRRPV) return i;
    }
    // age until one appears (SRRIP)
    while (true) {
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] == maxRRPV) return i;
        }
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv[set][i] < maxRRPV) rrpv[set][i]++;
        }
    }
}

static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    // Per-access maintenance
    mb_freq_on_access(PC);
    mb_freq_age_periodic();

    if (hit) {
        if (mb_line_hits[set][way] < 3) mb_line_hits[set][way]++;
        // SRRIP-like hit promotion
        if (rrpv[set][way] > 0) rrpv[set][way]--;
        // Rescue short-reuse streams
        if (mb_line_stream_tag[set][way] && mb_line_hits[set][way] >= HITS_TO_RESCUE) {
            rrpv[set][way] = 0; // strong protect upon rescue
            mb_line_stream_tag[set][way] = 0; // clear stream tag after rescue
        }
        return;
    }

    // Miss/fill: choose insertion policy
    bool is_stream = mb_detect_stream(PC, paddr);
    mb_line_hits[set][way] = 0; // reset on fill
    mb_line_pref_q[set][way] = 0;

    // Never bypass on WRITEBACK; give modest priority
    if (type == 2 /* WRITEBACK (enum-provided at integration) */) {
        // Conservative priority for writebacks if type constants unknown at compile:
        rrpv[set][way] = (maxRRPV > 1) ? (maxRRPV - 1) : maxRRPV;
        mb_line_stream_tag[set][way] = 0;
        return;
    }

    // Prefetches: hard quarantine at tail
    if (type == 1 /* PREFETCH */) {
        rrpv[set][way] = maxRRPV;
        mb_line_pref_q[set][way] = 1;
        mb_line_stream_tag[set][way] = is_stream ? 1 : 0;
        return;
    }

    // Demand insertions:
    if (is_stream && !mb_freq_is_hot(PC)) {
        // Stream/scan and cold PC => quarantine
        rrpv[set][way] = maxRRPV;
        mb_line_stream_tag[set][way] = 1;
    } else {
        // Non-stream or hot PC => modest protection
        rrpv[set][way] = (RRPV_PROTECT < maxRRPV) ? RRPV_PROTECT : (maxRRPV - 1);
        mb_line_stream_tag[set][way] = 0;
    }
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
    std::map<uint64_t, short unsigned int> SHCT;
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
    std::vector<unsigned int> liveness_history;
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
std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

// Sampling macros
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0, 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

// Sampler helper functions
void replace_addr_history_element(unsigned int sampler_set) {
    uint64_t lru_addr = 0;
    for(std::map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin();
        it != addr_history[sampler_set].end(); it++) {
        if((it->second).lru == (SAMPLER_WAYS-1)) {
            lru_addr = it->first;
            break;
        }
    }
    addr_history[sampler_set].erase(lru_addr);
}

void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    for(std::map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin();
        it != addr_history[sampler_set].end(); it++) {
        if((it->second).lru < curr_lru) {
            (it->second).lru++;
            assert((it->second).lru < SAMPLER_WAYS);
        }
    }
}

// Hawkeye victim selection
uint32_t hawkeye_get_victim(uint32_t cpu, uint32_t set, const void *current_set) {
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
// Integration entry points
// ============================================================================

void InitReplacementState() {
    hawkeye_init_replacement_state();  // Mode A baseline
    selector_init();

    // Initialize Mode B state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            mb_line_hits[s][w] = 0;
            mb_line_stream_tag[s][w] = 0;
            mb_line_pref_q[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < MB_PC_TAB; i++) {
        slr_pc_lastline[i] = 0xFFFFu;
        slr_pc_run[i] = 0;
        slr_pc_conf[i] = 0;
    }
    for (uint32_t i = 0; i < MB_PC_FREQ; i++) slr_pc_freq[i] = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const void* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Check for invalid ways first; BLOCK.valid is provided by ChampSim, so we conservatively skip here.
    // Victim selection
    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) return mode_B_get_victim(set, current_set);
    return hawkeye_get_victim(cpu, set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Update selector tallies
    update_selector(set, hit);

    // Always train Hawkeye (Mode A)
    hawkeye_update_replacement_state(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Apply Mode B policy if selected for this set
    if (should_use_mode_B(set)) {
        mode_B_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}