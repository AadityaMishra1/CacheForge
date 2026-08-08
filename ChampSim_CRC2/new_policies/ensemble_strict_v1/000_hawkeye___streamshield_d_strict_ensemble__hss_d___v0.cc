/*
 * STRICT DYNAMIC ENSEMBLE TEMPLATE - PHASE 1
 *
 * PROTECTION GUARANTEE: Mode A (Hawkeye) is the proven baseline.
 * Mode B can only help on weak workloads or fall back to Mode A everywhere.
 *
 * ARCHITECTURE:
 * - Mode A: Full Hawkeye with EXCLUSIVE state (rrpv_A + Hawkeye metadata)
 * - Mode B: HST-E derived (stride/TinyLFU/stream quarantine) with EXCLUSIVE state (rrpv_B)
 * - Strict Gating: High bias (30-35), high threshold (~5), follower cap
 * - No Shared State: Mode A and Mode B never touch each other's RRIP arrays
 *
 * DEFAULT BEHAVIOR: All followers use Mode A unless Mode B wins decisively.
 */

#include "../inc/champsim_crc2.h"
#include <map>
#include <cstring>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ============================================================================
// LEADER-FOLLOWER SAMPLING (64 sampled sets for mode selection)
// ============================================================================
#define LLC_SET_BITS 11  // log2(2048)

static inline bool SAMPLED_SET(uint32_t set) {
    // Sample 64 sets: low 6 bits == high 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}

static inline bool LEADER_A(uint32_t set) {
    return SAMPLED_SET(set) && ((set & 1u) == 0u);  // Even sampled sets
}

static inline bool LEADER_B(uint32_t set) {
    return SAMPLED_SET(set) && ((set & 1u) == 1u);  // Odd sampled sets
}

static inline uint32_t LEADER_SLOT(uint32_t set) {
    return (set & 63u);  // 0..63
}

// ============================================================================
// MODE SELECTOR STATE (Conservative gating)
// ============================================================================
static int32_t leaderA_score = 0;     // Hits - misses on Mode A leader sets
static int32_t leaderB_score = 0;     // Hits - misses on Mode B leader sets
static bool prefer_mode_B = false;    // Global gate: starts false, requires decisive win
static uint64_t selector_epoch = 0;   // Access counter for periodic evaluation

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 32;            // HIGH BIAS toward Hawkeye
static constexpr int8_t MODE_B_THRESHOLD = 5;         // Need strong confidence

// Per-set confidence counters for followers (saturating -8 to +7)
static int8_t mode_confidence[LLC_SETS];

// Follower cap: limit how many followers can ever use Mode B
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;  // ~12.5% of 2048 sets
static uint32_t num_mode_B_followers = 0;
static uint8_t used_mode_B[LLC_SETS]; // 0/1 latch per set once Mode B is assigned

// ============================================================================
// MODE A: HAWKEYE - EXCLUSIVE STATE
// ============================================================================

// EXCLUSIVE RRIP for Mode A
#define maxRRPV 7
uint32_t rrpv_A[LLC_SETS][LLC_WAYS];  // Mode A's private RRIP state

// Per-set timers (only used in 64 sampled sets)
#define TIMER_SIZE 1024
uint64_t perset_mytimer[LLC_SETS];

// Signatures for sampled sets (only 64 sets)
uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched[LLC_SETS][LLC_WAYS];

// Hawkeye PC Predictors (SHCT - Signature History Counter Table)
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)  // 2048 entries

// CRC hash function for PC signatures
uint64_t CRC(uint64_t _blockAddress) {
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal = _blockAddress;
    for (unsigned int i = 0; i < 32; i++)
        _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
    return _returnVal;
}

// SHCT: PC signature -> cache-friendliness prediction
class HAWKEYE_PC_PREDICTOR {
    std::map<uint64_t, short unsigned int> SHCT;

public:
    void increment(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(signature) == SHCT.end())
            SHCT[signature] = (1 + MAX_SHCT) / 2;
        SHCT[signature] = (SHCT[signature] < MAX_SHCT) ? (SHCT[signature] + 1) : MAX_SHCT;
    }

    void decrement(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(signature) == SHCT.end())
            SHCT[signature] = (1 + MAX_SHCT) / 2;
        if (SHCT[signature] != 0)
            SHCT[signature] = SHCT[signature] - 1;
    }

    bool get_prediction(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(signature) != SHCT.end() && SHCT[signature] < ((MAX_SHCT + 1) / 2))
            return false;  // Cache-averse
        return true;       // Cache-friendly
    }
};

HAWKEYE_PC_PREDICTOR* demand_predictor;
HAWKEYE_PC_PREDICTOR* prefetch_predictor;

// OPTgen: Belady-optimal occupancy tracking
#define OPTGEN_VECTOR_SIZE 128

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
            if (liveness_history[i] >= CACHE_SIZE) {
                is_cache = false;
                break;
            }
            i = (i + 1) % liveness_history.size();
        }

        if (is_cache) {
            i = last_quanta;
            while (i != curr_quanta) {
                liveness_history[i]++;
                i = (i + 1) % liveness_history.size();
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

OPTgen perset_optgen[LLC_SETS];

// Sampler to track 8x cache history for sampled sets
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

// ============================================================================
// MODE B: HST-E DERIVED - EXCLUSIVE STATE
// ============================================================================

/*
 * MODE B DESIGN (StreamShield-D):
 * - Stride detector (+1/+2 forward steps; arm after 2) with TinyLFU per-PC hotness and periodic decay.
 * - Stream/prefetch quarantine: cold streams and all prefetches insert at hard tail (RRIP=7).
 * - Dead-PC filter (2-bit): demand misses raise deadness; hits reduce; aged toward neutral.
 * - Multi-hit rescue: non-stream lines promote to MRU at 2 hits; stream-tagged need 3 hits (tiered).
 *
 * STRICT SEPARATION: Uses only rrpv_B and Mode B metadata.
 */

// EXCLUSIVE RRIP for Mode B
uint32_t rrpv_B[LLC_SETS][LLC_WAYS];  // Mode B's private RRIP state

// Per-line metadata for Mode B
static uint8_t mode_B_hitcnt[LLC_SETS][LLC_WAYS];      // Hit counter (0-3)
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS];  // Stream quarantine flag

// Mode B PC table (512 entries for stride/frequency/deadness tracking)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // Last accessed line (10-bit)
static uint8_t mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE];  // Stream confidence (2-bit)
static uint8_t mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];         // TinyLFU frequency (4-bit)
static uint8_t mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];         // Deadness (2-bit)

// Mode B tuning parameters
static constexpr uint8_t MODE_B_INSERT_WARM      = 2;   // Near-MRU for hot PCs
static constexpr uint8_t MODE_B_INSERT_COLD      = 6;   // Near-tail for cold PCs
static constexpr uint8_t MODE_B_STREAM_TAIL      = 7;   // Hard tail for streams
static constexpr uint8_t MODE_B_STREAM_THRESHOLD = 2;   // +1/+2 forward to arm stream
static constexpr uint8_t MODE_B_HOT_THRESHOLD    = 8;   // TinyLFU hot threshold (0-15)
static constexpr uint8_t MODE_B_PROMOTE_NONSTR   = 2;   // Non-stream promote to MRU at 2 hits
static constexpr uint8_t MODE_B_PROMOTE_STR_T1   = 2;   // Stream: at 2 hits, lift to warm
static constexpr uint8_t MODE_B_PROMOTE_STR_T2   = 3;   // Stream: at 3 hits, promote to MRU
static constexpr uint64_t MODE_B_DECAY_PERIOD    = 8192;// Decay TinyLFU/deadness periodically

static inline uint32_t mode_B_pc_index(uint64_t pc) {
    return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE - 1);
}

static inline uint16_t mode_B_line_id(uint64_t paddr) {
    return (uint16_t)((paddr >> 6) & 0x3FF);  // 10-bit cache line ID
}

static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }

// Mode B: Detect stride/stream and update PC table
static bool mode_B_detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool forward = false;
    if (last != 0xFFFF) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (line_id == exp1) || (line_id == exp2);
    }

    if (forward)
        sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else
        mode_B_pc_stride_conf[idx] = 0;

    mode_B_pc_last_line[idx] = line_id;
    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_THRESHOLD);
}

// Mode B: Update insertion policy (uses ONLY rrpv_B)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pc_idx = mode_B_pc_index(PC);

    // Update TinyLFU frequency on demand access
    if (is_demand(type)) {
        sat_inc_u4(mode_B_pc_freq[pc_idx]);
    }

    // Update deadness: demand miss => more dead; demand hit => less dead
    if (is_demand(type)) {
        if (hit) sat_dec_u2(mode_B_pc_dead[pc_idx]);
        else     sat_inc_u2(mode_B_pc_dead[pc_idx]);
    }

    // Detect stream
    bool is_stream = mode_B_detect_stream(PC, paddr);
    mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Update line hit counter
    if (hit) {
        sat_inc_u2(mode_B_hitcnt[set][way]);
    } else {
        mode_B_hitcnt[set][way] = 0;
    }

    // Insertion policy (affects ONLY rrpv_B)
    bool pc_hot  = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    uint8_t pc_dead = mode_B_pc_dead[pc_idx];

    if (is_prefetch(type)) {
        // Prefetch quarantine always
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (is_stream && (!pc_hot || pc_dead >= 2)) {
        // Cold stream or dead-ish PC: hard tail to mimic bypass
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (!is_stream && pc_dead >= 3) {
        // Very dead non-stream PCs: hard tail
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (pc_hot) {
        // Hot PC: give chance near MRU
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    } else {
        // Default cold: near-tail
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue (tiered)
    if (hit) {
        if (!is_stream) {
            if (mode_B_hitcnt[set][way] >= MODE_B_PROMOTE_NONSTR) {
                rrpv_B[set][way] = 0;  // Promote non-stream to MRU
            }
        } else {
            if (mode_B_hitcnt[set][way] >= MODE_B_PROMOTE_STR_T2) {
                rrpv_B[set][way] = 0;  // Promote stream to MRU after 3 hits
            } else if (mode_B_hitcnt[set][way] >= MODE_B_PROMOTE_STR_T1) {
                // Intermediate lift to warm to allow escape from quarantine
                if (rrpv_B[set][way] > MODE_B_INSERT_WARM)
                    rrpv_B[set][way] = MODE_B_INSERT_WARM;
            }
        }
    }
}

// Mode B: Select victim (uses ONLY rrpv_B)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set) {
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid)
            return w;
    }

    // Look for maxRRPV in rrpv_B
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv_B[set][w] == maxRRPV)
            return w;
    }

    // Age all rrpv_B and retry
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv_B[set][w] < maxRRPV)
            rrpv_B[set][w]++;
    }

    // Return first maxRRPV after aging
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv_B[set][w] == maxRRPV)
            return w;
    }

    return 0;  // Fallback
}

// ============================================================================
// MODE SELECTOR LOGIC (Conservative)
// ============================================================================

static inline bool should_use_mode_B(uint32_t set) {
    // Force Mode A on A-leader sets
    if (LEADER_A(set))
        return false;

    // Force Mode B on B-leader sets
    if (LEADER_B(set))
        return true;

    // Follower sets: use Mode B only if:
    // 1. Global gate prefers Mode B
    // 2. Local confidence >= threshold
    // 3. Haven't exceeded follower cap (latch assignment once granted)
    if (!prefer_mode_B)
        return false;

    if (mode_confidence[set] < MODE_B_THRESHOLD)
        return false;

    if (!used_mode_B[set]) {
        if (num_mode_B_followers >= MAX_MODE_B_FOLLOWERS)
            return false;
        used_mode_B[set] = 1;
        num_mode_B_followers++;
    }

    return true;
}

static void decay_mode_B_counters() {
    // Periodically decay TinyLFU and deadness toward neutral
    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) {
        mode_B_pc_freq[i] >>= 1;                 // Halve frequency (4-bit)
        if (mode_B_pc_dead[i] > 0) mode_B_pc_dead[i]--; // Age deadness toward 0
    }
}

static void update_selector(uint32_t set, uint8_t hit) {
    // Update leader scores (hits = +1, misses = -1)
    if (SAMPLED_SET(set)) {
        int32_t delta = hit ? 1 : -1;
        if (LEADER_A(set))
            leaderA_score += delta;
        if (LEADER_B(set))
            leaderB_score += delta;
    }

    // Periodic evaluation
    selector_epoch++;
    if (selector_epoch % SELECTOR_EPOCH_SIZE == 0) {
        // Global decision: Mode B preferred ONLY if it outperforms by BIAS margin
        prefer_mode_B = (leaderB_score > leaderA_score + MODE_A_BIAS);

        // Drift follower confidence toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!SAMPLED_SET(s)) {
                if (prefer_mode_B) {
                    if (mode_confidence[s] < 7)
                        mode_confidence[s]++;
                } else {
                    if (mode_confidence[s] > -8)
                        mode_confidence[s]--;
                }
            }
        }

        // Decay Mode B per-PC counters
        decay_mode_B_counters();

        // Reset leader scores
        leaderA_score = 0;
        leaderB_score = 0;
    }
}

// ============================================================================
// MODE A HELPER FUNCTIONS (from hawkeye_final.cc)
// ============================================================================

void replace_addr_history_element(unsigned int sampler_set) {
    uint64_t lru_addr = 0;
    for (auto it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++) {
        if ((it->second).lru == (SAMPLER_WAYS - 1)) {
            lru_addr = it->first;
            break;
        }
    }
    addr_history[sampler_set].erase(lru_addr);
}

void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    for (auto it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++) {
        if ((it->second).lru < curr_lru) {
            (it->second).lru++;
        }
    }
}

// ============================================================================
// CHAMPSIM INTERFACE
// ============================================================================

void InitReplacementState() {
    // Initialize Mode A (Hawkeye) and Mode B arrays
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        for (uint32_t j = 0; j < LLC_WAYS; j++) {
            rrpv_A[i][j] = maxRRPV;
            rrpv_B[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
            mode_B_hitcnt[i][j] = 0;
            mode_B_stream_tag[i][j] = 0;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2);
        mode_confidence[i] = 0;
        used_mode_B[i] = 0;
    }

    addr_history.resize(SAMPLER_SETS);
    for (int i = 0; i < SAMPLER_SETS; i++)
        addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // Initialize Mode B PC tables
    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) {
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
    }

    // Initialize selector
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        return mode_B_victim(set, current_set);
    } else {
        // Mode A: Hawkeye victim selection (uses rrpv_A)
        // Look for maxRRPV line
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv_A[set][i] == maxRRPV)
                return i;
        }

        // If no maxRRPV, evict oldest cache-friendly line
        uint32_t max_rrip = 0;
        int32_t lru_victim = -1;
        for (uint32_t i = 0; i < LLC_WAYS; i++) {
            if (rrpv_A[set][i] >= max_rrip) {
                max_rrip = rrpv_A[set][i];
                lru_victim = i;
            }
        }

        // Train predictor negatively on eviction (sampled sets only)
        if (SAMPLED_SET(set) && lru_victim != -1) {
            if (prefetched[set][lru_victim])
                prefetch_predictor->decrement(signatures[set][lru_victim]);
            else
                demand_predictor->decrement(signatures[set][lru_victim]);
        }

        return lru_victim;
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu;
    (void)victim_addr;

    paddr = (paddr >> 6) << 6;  // Align to cache line

    // Track prefetch flag
    if (type == ACCESS_PREFETCH) {
        if (!hit)
            prefetched[set][way] = true;
    } else {
        prefetched[set][way] = false;
    }

    // Ignore writebacks
    if (is_writeback(type))
        return;

    // Update selector
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // ALWAYS train Mode A on sampled sets (for OPTgen)
    if (SAMPLED_SET(set)) {
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        // OPTgen training (Hawkeye logic - simplified)
        auto it = addr_history[sampler_set].find(sampler_tag);
        if (it != addr_history[sampler_set].end()) {
            uint64_t last_quanta = it->second.last_quanta;
            bool is_cache_friendly = perset_optgen[set].should_cache(curr_quanta, last_quanta);

            if (it->second.prefetched) {
                if (is_cache_friendly)
                    prefetch_predictor->increment(it->second.PC);
                else
                    prefetch_predictor->decrement(it->second.PC);
            } else {
                if (is_cache_friendly)
                    demand_predictor->increment(it->second.PC);
                else
                    demand_predictor->decrement(it->second.PC);
            }

            update_addr_history_lru(sampler_set, it->second.lru);
            it->second.update(curr_quanta, PC, is_cache_friendly);
            it->second.lru = 0;
        } else {
            if (addr_history[sampler_set].size() == SAMPLER_WAYS)
                replace_addr_history_element(sampler_set);

            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            addr_history[sampler_set][sampler_tag].update(curr_quanta, PC, false);
            addr_history[sampler_set][sampler_tag].lru = 0;
            if (type == ACCESS_PREFETCH)
                addr_history[sampler_set][sampler_tag].mark_prefetch();

            update_addr_history_lru(sampler_set, 0);
        }

        perset_optgen[set].add_access(curr_quanta);
        perset_mytimer[set] = (perset_mytimer[set] + 1) % TIMER_SIZE;
    }

    // Get Hawkeye's prediction (always compute for training)
    bool hawkeye_prediction = demand_predictor->get_prediction(PC);
    if (type == ACCESS_PREFETCH)
        hawkeye_prediction = prefetch_predictor->get_prediction(PC);

    signatures[set][way] = PC;

    // EXCLUSIVE UPDATE: Apply chosen mode's insertion policy
    if (use_mode_B) {
        // Mode B insertion (updates ONLY rrpv_B)
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A: Hawkeye insertion (updates ONLY rrpv_A)
        if (!hawkeye_prediction) {
            rrpv_A[set][way] = maxRRPV;  // Cache-averse
        } else {
            rrpv_A[set][way] = 0;  // Cache-friendly at MRU
            if (!hit) {
                // Age cache-friendly lines
                bool saturated = false;
                for (uint32_t i = 0; i < LLC_WAYS; i++) {
                    if (rrpv_A[set][i] == maxRRPV - 1)
                        saturated = true;
                }
                if (!saturated) {
                    for (uint32_t i = 0; i < LLC_WAYS; i++) {
                        if (rrpv_A[set][i] < maxRRPV - 1)
                            rrpv_A[set][i]++;
                    }
                }
            }
            rrpv_A[set][way] = 0;
        }
    }

    // Periodic Mode B decay (in case selector epoch desynchronizes)
    if ((selector_epoch % MODE_B_DECAY_PERIOD) == 0 && selector_epoch != 0) {
        decay_mode_B_counters();
    }
}

void PrintStats_Heartbeat() {}

void PrintStats() {
    unsigned int hits = 0;
    unsigned int accesses = 0;
    for (unsigned int i = 0; i < LLC_SETS; i++) {
        accesses += perset_optgen[i].access;
        hits += perset_optgen[i].get_num_opt_hits();
    }

    std::cout << "OPTgen accesses: " << accesses << std::endl;
    std::cout << "OPTgen hits: " << hits << std::endl;
    std::cout << "OPTgen hit rate: " << 100.0 * (double)hits / (double)accesses << std::endl;
    std::cout << std::endl;
}