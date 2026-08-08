/*
 * STRICT DYNAMIC ENSEMBLE: Hawkeye (Mode A) + SPICE (Mode B)
 *
 * GUARANTEES:
 * - Strict state separation: rrpv_A and Hawkeye metadata are exclusive to Mode A.
 * - rrpv_B and Mode B metadata are exclusive to Mode B.
 * - Selector is conservative: high bias toward Hawkeye, per-set threshold, follower cap.
 * - Mode B may bypass on scans; never updates rrpv_A.
 *
 * TARGET LIFTS: gcc, omnetpp, mcf (scan/pointer-heavy)
 * KEY TUNABLES (see constants):
 *   - Stream detection window (±1/±2), arming threshold
 *   - TinyLFU hot threshold + periodic decay cadence
 *   - Dead-on-fill microfilter threshold
 *   - Quarantine depths and bypass gating
 *   - Multi-hit rescue thresholds (stream vs non-stream) and set temperature
 */

#include "../inc/champsim_crc2.h"
#include <map>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <iostream>

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
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u)); // low6 == high6
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// MODE SELECTOR STATE (conservative gating)
// ============================================================================
static int32_t leaderA_score = 0;
static int32_t leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;     // High bias toward Hawkeye
static constexpr int8_t MODE_B_THRESHOLD = 5;  // Per-set confidence gate

// Per-set confidence counters (-8..+7). Implemented as int8_t.
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5% of sets)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1 tag per follower set

// ============================================================================
// MODE A: HAWKEYE - EXCLUSIVE STATE
// (Compact Hawkeye embedding; state is separate from Mode B)
// ============================================================================
#define maxRRPV 7
static uint32_t rrpv_A[LLC_SETS][LLC_WAYS];

#define TIMER_SIZE 1024
static uint64_t perset_mytimer[LLC_SETS];

// Signatures for sampled sets (only used on sampled sets)
static uint64_t signatures[LLC_SETS][LLC_WAYS];
static bool prefetched[LLC_SETS][LLC_WAYS];

// SHCT predictors (PC -> friendliness)
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)

static uint64_t CRC(uint64_t _blockAddress) {
    static const uint64_t crcPolynomial = 3988292384ULL;
    uint64_t v = _blockAddress;
    for (unsigned int i = 0; i < 32; i++)
        v = ((v & 1) ? ((v >> 1) ^ crcPolynomial) : (v >> 1));
    return v;
}

class HAWKEYE_PC_PREDICTOR {
    std::map<uint64_t, uint8_t> SHCT;
public:
    void increment(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        auto &c = SHCT[sig];
        if (c == 0) c = (MAX_SHCT + 1) / 2;
        c = std::min<uint8_t>(c + 1, MAX_SHCT);
    }
    void decrement(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        auto &c = SHCT[sig];
        if (c == 0) c = (MAX_SHCT + 1) / 2;
        if (c) c--;
    }
    bool get_prediction(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        auto it = SHCT.find(sig);
        if (it != SHCT.end() && it->second < ((MAX_SHCT + 1) / 2)) return false;
        return true;
    }
};

static HAWKEYE_PC_PREDICTOR* demand_predictor;
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;

// OPTgen-lite sampler for Hawkeye (compact)
#define OPTGEN_VECTOR_SIZE 128
struct ADDR_INFO {
    uint32_t last_quanta;
    uint64_t PC;
    bool prefetched;
    uint8_t lru;
    void init(unsigned int q) { last_quanta = q; PC = 0; prefetched = false; lru = 0; }
    void update(unsigned int q, uint64_t pc) { last_quanta = q; PC = pc; }
    void mark_prefetch() { prefetched = true; }
};

struct OPTgen {
    std::vector<uint8_t> live; // tiny vector
    uint64_t access;
    uint8_t CAP;
    void init(uint8_t size) { CAP = size; access = 0; live.assign(OPTGEN_VECTOR_SIZE, 0); }
    void add_access(uint64_t q) { access++; live[q] = 0; }
    bool should_cache(uint64_t curr_q, uint64_t last_q) {
        // count occupancy in [last_q, curr_q)
        uint8_t occ = 0;
        uint64_t i = last_q;
        while (i != curr_q) {
            occ = std::max<uint8_t>(occ, live[i]);
            i = (i + 1) % live.size();
        }
        bool cache = (occ < CAP);
        // increment occupancy if cached
        if (cache) {
            i = last_q;
            while (i != curr_q) {
                if (live[i] < 255) live[i]++;
                i = (i + 1) % live.size();
            }
        }
        return cache;
    }
};

static OPTgen perset_optgen[LLC_SETS];

#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
static std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

// Hawkeye helpers
static void replace_addr_history_element(unsigned int sampler_set) {
    if (addr_history[sampler_set].empty()) return;
    // Erase an entry with LRU == SAMPLER_WAYS-1
    for (auto it = addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); ++it) {
        if (it->second.lru == SAMPLER_WAYS - 1) { addr_history[sampler_set].erase(it); return; }
    }
    // Fallback: erase first
    addr_history[sampler_set].erase(addr_history[sampler_set].begin());
}
static void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru) {
    for (auto &kv : addr_history[sampler_set]) {
        if (kv.second.lru < curr_lru) kv.second.lru++;
        if (kv.second.lru >= SAMPLER_WAYS) kv.second.lru = SAMPLER_WAYS - 1;
    }
}

// ============================================================================
// MODE B: SPICE - EXCLUSIVE STATE
// - 2-bit RRIP to fit budget
// - Stride/run detector (±1/±2), TinyLFU with decay, PC dead microfilter
// - Stream/prefetch quarantine, guarded bypass, multi-hit rescue with set temp
// ============================================================================
#define maxRRPV_B 3 // 2-bit RRIP for Mode B (0..3)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS]; // ONLY Mode B updates

// Per-line Mode B metadata (exclusive)
static uint8_t mode_B_hit2[LLC_SETS][LLC_WAYS];     // 2-bit hit counter (0..3)
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS]; // 0/1

// PC table (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE]; // 10-bit id; init 0xFFFF
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 0..3
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];  // 4-bit TinyLFU (0..15)
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];  // 2-bit dead-on-fill score (0..3)

static inline uint32_t mode_B_pc_index(uint64_t pc) { return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE - 1); }
static inline uint16_t mode_B_line_id(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit line id
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }

// Tunables
static constexpr uint8_t  MODE_B_STREAM_ARM = 2;     // need 2 steps to arm stream
static constexpr uint8_t  MODE_B_HOT_THR    = 8;     // TinyLFU hot if >= 8
static constexpr uint8_t  MODE_B_DEAD_THR   = 2;     // PC dead if >= 2
static constexpr uint8_t  MODE_B_INS_WARM   = 1;     // near-MRU for hot PCs (2-bit RRIP)
static constexpr uint8_t  MODE_B_INS_COLD   = 3;     // near-tail for cold PCs
static constexpr uint8_t  MODE_B_INS_TAIL   = 3;     // hard tail for streams/prefetch
static constexpr uint8_t  MODE_B_PROMO_NONSTREAM = 2;// 2 hits to MRU
static constexpr uint8_t  MODE_B_PROMO_STREAM    = 3;// 3 hits to MRU (relaxed if hot set)
static constexpr uint64_t MODE_B_DECAY_PERIOD    = 8192; // accesses per decay
static constexpr uint64_t MODE_B_SETCLR_PERIOD   = 4096; // accesses per set-hot clear
static constexpr bool     MODE_B_BYPASS_ENABLE   = true;

// Per-set temperature (1-bit: 0=cold, 1=hot bursty)
static uint8_t mode_B_set_hot[LLC_SETS];

// Access accounting for decay and set-hot maintenance
static uint64_t mode_B_access_ctr = 0;

// Eviction-dead flag latched per set for the last victim taken by Mode B
static uint8_t mode_B_evicted_dead[LLC_SETS];

// Peek stream/run without mutating state (±1/±2, forward/back)
static bool mode_B_peek_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];
    if (last == 0xFFFF) return false;
    bool step = (line == (uint16_t)(last + 1)) || (line == (uint16_t)(last + 2)) ||
                (line == (uint16_t)(last - 1)) || (line == (uint16_t)(last - 2));
    // Would arming happen this access?
    return step && (mode_B_pc_stride_conf[idx] + 1 >= MODE_B_STREAM_ARM);
}

// Update stride/run detector (mutating)
static bool mode_B_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];
    bool step = false;
    if (last != 0xFFFF) {
        step = (line == (uint16_t)(last + 1)) || (line == (uint16_t)(last + 2)) ||
               (line == (uint16_t)(last - 1)) || (line == (uint16_t)(last - 2));
    }
    if (step) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else mode_B_pc_stride_conf[idx] = 0;
    mode_B_pc_last_line[idx] = line;
    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM);
}

// TinyLFU periodic decay
static void mode_B_decay_freq() {
    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) mode_B_pc_freq[i] >>= 1;
}

// Mode B bypass decision (guard scans/prefetch pollution)
static bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type) {
    if (!MODE_B_BYPASS_ENABLE) return false;
    uint32_t pi = mode_B_pc_index(PC);
    bool hot_pc   = (mode_B_pc_freq[pi] >= MODE_B_HOT_THR) && (mode_B_pc_dead[pi] < MODE_B_DEAD_THR);
    bool stream   = mode_B_peek_stream(PC, paddr);
    bool hot_set  = (mode_B_set_hot[set] != 0);
    if (is_prefetch(type)) {
        // Conservative: quarantine prefetches; bypass allowed if stream and cold PC
        return (stream && !hot_pc && !hot_set);
    }
    // Demand: bypass only on clear scans from dead/cold PCs in cold sets
    if (is_demand(type) && stream && !hot_pc && !hot_set && (mode_B_pc_dead[pi] >= MODE_B_DEAD_THR))
        return true;
    return false;
}

// Mode B victim (only rrpv_B)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    // Try bypass for scans/prefetches
    if (mode_B_should_bypass(set, PC, paddr, type))
        return 16; // bypass

    // Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (!current_set[w].valid) return w;

    // Look for maxRRPV in rrpv_B
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV_B) {
            // Latch dead-on-fill for training if victim had zero hits
            mode_B_evicted_dead[set] = (mode_B_hit2[set][w] == 0) ? 1 : 0;
            return w;
        }

    // Age all and retry
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] < maxRRPV_B) rrpv_B[set][w]++;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV_B) {
            mode_B_evicted_dead[set] = (mode_B_hit2[set][w] == 0) ? 1 : 0;
            return w;
        }

    mode_B_evicted_dead[set] = 0;
    return 0;
}

// Mode B insertion/update (only rrpv_B + Mode B metadata)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pi = mode_B_pc_index(PC);
    if (is_demand(type)) sat_inc_u4(mode_B_pc_freq[pi]);

    // Decay TinyLFU + clear set-hot periodically
    mode_B_access_ctr++;
    if ((mode_B_access_ctr % MODE_B_DECAY_PERIOD) == 0) mode_B_decay_freq();
    if ((mode_B_access_ctr % MODE_B_SETCLR_PERIOD) == 0) {
        // epoch clear for set temperature
        std::memset(mode_B_set_hot, 0, sizeof(mode_B_set_hot));
    }

    // Promote set temperature on hits
    if (hit) mode_B_set_hot[set] = 1;

    // Stream detect (mutating)
    bool is_stream = mode_B_update_stream(PC, paddr);
    mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Dead-on-fill training: if the last victim (same set) died without a hit, blame current PC
    if (!hit && mode_B_evicted_dead[set]) {
        if (mode_B_pc_dead[pi] < 3) mode_B_pc_dead[pi]++;
        mode_B_evicted_dead[set] = 0;
    }
    // Redeem PC on useful hits
    if (hit && mode_B_pc_dead[pi] > 0) mode_B_pc_dead[pi]--;

    // Update per-line hit count
    if (hit) {
        if (mode_B_hit2[set][way] < 3) mode_B_hit2[set][way]++;
    } else {
        mode_B_hit2[set][way] = 0;
    }

    // Insertion policy
    bool pc_hot   = (mode_B_pc_freq[pi] >= MODE_B_HOT_THR) && (mode_B_pc_dead[pi] < MODE_B_DEAD_THR);
    bool hot_set  = (mode_B_set_hot[set] != 0);

    if (is_prefetch(type) || (is_stream && !pc_hot)) {
        rrpv_B[set][way] = MODE_B_INS_TAIL; // quarantine
    } else if (pc_hot || hot_set) {
        rrpv_B[set][way] = MODE_B_INS_WARM; // shallow insert
    } else {
        rrpv_B[set][way] = MODE_B_INS_COLD; // near-tail
    }

    // Multi-hit rescue
    if (hit) {
        uint8_t need = mode_B_stream_tag[set][way] ? MODE_B_PROMO_STREAM : MODE_B_PROMO_NONSTREAM;
        if (hot_set && need > 1) need--; // Relax in hot sets
        if (mode_B_hit2[set][way] >= need) {
            rrpv_B[set][way] = 0; // MRU
        }
    }
}

// ============================================================================
// SELECTOR LOGIC
// ============================================================================
static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    if (!prefer_mode_B) return false;
    if (mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if (follower_is_B[set] == 0 && num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
    return true;
}

static void update_selector(uint32_t set, uint8_t hit, bool used_B) {
    if (SAMPLED_SET(set)) {
        int32_t delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if (selector_epoch % SELECTOR_EPOCH_SIZE == 0) {
        prefer_mode_B = (leaderB_score > leaderA_score + MODE_A_BIAS);

        // Drift local confidence toward global preference
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (SAMPLED_SET(s)) continue;
            if (prefer_mode_B) { if (mode_confidence[s] < 7) mode_confidence[s]++; }
            else               { if (mode_confidence[s] > -8) mode_confidence[s]--; }

            // Allocate follower slots opportunistically
            if (prefer_mode_B && mode_confidence[s] >= MODE_B_THRESHOLD && follower_is_B[s] == 0 && num_mode_B_followers < MAX_MODE_B_FOLLOWERS) {
                follower_is_B[s] = 1;
                num_mode_B_followers++;
            }
            if (!prefer_mode_B && follower_is_B[s]) {
                follower_is_B[s] = 0;
                if (num_mode_B_followers) num_mode_B_followers--;
            }
        }
        leaderA_score = 0; leaderB_score = 0;
    }
}

// ============================================================================
// CHAMPSIM INTERFACE
// ============================================================================
void InitReplacementState() {
    // Mode A init
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        for (uint32_t j = 0; j < LLC_WAYS; j++) {
            rrpv_A[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2);
        mode_confidence[i] = 0;
        follower_is_B[i] = 0;
    }

    addr_history.resize(SAMPLER_SETS);
    for (auto &m : addr_history) m.clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // Mode B init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv_B[s][w] = maxRRPV_B;
            mode_B_hit2[s][w] = 0;
            mode_B_stream_tag[s][w] = 0;
        }
        mode_B_set_hot[s] = 0;
        mode_B_evicted_dead[s] = 0;
    }
    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) {
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
    }
    mode_B_access_ctr = 0;

    // Selector init
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu;
    bool use_B = should_use_mode_B(set);
    if (use_B) {
        uint32_t v = mode_B_victim(set, current_set, PC, paddr, type);
        return v;
    }

    // Mode A: Hawkeye victim selection (only rrpv_A)
    // Prefer invalid
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (!current_set[i].valid) return i;

    // First try maxRRPV
    for (uint32_t i = 0; i < LLC_WAYS; i++)
        if (rrpv_A[set][i] == maxRRPV) return i;

    // Otherwise, evict the oldest among high RRPV
    uint32_t max_rrip = 0; int32_t vic = -1;
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        if (rrpv_A[set][i] >= max_rrip) { max_rrip = rrpv_A[set][i]; vic = i; }
    }
    return (vic >= 0) ? (uint32_t)vic : 0;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;

    // Track prefetch flag for Hawkeye (sampled sets)
    if (type == ACCESS_PREFETCH) {
        if (!hit) prefetched[set][way] = true;
    } else prefetched[set][way] = false;

    if (is_writeback(type)) return;

    // Selector bookkeeping
    bool used_B = should_use_mode_B(set);
    update_selector(set, hit, used_B);

    // Always train Hawkeye on sampled sets (OPTgen and SHCT)
    if (SAMPLED_SET(set)) {
        uint64_t curr_q = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        auto &hist = addr_history[sampler_set];
        auto it = hist.find(sampler_tag);
        if (it != hist.end()) {
            // Seen before
            bool cached = perset_optgen[set].should_cache(curr_q, it->second.last_quanta);
            if (it->second.prefetched) {
                if (cached) prefetch_predictor->increment(it->second.PC);
                else        prefetch_predictor->decrement(it->second.PC);
            } else {
                if (cached) demand_predictor->increment(it->second.PC);
                else        demand_predictor->decrement(it->second.PC);
            }
            update_addr_history_lru(sampler_set, it->second.lru);
            it->second.update(curr_q, PC);
            it->second.lru = 0;
        } else {
            if (hist.size() == SAMPLER_WAYS) replace_addr_history_element(sampler_set);
            auto &e = hist[sampler_tag];
            e.init(curr_q); e.update(curr_q, PC); e.lru = 0;
            if (type == ACCESS_PREFETCH) e.mark_prefetch();
            update_addr_history_lru(sampler_set, 0);
        }
        perset_optgen[set].add_access(curr_q);
        perset_mytimer[set] = (perset_mytimer[set] + 1) % TIMER_SIZE;
    }

    // Get Hawkeye prediction (for insertion when Mode A is used)
    bool hawkeye_pred = (type == ACCESS_PREFETCH) ? prefetch_predictor->get_prediction(PC)
                                                  : demand_predictor->get_prediction(PC);
    signatures[set][way] = PC;

    // EXCLUSIVE UPDATES
    if (used_B) {
        // Mode B insertion/update (only rrpv_B + Mode B metadata)
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A insertion/update (only rrpv_A + Hawkeye metadata)
        if (!hawkeye_pred) {
            rrpv_A[set][way] = maxRRPV; // averse
        } else {
            // friendly insert (near MRU)
            rrpv_A[set][way] = 0;
            if (!hit) {
                // gentle aging of others
                bool sat = false;
                for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_A[set][i] == maxRRPV - 1) sat = true;
                if (!sat) for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_A[set][i] < maxRRPV - 1) rrpv_A[set][i]++;
                rrpv_A[set][way] = 0;
            }
        }
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}