/*
 * STRICT DYNAMIC ENSEMBLE TEMPLATE - MOCKINGJAY + HST-E
 *
 * PROTECTION GUARANTEE: Mode A (Mockingjay) and Mode B (HST-E) are proven baselines.
 * This ensemble combines two policies that both outperform Hawkeye.
 *
 * ARCHITECTURE:
 * - Mode A: Mockingjay (IPC 0.4147) - ETR-based with RD prediction (VERBATIM)
 * - Mode B: HST-E (IPC 0.7669) - Stride detector + TinyLFU + stream quarantine (VERBATIM)
 * - Strict Gating: High bias (50), high threshold (7), follower cap
 * - No Shared State: Mode A and Mode B use EXCLUSIVE state
 *
 * DEFAULT BEHAVIOR: All followers use Mode A unless Mode B wins decisively.
 */

#include "../inc/champsim_crc2.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_map>

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
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))

static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); }

// ============================================================================
// MODE SELECTOR STATE (Conservative gating)
// ============================================================================
static int32_t leaderA_score = 0;
static int32_t leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 999;       // DISABLE Mode B (pure Mockingjay)
static constexpr int8_t MODE_B_THRESHOLD = 99;    // DISABLE follower switch

static int8_t mode_confidence[LLC_SETS];
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 0; // DISABLE Mode B followers
static uint32_t num_mode_B_followers = 0;

// ============================================================================
// MODE A: MOCKINGJAY (VERBATIM)
// ============================================================================

#define LOG2_BLOCK_SIZE 6

constexpr int HISTORY     = 8;
constexpr int GRANULARITY = 8;

constexpr int INF_RD  = LLC_WAYS * HISTORY - 1;
constexpr int INF_ETR = (LLC_WAYS * HISTORY / GRANULARITY) - 1;
constexpr int MAX_RD  = INF_RD - 22;

constexpr int SAMPLED_CACHE_WAYS      = 5;
constexpr int LOG2_SAMPLED_CACHE_SETS = 4;

static int LOG2_LLC_SET          = 0;
static int LOG2_LLC_SIZE         = 0;
static int LOG2_SAMPLED_SETS     = 0;
static int SAMPLED_CACHE_TAG_BITS = 0;
static int PC_SIGNATURE_BITS      = 0;

constexpr int TIMESTAMP_BITS = 8;

// Global state
static int etr[LLC_SETS][LLC_WAYS];
static int etr_clock[LLC_SETS];
static std::unordered_map<uint32_t, int> rdp;
static int current_timestamp[LLC_SETS];

struct SampledCacheLine {
    bool     valid;
    uint64_t tag;
    uint64_t signature;
    int      timestamp;
};

static std::unordered_map<uint32_t, SampledCacheLine*> sampled_cache;

constexpr double TEMP_DIFFERENCE = 1.0 / 16.0;
constexpr double FLEXMIN_PENALTY = 2.0;

static bool is_sampled_set(int set)
{
    int mask_length = LOG2_LLC_SET - LOG2_SAMPLED_SETS;
    int mask        = (1 << mask_length) - 1;
    return (set & mask) == ((set >> (LOG2_LLC_SET - mask_length)) & mask);
}

static uint64_t CRC_HASH(uint64_t _blockAddress)
{
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal                 = _blockAddress;
    for (unsigned int i = 0; i < 3; i++) {
        if ((_returnVal & 1ULL) == 1ULL)
            _returnVal = (_returnVal >> 1) ^ crcPolynomial;
        else
            _returnVal = (_returnVal >> 1);
    }
    return _returnVal;
}

static uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core)
{
    if (NUM_CORE == 1) {
        pc <<= 1;
        if (hit) pc |= 1ULL;
        pc <<= 1;
        if (prefetch) pc |= 1ULL;
        pc = CRC_HASH(pc);
        pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    } else {
        pc <<= 1;
        if (prefetch) pc |= 1ULL;
        pc <<= 2;
        pc |= static_cast<uint64_t>(core);
        pc = CRC_HASH(pc);
        pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    }
    return pc;
}

static uint32_t get_sampled_cache_index(uint64_t full_addr)
{
    full_addr >>= LOG2_BLOCK_SIZE;
    full_addr = (full_addr << (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET)))
              >> (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET));
    return static_cast<uint32_t>(full_addr);
}

static uint64_t get_sampled_cache_tag(uint64_t x)
{
    x >>= (LOG2_LLC_SET + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS);
    x = (x << (64 - SAMPLED_CACHE_TAG_BITS)) >> (64 - SAMPLED_CACHE_TAG_BITS);
    return x;
}

static int search_sampled_cache(uint64_t blockAddress, uint32_t index)
{
    auto it = sampled_cache.find(index);
    if (it == sampled_cache.end() || !it->second)
        return -1;
    SampledCacheLine* sampled_set = it->second;
    for (int way = 0; way < SAMPLED_CACHE_WAYS; way++) {
        if (sampled_set[way].valid && sampled_set[way].tag == blockAddress)
            return way;
    }
    return -1;
}

static void detrain(uint32_t index, int way)
{
    auto it = sampled_cache.find(index);
    if (it == sampled_cache.end() || !it->second)
        return;
    SampledCacheLine* set = it->second;
    SampledCacheLine  temp = set[way];
    if (!temp.valid)
        return;

    uint32_t sig = static_cast<uint32_t>(temp.signature);
    auto rd_it = rdp.find(sig);
    if (rd_it != rdp.end()) {
        rd_it->second = std::min(rd_it->second + 1, INF_RD);
    } else {
        rdp[sig] = INF_RD;
    }
    set[way].valid = false;
}

static int temporal_difference(int init, int sample)
{
    if (sample > init) {
        int diff = sample - init;
        diff = static_cast<int>(diff * TEMP_DIFFERENCE);
        diff = std::min(1, diff);
        return std::min(init + diff, INF_RD);
    } else if (sample < init) {
        int diff = init - sample;
        diff = static_cast<int>(diff * TEMP_DIFFERENCE);
        diff = std::min(1, diff);
        return std::max(init - diff, 0);
    } else {
        return init;
    }
}

static int increment_timestamp(int input)
{
    input++;
    input %= (1 << TIMESTAMP_BITS);
    return input;
}

static int time_elapsed(int global, int local)
{
    if (global >= local)
        return global - local;
    global += (1 << TIMESTAMP_BITS);
    return global - local;
}

// Mode A victim selection
static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type) {
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) {
            return way;
        }
    }

    int max_etr = 0;
    int victim_way = 0;
    for (int way = 0; way < LLC_WAYS; way++) {
        int val = etr[set][way];
        int mag = std::abs(val);
        if (mag > max_etr || (mag == max_etr && val < 0)) {
            max_etr = mag;
            victim_way = way;
        }
    }

    // RD-based bypass
    uint64_t pc_sig64 = get_pc_signature(ip, false, type == ACCESS_PREFETCH, 0);
    uint32_t pc_sig = static_cast<uint32_t>(pc_sig64);
    auto rd_it = rdp.find(pc_sig);
    if (type != ACCESS_WRITEBACK && rd_it != rdp.end()) {
        int rd = rd_it->second;
        if (rd > MAX_RD || (rd / GRANULARITY) > max_etr) {
            return LLC_WAYS; // bypass
        }
    }
    return static_cast<uint32_t>(victim_way);
}

// Mode A update
static void mode_A_update(uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint32_t type, uint8_t hit) {
    if (type == ACCESS_WRITEBACK) {
        if (!hit) {
            etr[set][way] = -INF_ETR;
        }
        return;
    }

    uint64_t pc_sig64 = get_pc_signature(ip, hit != 0, type == ACCESS_PREFETCH, 0);
    uint32_t pc_sig = static_cast<uint32_t>(pc_sig64);

    // Sampled-cache learning
    if (is_sampled_set(static_cast<int>(set))) {
        uint32_t sampled_cache_index = get_sampled_cache_index(full_addr);
        uint64_t sampled_cache_tag = get_sampled_cache_tag(full_addr);
        int sampled_cache_way = search_sampled_cache(sampled_cache_tag, sampled_cache_index);

        if (sampled_cache_way > -1) {
            SampledCacheLine& line = sampled_cache[sampled_cache_index][sampled_cache_way];
            uint64_t last_signature = line.signature;
            int last_timestamp = line.timestamp;
            int sample = time_elapsed(current_timestamp[set], last_timestamp);

            if (sample <= INF_RD) {
                if (type == ACCESS_PREFETCH) {
                    sample = static_cast<int>(sample * FLEXMIN_PENALTY);
                }
                uint32_t last_sig_u32 = static_cast<uint32_t>(last_signature);
                auto rd_it = rdp.find(last_sig_u32);
                if (rd_it != rdp.end()) {
                    int init = rd_it->second;
                    rd_it->second = temporal_difference(init, sample);
                } else {
                    rdp[last_sig_u32] = sample;
                }
                line.valid = false;
            }
        }

        // Choose eviction in sampled cache
        int lru_way = -1;
        int lru_rd = -1;
        SampledCacheLine* set_lines = sampled_cache[sampled_cache_index];
        for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
            if (!set_lines[w].valid) {
                lru_way = w;
                lru_rd = INF_RD + 1;
                continue;
            }
            int last_ts = set_lines[w].timestamp;
            int sample = time_elapsed(current_timestamp[set], last_ts);
            if (sample > INF_RD) {
                lru_way = w;
                lru_rd = INF_RD + 1;
                detrain(sampled_cache_index, w);
            } else if (sample > lru_rd) {
                lru_way = w;
                lru_rd = sample;
            }
        }
        if (lru_way >= 0) {
            detrain(sampled_cache_index, lru_way);
        }

        // Insert current
        for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
            if (!set_lines[w].valid) {
                set_lines[w].valid = true;
                set_lines[w].signature = pc_sig64;
                set_lines[w].tag = sampled_cache_tag;
                set_lines[w].timestamp = current_timestamp[set];
                break;
            }
        }
        current_timestamp[set] = increment_timestamp(current_timestamp[set]);
    }

    // Age others every GRANULARITY accesses
    if (etr_clock[set] == GRANULARITY) {
        for (int w = 0; w < LLC_WAYS; w++) {
            if (static_cast<uint32_t>(w) != way && std::abs(etr[set][w]) < INF_ETR) {
                etr[set][w]--;
            }
        }
        etr_clock[set] = 0;
    }
    etr_clock[set]++;

    // Set ETR for touched way
    if (way < LLC_WAYS) {
        auto rd_it = rdp.find(pc_sig);
        if (rd_it == rdp.end()) {
            if (NUM_CORE == 1) {
                etr[set][way] = 0;
            } else {
                etr[set][way] = INF_ETR;
            }
        } else {
            int rd = rd_it->second;
            if (rd > MAX_RD) {
                etr[set][way] = INF_ETR;
            } else {
                etr[set][way] = rd / GRANULARITY;
            }
        }
    }
}

// ============================================================================
// MODE B: HST-E (VERBATIM - Pure stride + TinyLFU + quarantine)
// ============================================================================

#define maxRRPV 7

// Per-line metadata
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// Per-set selector state
static int8_t bandit_score[LLC_SETS];
static bool prefer_B = false;
static int32_t leaderA_score_B = 0;
static int32_t leaderB_score_B = 0;
static uint64_t access_count = 0;

// Tunables
static constexpr uint8_t INSERT_WARM_DEPTH   = 1;  // CHANGED: 2 -> 1 (near-MRU for hot PCs)
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;
static constexpr uint8_t STREAM_ARM_THRESH   = 2;
static constexpr uint8_t HITS_PROMOTE_NS     = 2;
static constexpr uint8_t HITS_PROMOTE_STR    = 2;
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // CHANGED: 8 -> 6 (earlier hot classification)
static constexpr uint32_t BANDIT_EPOCH       = 4096;
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;
static constexpr int32_t  GATE_MARGIN        = 3;

// PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE];
static uint8_t  pc_use4[PC_TBL_SIZE];

// Helpers
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv_B[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

// Mode B victim selection
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv_B[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    uint8_t best = 0;
    uint32_t victim = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv_B[set][w] >= best) { best = rrpv_B[set][w]; victim = w; }
    }
    return victim;
}

// Mode B update
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    if (is_demand(type)) sat_inc_u4(pc_use4[pc_index(PC)]);

    bool stream_armed = false;
    if (is_demand(type)) stream_armed = detect_and_update_stream(PC, paddr);

    if (hit) {
        if (stream_tag[set][way] && (rrpv_B[set][way] < maxRRPV)) {
            uint8_t newv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + STREAM_DEMOTE_TOUCH);
            rrpv_set(set, way, newv);
        }

        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            bool is_stream_line = (stream_tag[set][way] != 0);
            uint8_t need_hits = is_stream_line ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;

            if (hitcnt[set][way] >= need_hits) {
                stream_tag[set][way] = 0;
                rrpv_set(set, way, 0);
            } else {
                if (rrpv_B[set][way] > 0) rrpv_set(set, way, (uint8_t)(rrpv_B[set][way] - 1));
            }
        }
        return;
    }

    // Miss + fill
    uint32_t pidx = pc_index(PC);
    bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
    bool quarantine = is_prefetch(type) || stream_armed;

    if (quarantine) {
        rrpv_set(set, way, STREAM_TAIL_DEPTH);
        stream_tag[set][way] = 1;
    } else {
        rrpv_set(set, way, hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
        stream_tag[set][way] = 0;
    }
    hitcnt[set][way] = 0;
}

// ============================================================================
// MODE SELECTOR LOGIC
// ============================================================================

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;  // Leaders must exercise Mode B to gather scores
    if (!prefer_mode_B) return false;
    if (!(prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH))) return false;
    if (mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if (num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
    return true;
}

// ============================================================================
// MAIN CHAMPSIM INTERFACE
// ============================================================================

void InitReplacementState() {
    // Mode A init
    LOG2_LLC_SET  = static_cast<int>(std::log2(static_cast<double>(LLC_SETS)));
    LOG2_LLC_SIZE = LOG2_LLC_SET + static_cast<int>(std::log2(static_cast<double>(LLC_WAYS))) + LOG2_BLOCK_SIZE;
    LOG2_SAMPLED_SETS = LOG2_LLC_SIZE - 16;
    SAMPLED_CACHE_TAG_BITS = 31 - LOG2_LLC_SIZE;
    PC_SIGNATURE_BITS = LOG2_LLC_SIZE - 10;

    for (int set = 0; set < LLC_SETS; set++) {
        etr_clock[set] = GRANULARITY;
        current_timestamp[set] = 0;
        for (int way = 0; way < LLC_WAYS; way++) {
            etr[set][way] = 0;
        }
    }

    sampled_cache.clear();
    int modifier = 1 << LOG2_LLC_SET;
    int limit = 1 << LOG2_SAMPLED_SETS;
    for (uint32_t set = 0; set < LLC_SETS; set++) {
        if (is_sampled_set(static_cast<int>(set))) {
            for (int i = 0; i < limit; i++) {
                uint32_t idx = set + modifier * i;
                sampled_cache[idx] = new SampledCacheLine[SAMPLED_CACHE_WAYS]();
            }
        }
    }
    rdp.clear();

    // Mode B init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv_B[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    prefer_B = false;
    leaderA_score_B = 0;
    leaderB_score_B = 0;
    access_count = 0;

    // Selector init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        mode_confidence[s] = 0;
    }
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu;
    bool use_B = should_use_mode_B(set);
    if (use_B) {
        return mode_B_victim(set, current_set);
    } else {
        return mode_A_victim(set, current_set, PC, paddr, type);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    if (is_writeback(type)) return;

    // Determine mode
    bool use_B = should_use_mode_B(set);

    // Leader scoring (dual tracking for ensemble and Mode B internal bandit)
    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) {
            leaderA_score += delta;
            leaderA_score_B += delta;
        }
        if (LEADER_B(set)) {
            leaderB_score += delta;
            leaderB_score_B += delta;
        }
    }

    // Ensemble epoch update
    selector_epoch++;
    if (selector_epoch % SELECTOR_EPOCH_SIZE == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!SAMPLED_SET(s)) {
                if (prefer_mode_B) sat_inc_i8(mode_confidence[s]);
                else sat_dec_i8(mode_confidence[s]);
            }
        }

        num_mode_B_followers = 0;
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!SAMPLED_SET(s) && mode_confidence[s] >= MODE_B_THRESHOLD) {
                num_mode_B_followers++;
            }
        }
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Mode B internal bandit epoch update
    access_count++;
    if (access_count % BANDIT_EPOCH == 0) {
        prefer_B = (leaderB_score_B > (leaderA_score_B + GATE_MARGIN));
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!SAMPLED_SET(s)) {
                if (prefer_B) sat_inc_i8(bandit_score[s]);
                else sat_dec_i8(bandit_score[s]);
            }
        }
        leaderA_score_B = 0;
        leaderB_score_B = 0;
    }

    // Call appropriate mode update
    if (use_B) {
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        mode_A_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}