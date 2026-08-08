/*
 * Hawkeye + HST-E TightQuarantine (HST-E TQ)
 * STRICT DYNAMIC ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Selector (very conservative):
 *   - Global bias toward Hawkeye: MODE_A_BIAS = 50
 *   - Per-set confidence threshold: MODE_B_THRESHOLD = 7
 *   - B leader safety: LEADER_B uses Mode B only if prefer_mode_B==true
 *   - Follower cap: MAX_MODE_B_FOLLOWERS = 256 (~12.5% of sets)
 *
 * Mode B (HST-E derived, required components):
 *   - Bidirectional ±1/±2 stride/run detector; arm after 2 consecutive steps
 *   - 8-bit unique-set mask to detect long runs (>=4 unique sets)
 *   - TinyLFU per-PC (4-bit) with epochic decay; HOT>=6 inserts MRU (RRIP=0), else mid-age (RRIP=3)
 *   - Dead PC (2-bit) hard-tail insert (RRIP=7)
 *   - Stream quarantine: long stream or prefetch -> tail (RRIP=7)
 *   - Multi-hit rescue: promote to MRU on 2nd hit if (non-stream) or (stream with PC hitstreak>=2)
 *   - Conservative bypass: ONLY if (is_stream && long_run && cold && dead)
 *
 * Tunables (kept within allowed ranges; telemetry knobs):
 *   - MODE_B_HOT_THRESHOLD = 6   // [5..8]
 *   - MODE_B_LONGRUN_UNIQUESETS = 4 // [3..6]
 *   - MODE_B_INSERT_WARM = 0     // [0..2]
 *   - MODE_B_INSERT_COLD = 3     // [2..4]
 *
 * IMPORTANT:
 *   - EXCLUSIVE updates: if use_mode_B -> update ONLY rrpv_B; else ONLY rrpv_A.
 *   - Never touch the other mode’s RRIP array or metadata.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <map>
#include <vector>
#include <cstring>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11
#define maxRRPV 7

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// Sample 64 sets: low 6 bits == high 6 bits
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// Simple 64-bit mix (CRC-like) for hashing
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// Hawkeye CRC (from reference)
static inline uint64_t CRC(uint64_t _blockAddress) {
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal = _blockAddress;
    for (unsigned int i = 0; i < 32; i++)
        _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
    return _returnVal;
}

// ============================================================================
// Selector (very conservative)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // strong Hawkeye bias
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict local per-set threshold

// Per-set confidence: -8..+7 (store as int8_t; counted logically as 4-bit)
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Drift confidence toward global preference each epoch
static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 7) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -8) mode_confidence[s]--; }
        }
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    // B LEADER SAFETY: only use B if globally preferred
    if(LEADER_B(set)) return prefer_mode_B;

    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    // Enforce follower cap lazily
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// Mode A: Hawkeye (exclusive rrpv_A + OPTgen/SHCT sampling)
// ============================================================================
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

// Hawkeye predictors (2K entries, 5-bit each)
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
class HAWKEYE_PC_PREDICTOR {
    std::map<uint64_t, short unsigned int> SHCT;
public:
    void increment(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(sig) == SHCT.end()) SHCT[sig] = (1 + MAX_SHCT) / 2;
        SHCT[sig] = (SHCT[sig] < MAX_SHCT) ? (SHCT[sig] + 1) : MAX_SHCT;
    }
    void decrement(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(sig) == SHCT.end()) SHCT[sig] = (1 + MAX_SHCT) / 2;
        if (SHCT[sig] != 0) SHCT[sig] = SHCT[sig] - 1;
    }
    bool get_prediction(uint64_t pc) {
        uint64_t sig = CRC(pc) % SHCT_SIZE;
        if (SHCT.find(sig) != SHCT.end() && SHCT[sig] < ((MAX_SHCT + 1) / 2)) return false;
        return true;
    }
};
static HAWKEYE_PC_PREDICTOR* demand_predictor;
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;

// OPTgen sampling (per-set, only 64 sampled sets are used at runtime)
#define OPTGEN_VECTOR_SIZE 128
struct ADDR_INFO {
    uint64_t addr;
    uint32_t last_quanta;
    uint64_t PC;
    bool prefetched;
    uint32_t lru;
    void init(unsigned int curr_quanta){ last_quanta=0; PC=0; prefetched=false; lru=0; }
    void update(unsigned int curr_quanta, uint64_t _pc, bool prediction){ last_quanta=curr_quanta; PC=_pc; }
    void mark_prefetch(){ prefetched=true; }
};
struct OPTgen {
    std::vector<unsigned int> liveness_history;
    uint64_t num_cache=0, num_dont_cache=0, access=0, CACHE_SIZE=0;
    void init(uint64_t size){ CACHE_SIZE=size; liveness_history.assign(OPTGEN_VECTOR_SIZE,0); }
    void add_access(uint64_t curr_quanta){ access++; liveness_history[curr_quanta]=0; }
    bool should_cache(uint64_t curr_quanta, uint64_t last_quanta){
        bool is_cache=true; unsigned int i=last_quanta;
        while(i!=curr_quanta){ if(liveness_history[i] >= CACHE_SIZE){ is_cache=false; break; } i=(i+1)%liveness_history.size(); }
        if(is_cache){ i=last_quanta; while(i!=curr_quanta){ liveness_history[i]++; i=(i+1)%liveness_history.size(); } }
        if(is_cache) num_cache++; else num_dont_cache++; return is_cache;
    }
};
static uint64_t perset_mytimer[LLC_SETS];
static OPTgen perset_optgen[LLC_SETS];

// Sampler to track 8x cache history for sampled sets
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE/SAMPLER_WAYS)
static std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

static inline void replace_addr_history_element(unsigned int sampler_set){
    uint64_t lru_addr = 0;
    for(auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); it++){
        if((it->second).lru == (SAMPLER_WAYS-1)){ lru_addr = it->first; break; }
    }
    addr_history[sampler_set].erase(lru_addr);
}
static inline void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru){
    for(auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); it++){
        if((it->second).lru < curr_lru){ (it->second).lru++; }
    }
}

// ============================================================================
// Mode B: HST-E TightQuarantine (exclusive rrpv_B)
// ============================================================================
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];             // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS]; // 1-bit first-hit flag

// PC table (compact, 128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];    // 10-bit
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE];  // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];         // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];         // 2-bit dead score
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];    // last set low-8
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];     // unique-set mask
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];      // 4-bit (0-15)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];    // 2-bit (0-3)

// Tunables (within allowed ranges)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // LRU for streams
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >=6
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >=2
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // >=4 unique sets

// Helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline uint8_t popcount8(uint8_t x){ x = x - ((x>>1) & 0x55); x = (x & 0x33) + ((x>>2) & 0x33); return ((x + (x>>4)) & 0x0F); }
static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); }

static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set) {
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];

    bool step = false;
    if (last != 0xFFFF) {
        // Bidirectional ±1/±2
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }

    if (step) {
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) {
            // Grow run length and unique-set mask
            sat_inc_u4(mode_B_pc_run_len[pc_idx]);
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    } else {
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        mode_B_pc_setmask8[pc_idx] = 0;
    }

    mode_B_pc_last_line[pc_idx] = line_id;
    mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 0xFFu);
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx) {
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type) {
    if (is_prefetch(type)) return false; // quarantine instead
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // ONLY bypass if long stream AND cold AND dead
    if (is_stream && long_run && !pc_hot && pc_dead)
        return true;

    return false;
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU frequency update
    if (is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead block tracking and hit streak
    if (hit) {
        if (mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if (mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    } else {
        if (is_demand(type) && mode_B_pc_hitstreak[pc_idx] == 0) sat_inc_u2(mode_B_pc_dead[pc_idx]);
        if (mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection (updates stride/run state)
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion policy (quarantine/bias)
    if (is_prefetch(type)) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (is_stream && long_run) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (pc_dead) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (pc_hot) {
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    } else {
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue with stream awareness
    if (hit) {
        if (mode_B_line_touched[set][way]) {
            if (!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2) {
                rrpv_B[set][way] = 0; // promote to MRU
            }
        } else {
            mode_B_line_touched[set][way] = 1;
        }
    } else {
        mode_B_line_touched[set][way] = 0;
    }
}

static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (!current_set[w].valid) return w;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    return 0;
}

// ============================================================================
// ChampSim hooks
// ============================================================================
void InitReplacementState() {
    // Initialize RRIPs and metadata
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        for (uint32_t j = 0; j < LLC_WAYS; j++) {
            rrpv_A[i][j] = maxRRPV;
            rrpv_B[i][j] = maxRRPV;
            mode_B_line_touched[i][j] = 0;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS - 2);
        mode_confidence[i] = 0;
    }
    std::memset(follower_bits, 0, sizeof(follower_bits));
    num_mode_B_followers = 0;

    addr_history.resize(SAMPLER_SETS);
    for (int i = 0; i < SAMPLER_SETS; i++) addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) {
        mode_B_pc_last_line[i]   = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i]        = 0;
        mode_B_pc_dead[i]        = 0;
        mode_B_pc_last_set8[i]   = 0;
        mode_B_pc_setmask8[i]    = 0;
        mode_B_pc_run_len[i]     = 0;
        mode_B_pc_hitstreak[i]   = 0;
    }

    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false; selector_epoch = 0;
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu;
    bool use_B = should_use_mode_B(set);

    if (use_B) {
        if (mode_B_should_bypass(set, PC, paddr, type))
            return LLC_WAYS; // bypass
        return mode_B_victim(set, current_set);
    }

    // Mode A: Hawkeye victim (RRIP_A only)
    for (uint32_t i=0; i<LLC_WAYS; i++)
        if (rrpv_A[set][i] == maxRRPV) return i;

    uint32_t max_rrip = 0; int32_t lru_victim = -1;
    for (uint32_t i=0; i<LLC_WAYS; i++){
        if (rrpv_A[set][i] >= max_rrip){ max_rrip = rrpv_A[set][i]; lru_victim = i; }
    }
    return (uint32_t)lru_victim;
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;

    // Update selector first
    update_selector(set, hit);

    // Hawkeye sampling on sampled sets (train OPTgen + predictors)
    if(SAMPLED_SET(set)){
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        auto it = addr_history[sampler_set].find(sampler_tag);
        if (it != addr_history[sampler_set].end()) {
            uint64_t last_quanta = it->second.last_quanta;
            bool is_cache_friendly = perset_optgen[set].should_cache(curr_quanta, last_quanta);
            if (it->second.prefetched) {
                if (is_cache_friendly) prefetch_predictor->increment(it->second.PC);
                else                   prefetch_predictor->decrement(it->second.PC);
            } else {
                if (is_cache_friendly) demand_predictor->increment(it->second.PC);
                else                   demand_predictor->decrement(it->second.PC);
            }
            update_addr_history_lru(sampler_set, it->second.lru);
            it->second.update(curr_quanta, PC, is_cache_friendly);
            it->second.lru = 0;
        } else {
            if (addr_history[sampler_set].size() == SAMPLER_WAYS) replace_addr_history_element(sampler_set);
            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            addr_history[sampler_set][sampler_tag].update(curr_quanta, PC, false);
            addr_history[sampler_set][sampler_tag].lru = 0;
            if (is_prefetch(type)) addr_history[sampler_set][sampler_tag].mark_prefetch();
            update_addr_history_lru(sampler_set, 0);
        }
        perset_optgen[set].add_access(curr_quanta);
        perset_mytimer[set] = (perset_mytimer[set] + 1);
    }

    bool use_B = should_use_mode_B(set);

    // EXCLUSIVE updates: only the chosen mode's RRIP is changed
    if (use_B) {
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A insertion policy (RRIP_A only), guided by Hawkeye predictors
        bool friendly = demand_predictor->get_prediction(PC);
        if (is_prefetch(type)) friendly = prefetch_predictor->get_prediction(PC);

        if (!friendly) {
            rrpv_A[set][way] = maxRRPV; // cache-averse -> tail
        } else {
            // cache-friendly -> MRU; light aging of others on miss
            rrpv_A[set][way] = 0;
            if (!hit) {
                bool saturated = false;
                for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_A[set][i] == maxRRPV-1) saturated = true;
                if (!saturated) {
                    for (uint32_t i = 0; i < LLC_WAYS; i++) if (rrpv_A[set][i] < maxRRPV-1) rrpv_A[set][i]++;
                }
                rrpv_A[set][way] = 0;
            }
        }
    }

    // Epochic decay for Mode B PC stats (tied to selector epoch boundary)
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i]      >>= 1;
            mode_B_pc_dead[i]      >>= 1;
            mode_B_pc_run_len[i]   >>= 1;
            mode_B_pc_setmask8[i]  >>= 1;
            mode_B_pc_hitstreak[i] >>= 1;
        }
    }
}

void PrintStats_Heartbeat(){}

void PrintStats(){}