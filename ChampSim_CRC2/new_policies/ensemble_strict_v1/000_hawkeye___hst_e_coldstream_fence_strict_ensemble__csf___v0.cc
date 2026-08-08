/*
 * Hawkeye + HST-E ColdStream-Fence (CSF)
 * STRICT DYNAMIC ENSEMBLE — packed RRIP, full Hawkeye behavior restored
 *
 * Notes:
 * - OPTgen/sampler and per-line PC signatures/prefetch bits retained for Hawkeye training
 * - SHCT implemented as fixed array (no map) to trim overhead; CRC hashing unchanged
 * - Packs Mode A/B RRIP (3 bits each) + Mode B touched bit into 1 byte/line to shave bytes
 * - Selector and Mode B PC/stream logic retained
 */

#include "../inc/champsim_crc2.h"
#include <map>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>

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

#define maxRRPV 7

// Packed RRIP state: bits0-2 = Mode A RRPV, bits3-5 = Mode B RRPV, bit6 = Mode B touched
static uint8_t packed_rrip[LLC_SETS][LLC_WAYS];
static inline uint8_t get_rrpv_A(uint32_t set, uint32_t way) { return packed_rrip[set][way] & 0x7u; }
static inline void set_rrpv_A(uint32_t set, uint32_t way, uint8_t v) {
    packed_rrip[set][way] = (uint8_t)((packed_rrip[set][way] & 0xF8u) | (v & 0x7u));
}
static inline uint8_t get_rrpv_B(uint32_t set, uint32_t way) { return (packed_rrip[set][way] >> 3) & 0x7u; }
static inline void set_rrpv_B(uint32_t set, uint32_t way, uint8_t v) {
    packed_rrip[set][way] = (uint8_t)((packed_rrip[set][way] & 0xC7u) | ((v & 0x7u) << 3));
}
static inline bool mode_B_touched(uint32_t set, uint32_t way) { return (packed_rrip[set][way] >> 6) & 0x1u; }
static inline void mode_B_set_touched(uint32_t set, uint32_t way, bool v) {
    if (v) packed_rrip[set][way] |= 0x40u;
    else packed_rrip[set][way] &= (uint8_t)(~0x40u);
}

// ============================================================================
// LEADER-FOLLOWER SAMPLING: match Hawkeye macro
// ============================================================================
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// SELECTOR (Very conservative gating + B-leader safety)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50; // huge bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;  // strict local threshold

// Per-set confidence: -8..+7 (stored as int8_t)
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // bitset for assigned Mode B follower sets
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// ============================================================================
// MODE A: HAWKEYE (EXCLUSIVE STATE)
// ============================================================================
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)

uint64_t CRC(uint64_t _blockAddress) {
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal = _blockAddress;
    for (unsigned int i = 0; i < 32; i++)
        _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
    return _returnVal;
}

class HAWKEYE_PC_PREDICTOR {
    uint8_t SHCT[SHCT_SIZE];
public:
    HAWKEYE_PC_PREDICTOR(){ for(uint32_t i=0;i<SHCT_SIZE;i++) SHCT[i]=(1+MAX_SHCT)/2; }
    void increment(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if (SHCT[signature] < MAX_SHCT) SHCT[signature]++;
    }
    void decrement(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        if (SHCT[signature] != 0) SHCT[signature]--;
    }
    bool get_prediction(uint64_t pc) {
        uint64_t signature = CRC(pc) % SHCT_SIZE;
        return (SHCT[signature] >= ((MAX_SHCT + 1) / 2));
    }
};

HAWKEYE_PC_PREDICTOR* demand_predictor;
HAWKEYE_PC_PREDICTOR* prefetch_predictor;

#define OPTGEN_VECTOR_SIZE 128
struct ADDR_INFO {
    uint64_t addr;
    uint32_t last_quanta;
    uint64_t PC;
    bool prefetched;
    uint32_t lru;
    void init(unsigned int curr_quanta) { last_quanta = 0; PC = 0; prefetched = false; lru = 0; }
    void update(unsigned int curr_quanta, uint64_t _pc, bool prediction) { last_quanta = curr_quanta; PC = _pc; }
    void mark_prefetch() { prefetched = true; }
};
struct OPTgen {
    std::vector<unsigned int> liveness_history;
    uint64_t num_cache, num_dont_cache, access, CACHE_SIZE;
    void init(uint64_t size) {
        num_cache = 0; num_dont_cache = 0; access = 0; CACHE_SIZE = size;
        liveness_history.resize(OPTGEN_VECTOR_SIZE, 0);
    }
    void add_access(uint64_t curr_quanta){ access++; liveness_history[curr_quanta] = 0; }
    void add_prefetch(uint64_t curr_quanta){ liveness_history[curr_quanta] = 0; }
    bool should_cache(uint64_t curr_quanta, uint64_t last_quanta) {
        bool is_cache = true; unsigned int i = last_quanta;
        while (i != curr_quanta) { if (liveness_history[i] >= CACHE_SIZE) { is_cache = false; break; } i = (i + 1) % liveness_history.size(); }
        if (is_cache) { i = last_quanta; while (i != curr_quanta) { liveness_history[i]++; i = (i + 1) % liveness_history.size(); } }
        if (is_cache) num_cache++; else num_dont_cache++; return is_cache;
    }
    uint64_t get_num_opt_hits() { return num_cache; }
};

OPTgen perset_optgen[LLC_SETS];
uint64_t perset_mytimer[LLC_SETS];
uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched[LLC_SETS][LLC_WAYS];
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

static void replace_addr_history_element(unsigned int sampler_set){
    uint64_t lru_addr = 0;
    for(auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); ++it){
        if((it->second).lru == (SAMPLER_WAYS-1)){ lru_addr = it->first; break; }
    }
    addr_history[sampler_set].erase(lru_addr);
}
static void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru){
    for(auto it=addr_history[sampler_set].begin(); it!=addr_history[sampler_set].end(); ++it)
        if((it->second).lru < curr_lru) (it->second).lru++;
}

static uint32_t hawkeye_get_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t i=0; i<LLC_WAYS; i++) if(get_rrpv_A(set, i) == maxRRPV) return i;
    uint32_t max_rrip = 0; int32_t lru_victim = -1;
    for(uint32_t i=0; i<LLC_WAYS; i++){ if(get_rrpv_A(set, i) >= max_rrip){ max_rrip = get_rrpv_A(set, i); lru_victim = i; } }
    if(SAMPLED_SET(set) && lru_victim != -1){
        if(prefetched[set][lru_victim]) prefetch_predictor->decrement(signatures[set][lru_victim]);
        else demand_predictor->decrement(signatures[set][lru_victim]);
    }
    return (lru_victim==-1)?0u:(uint32_t)lru_victim;
}

static void hawkeye_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    paddr = (paddr >> 6) << 6;
    if(type == ACCESS_PREFETCH){ if(!hit) prefetched[set][way] = true; }
    else prefetched[set][way] = false;
    if(type == ACCESS_WRITEBACK) return;

    if(SAMPLED_SET(set)){
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;
        auto it = addr_history[sampler_set].find(sampler_tag);

        if(it != addr_history[sampler_set].end() && type != ACCESS_PREFETCH){
            unsigned int curr_timer = perset_mytimer[set];
            if(curr_timer < it->second.last_quanta) curr_timer = curr_timer + 1024;
            bool wrap = ((curr_timer - it->second.last_quanta) > OPTGEN_VECTOR_SIZE);
            uint64_t last_quanta = it->second.last_quanta % OPTGEN_VECTOR_SIZE;
            if(!wrap && perset_optgen[set].should_cache(curr_quanta, last_quanta)){
                if(it->second.prefetched) prefetch_predictor->increment(it->second.PC);
                else demand_predictor->increment(it->second.PC);
            } else {
                if(it->second.prefetched) prefetch_predictor->decrement(it->second.PC);
                else demand_predictor->decrement(it->second.PC);
            }
            perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, it->second.lru);
            it->second.prefetched = false;
        } else if(it == addr_history[sampler_set].end()){
            if(addr_history[sampler_set].size() == SAMPLER_WAYS) replace_addr_history_element(sampler_set);
            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            if(type == ACCESS_PREFETCH) { addr_history[sampler_set][sampler_tag].mark_prefetch(); perset_optgen[set].add_prefetch(curr_quanta); }
            else { perset_optgen[set].add_access(curr_quanta); }
            update_addr_history_lru(sampler_set, SAMPLER_WAYS-1);
        } else { // prefetch seen
            uint64_t last_quanta = it->second.last_quanta % OPTGEN_VECTOR_SIZE;
            if(perset_mytimer[set] - it->second.last_quanta < 5*NUM_CORE){
                if(perset_optgen[set].should_cache(curr_quanta, last_quanta)){
                    if(it->second.prefetched) prefetch_predictor->increment(it->second.PC);
                    else demand_predictor->increment(it->second.PC);
                }
            }
            it->second.mark_prefetch();
            perset_optgen[set].add_prefetch(curr_quanta);
            update_addr_history_lru(sampler_set, it->second.lru);
        }

        bool new_prediction = demand_predictor->get_prediction(PC);
        if(type == ACCESS_PREFETCH) new_prediction = prefetch_predictor->get_prediction(PC);
        addr_history[sampler_set][sampler_tag].update(perset_mytimer[set], PC, new_prediction);
        addr_history[sampler_set][sampler_tag].lru = 0;
        perset_mytimer[set] = (perset_mytimer[set] + 1) % 1024;
    }

    bool friendly = demand_predictor->get_prediction(PC);
    if(type == ACCESS_PREFETCH) friendly = prefetch_predictor->get_prediction(PC);
    signatures[set][way] = PC;

    if(!friendly){
        set_rrpv_A(set, way, maxRRPV);
    } else {
        set_rrpv_A(set, way, 0);
        if(!hit){
            bool saturated = false;
            for(uint32_t i=0; i<LLC_WAYS; i++) if(get_rrpv_A(set, i) == maxRRPV-1) saturated = true;
            for(uint32_t i=0; i<LLC_WAYS; i++){
                uint8_t age = get_rrpv_A(set, i);
                if(!saturated && age < maxRRPV-1) set_rrpv_A(set, i, (uint8_t)(age + 1));
            }
        }
        set_rrpv_A(set, way, 0);
    }
}

// ============================================================================
// MODE B: HST-E ColdStream-Fence (EXCLUSIVE STATE)
// ============================================================================
// Tiny PC table (128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit dead score
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];    // 8-bit unique-set mask
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];     // 4-bit (0-15)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];   // 2-bit (0-3)

// Tunables (fixed per HST-E spec)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // LRU for streams
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >=6
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >=2
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // >=4 unique sets

// Helpers
static inline uint32_t mode_B_pc_index(uint64_t pc) { return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE - 1); }
static inline uint16_t mode_B_line_id(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit
static inline void sat_inc_u2(uint8_t &x){ if(x < 3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x < 15) x++; }
static inline uint8_t popcount8(uint8_t x){ x = x - ((x>>1)&0x55); x = (x&0x33)+((x>>2)&0x33); return ((x + (x>>4)) & 0x0F); }

// Detect stride/stream with run tracking
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
            sat_inc_u4(mode_B_pc_run_len[pc_idx]);
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    } else {
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        // setmask kept to allow quick long-run recognition on continued streams
    }

    mode_B_pc_last_line[pc_idx] = line_id;
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
    // ONLY bypass if: long stream AND cold AND dead
    if (is_stream && long_run && !pc_hot && pc_dead) return true;
    return false;
}

// Mode B update (updates ONLY rrpv_B and Mode B metadata)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU frequency
    if (is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead block tracking + hit streak
    if (hit) {
        if (mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if (mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    } else {
        if (is_demand(type) && mode_B_pc_hitstreak[pc_idx] == 0) sat_inc_u2(mode_B_pc_dead[pc_idx]);
        if (mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion
    if (is_prefetch(type)) {
        set_rrpv_B(set, way, MODE_B_STREAM_TAIL);
    } else if (is_stream && long_run) {
        set_rrpv_B(set, way, MODE_B_STREAM_TAIL);
    } else if (pc_dead) {
        set_rrpv_B(set, way, MODE_B_STREAM_TAIL);
    } else if (pc_hot) {
        set_rrpv_B(set, way, MODE_B_INSERT_WARM);
    } else {
        set_rrpv_B(set, way, MODE_B_INSERT_COLD);
    }

    // Multi-hit rescue with stream awareness
    if (hit) {
        if (mode_B_touched(set, way)) {
            if (!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2) {
                set_rrpv_B(set, way, 0); // MRU promote
            }
        } else {
            mode_B_set_touched(set, way, true);
        }
    } else {
        mode_B_set_touched(set, way, false);
    }
}

// Mode B victim (uses ONLY rrpv_B)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (!current_set[w].valid) return w;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (get_rrpv_B(set, w) == maxRRPV) return w;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint8_t age = get_rrpv_B(set, w);
        if (age < maxRRPV) set_rrpv_B(set, w, (uint8_t)(age + 1));
    }
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (get_rrpv_B(set, w) == maxRRPV) return w;
    return 0;
}

// ============================================================================
// SELECTOR functions
// ============================================================================
static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // drift followers toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 7) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -8) mode_confidence[s]--; }
        }
        // epoch decay for Mode B PC stats
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i] >>= 1;
            mode_B_pc_dead[i] >>= 1;
            mode_B_pc_run_len[i] >>= 1;
            mode_B_pc_setmask8[i] >>= 1;
            mode_B_pc_hitstreak[i] >>= 1;
        }
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    // B Leader Safety: only switch if B is globally preferred
    if(LEADER_B(set)) return prefer_mode_B;
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set); num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// ChampSim Interface
// ============================================================================
void InitReplacementState() {
    // Clear packed RRIP and predictors
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            set_rrpv_A(s, w, maxRRPV);
            set_rrpv_B(s, w, maxRRPV);
            mode_B_set_touched(s, w, false);
            prefetched[s][w] = false;
            signatures[s][w] = 0;
        }
        perset_mytimer[s] = 0;
    }
    memset(mode_confidence, 0, sizeof(mode_confidence));
    memset(follower_bits, 0, sizeof(follower_bits));

    // OPTgen init (Hawkeye)
    for (uint32_t i=0; i<LLC_SETS; i++) perset_optgen[i].init(LLC_WAYS-2);
    addr_history.resize(SAMPLER_SETS);
    for (int i=0; i<SAMPLER_SETS; i++) addr_history[i].clear();

    // Predictors
    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // Mode B PC table init
    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_setmask8[i] = 0;
        mode_B_pc_run_len[i] = 0;
        mode_B_pc_hitstreak[i] = 0;
    }

    // Selector init
    leaderA_score = 0; leaderB_score = 0; prefer_mode_B = false; selector_epoch = 0; num_mode_B_followers = 0;

    std::cout << "Initialize Hawkeye + HST-E CSF ensemble" << std::endl;
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set,
                         uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_B = should_use_mode_B(set);
    if(use_B){
        // Optional bypass under strict conditions
        if(mode_B_should_bypass(set, PC, paddr, type)) return LLC_WAYS; // bypass
        return mode_B_victim(set, current_set);
    } else {
        return hawkeye_get_victim(set, current_set);
    }
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way,
                             uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                             uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Update selector (leaders)
    update_selector(set, hit);

    bool use_B = should_use_mode_B(set);
    if(use_B){
        // Update ONLY Mode B state
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Update ONLY Mode A (Hawkeye) state
        hawkeye_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats_Heartbeat() {}

void PrintStats() {}
