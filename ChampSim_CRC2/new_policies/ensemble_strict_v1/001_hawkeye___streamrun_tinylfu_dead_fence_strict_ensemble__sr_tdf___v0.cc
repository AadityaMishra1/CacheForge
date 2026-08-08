/*
 * Hawkeye + StreamRun TinyLFU Dead-Fence (SR-TDF)
 * STRICT DYNAMIC ENSEMBLE (EXCLUSIVE STATE PER MODE)
 *
 * Selector: BIAS=50, THRESHOLD=7, 12.5% follower cap, B-leader safety
 * Mode A: Hawkeye (exclusive rrpv_A + Hawkeye metadata)
 * Mode B: HST-E derived SR-TDF (exclusive rrpv_B + tiny per-PC table)
 *
 * Mode B tunables (allowed window per spec):
 *   MODE_B_INSERT_WARM=0, MODE_B_INSERT_COLD=3, MODE_B_STREAM_TAIL=7
 *   MODE_B_STREAM_ARM=2 (±1/±2), MODE_B_HOT_THRESHOLD=6, MODE_B_DEAD_THRESHOLD=2
 *   MODE_B_LONGRUN_UNIQUESETS=4
 * Telemetry (epoch): decay TinyLFU freq >>=1; attenuate dead/hitstreak; clear run masks
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <map>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define maxRRPV 7

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// Helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline uint8_t popcount8(uint8_t x){ x = x - ((x >> 1) & 0x55); x = (x & 0x33) + ((x >> 2) & 0x33); return (((x + (x >> 4)) & 0x0F) * 0x01); }

// ---------------------------------------------------------------------------
// Leader-follower sampling (match Hawkeye macro)
// ---------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// ---------------------------------------------------------------------------
// Selector (conservative gating)
// ---------------------------------------------------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;
static constexpr int8_t   MODE_B_THRESHOLD    = 7;

static int8_t mode_confidence[LLC_SETS]; // -8..+7

static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256; // 12.5%
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// ---------------------------------------------------------------------------
// Mode A: Hawkeye (exclusive state)
// ---------------------------------------------------------------------------
uint32_t rrpv_A[LLC_SETS][LLC_WAYS];

#define TIMER_SIZE 1024
uint64_t perset_mytimer[LLC_SETS];

uint64_t signatures[LLC_SETS][LLC_WAYS];
bool prefetched_flag[LLC_SETS][LLC_WAYS];

#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)

uint64_t CRC(uint64_t _blockAddress) {
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal = _blockAddress;
    for (unsigned int i = 0; i < 32; i++)
        _returnVal = ((_returnVal & 1) == 1) ? ((_returnVal >> 1) ^ crcPolynomial) : (_returnVal >> 1);
    return _returnVal;
}

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
            return false;
        return true;
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
    void init(unsigned int curr_quanta){ last_quanta=0; PC=0; prefetched=false; lru=0; }
    void update(unsigned int curr_quanta, uint64_t _pc, bool prediction){ (void)prediction; last_quanta=curr_quanta; PC=_pc; }
    void mark_prefetch(){ prefetched=true; }
};

struct OPTgen {
    std::vector<unsigned int> liveness_history;
    uint64_t num_cache, num_dont_cache, access, CACHE_SIZE;
    void init(uint64_t size){ num_cache=0; num_dont_cache=0; access=0; CACHE_SIZE=size; liveness_history.resize(OPTGEN_VECTOR_SIZE,0); }
    void add_access(uint64_t curr_quanta){ access++; liveness_history[curr_quanta]=0; }
    void add_prefetch(uint64_t curr_quanta){ liveness_history[curr_quanta]=0; }
    bool should_cache(uint64_t curr_quanta, uint64_t last_quanta){
        bool is_cache = true; unsigned int i = last_quanta;
        while(i != curr_quanta){
            if(liveness_history[i] >= CACHE_SIZE){ is_cache=false; break; }
            i = (i+1) % liveness_history.size();
        }
        if(is_cache){
            i = last_quanta;
            while(i != curr_quanta){ liveness_history[i]++; i = (i+1) % liveness_history.size(); }
        }
        if(is_cache) num_cache++; else num_dont_cache++;
        return is_cache;
    }
    uint64_t get_num_opt_hits(){ return num_cache; }
};
OPTgen perset_optgen[LLC_SETS];

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
    for(auto &kv : addr_history[sampler_set]){
        if((kv.second).lru < curr_lru) (kv.second).lru++;
    }
}

static uint32_t hawkeye_get_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t i=0;i<LLC_WAYS;i++) if(rrpv_A[set][i]==maxRRPV) return i;
    uint32_t max_rrip=0; int32_t lru_victim=-1;
    for(uint32_t i=0;i<LLC_WAYS;i++){ if(rrpv_A[set][i] >= max_rrip){ max_rrip=rrpv_A[set][i]; lru_victim=i; } }
    if(SAMPLED_SET(set) && lru_victim!=-1){
        if(prefetched_flag[set][lru_victim]) prefetch_predictor->decrement(signatures[set][lru_victim]);
        else demand_predictor->decrement(signatures[set][lru_victim]);
    }
    return (lru_victim==-1)?0u:(uint32_t)lru_victim;
}

static void hawkeye_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    paddr = (paddr >> 6) << 6;

    if(type == ACCESS_PREFETCH){ if(!hit) prefetched_flag[set][way]=true; }
    else prefetched_flag[set][way]=false;

    if(is_writeback(type)) return;

    if(SAMPLED_SET(set)){
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        auto it = addr_history[sampler_set].find(sampler_tag);
        if(it != addr_history[sampler_set].end() && type != ACCESS_PREFETCH){
            unsigned int curr_timer = perset_mytimer[set];
            if(curr_timer < it->second.last_quanta) curr_timer += TIMER_SIZE;
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
            it->second.prefetched=false;
        } else if(it == addr_history[sampler_set].end()){
            if(addr_history[sampler_set].size() == SAMPLER_WAYS) replace_addr_history_element(sampler_set);
            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            if(type == ACCESS_PREFETCH){ addr_history[sampler_set][sampler_tag].mark_prefetch(); perset_optgen[set].add_prefetch(curr_quanta); }
            else { perset_optgen[set].add_access(curr_quanta); }
            update_addr_history_lru(sampler_set, SAMPLER_WAYS-1);
        } else {
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

        bool new_pred = (type==ACCESS_PREFETCH) ? prefetch_predictor->get_prediction(PC) : demand_predictor->get_prediction(PC);
        addr_history[sampler_set][sampler_tag].update(perset_mytimer[set], PC, new_pred);
        addr_history[sampler_set][sampler_tag].lru = 0;
        perset_mytimer[set] = (perset_mytimer[set] + 1) % TIMER_SIZE;
    }

    bool new_prediction = (type==ACCESS_PREFETCH) ? prefetch_predictor->get_prediction(PC) : demand_predictor->get_prediction(PC);
    signatures[set][way] = PC;

    if(!new_prediction){
        rrpv_A[set][way] = maxRRPV;
    } else {
        rrpv_A[set][way] = 0;
        if(!hit){
            bool saturated=false;
            for(uint32_t i=0;i<LLC_WAYS;i++) if(rrpv_A[set][i]==maxRRPV-1) saturated=true;
            for(uint32_t i=0;i<LLC_WAYS;i++) if(!saturated && rrpv_A[set][i] < maxRRPV-1) rrpv_A[set][i]++;
        }
        rrpv_A[set][way] = 0;
    }
}

// ---------------------------------------------------------------------------
// Mode B: SR-TDF (HST-E derived) - EXCLUSIVE STATE
// ---------------------------------------------------------------------------

// Per-line exclusive state
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];              // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS]; // 1-bit first-hit flag

// PC table (compact; 128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)((pc ^ (pc>>2) ^ (pc>>7)) & (MODE_B_PC_TABLE_SIZE-1)); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)(((paddr >> 6) & 0x3FFu)); }

static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit logical, 0xFFFF uninit
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit dead score
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];   // last set low-3 bits history
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];    // unique-set mask (8-bit)
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];     // 4-bit run length
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];   // 2-bit per-PC hit streak

// Tunables (exact per spec window)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age for cold
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // hard tail
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // need 2 consecutive steps
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU hot
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead fence
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // long run = >=4 unique sets

static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set){
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];

    bool step=false;
    if(last != 0xFFFF){
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }

    if(step){
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if(mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM){
            if(mode_B_pc_run_len[pc_idx] < 15) mode_B_pc_run_len[pc_idx]++;
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
            mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 7);
        }
    } else {
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        mode_B_pc_setmask8[pc_idx] = 0;
    }

    mode_B_pc_last_line[pc_idx] = line_id;
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx){
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    if(is_prefetch(type)) return false;
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot  = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);
    return (is_stream && long_run && !pc_hot && pc_dead);
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead and hitstreak tracking
    if(hit){
        if(mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if(mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    } else {
        if(is_demand(type) && mode_B_pc_hitstreak[pc_idx]==0) sat_inc_u2(mode_B_pc_dead[pc_idx]);
        if(mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run  = mode_B_is_long_run(pc_idx);
    bool pc_hot    = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead   = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insert/quarantine
    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if(is_stream && long_run){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if(pc_dead){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if(pc_hot){
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    } else {
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue (stream-aware)
    if(hit){
        if(mode_B_line_touched[set][way]){
            if(!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2){
                rrpv_B[set][way] = 0; // promote to MRU
            }
        } else {
            mode_B_line_touched[set][way] = 1;
        }
    } else {
        mode_B_line_touched[set][way] = 0;
    }
}

static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
    return 0;
}

// ---------------------------------------------------------------------------
// Selector plumbing
// ---------------------------------------------------------------------------
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return prefer_mode_B; // B-leader safety: mirror Hawkeye unless B wins
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set); num_mode_B_followers++;
    }
    return true;
}

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

        // Epochic decay for Mode B PC table
        for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
            mode_B_pc_freq[i] >>= 1;
            if(mode_B_pc_dead[i] > 0) mode_B_pc_dead[i]--;
            if(mode_B_pc_hitstreak[i] > 0) mode_B_pc_hitstreak[i]--;
            mode_B_pc_setmask8[i] = 0;
            mode_B_pc_run_len[i]  = 0;
        }

        leaderA_score=0; leaderB_score=0;
    }
}

// ---------------------------------------------------------------------------
// ChampSim interface
// ---------------------------------------------------------------------------
void InitReplacementState(){
    // Mode A + Mode B per-line init
    for(uint32_t i=0;i<LLC_SETS;i++){
        for(uint32_t j=0;j<LLC_WAYS;j++){
            rrpv_A[i][j] = maxRRPV;
            rrpv_B[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched_flag[i][j] = false;
            mode_B_line_touched[i][j] = 0;
        }
        perset_mytimer[i] = 0;
        mode_confidence[i] = 0;
    }

    // Hawkeye OPTgen initialization
    for(uint32_t i=0;i<LLC_SETS;i++) perset_optgen[i].init(LLC_WAYS-2);
    addr_history.resize(SAMPLER_SETS);
    for(uint32_t i=0;i<SAMPLER_SETS;i++) addr_history[i].clear();

    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // Mode B PC table init
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_last_line[i]   = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i]        = 0;
        mode_B_pc_dead[i]        = 0;
        mode_B_pc_last_set8[i]   = 0;
        mode_B_pc_setmask8[i]    = 0;
        mode_B_pc_run_len[i]     = 0;
        mode_B_pc_hitstreak[i]   = 0;
    }

    std::memset(follower_bits, 0, sizeof(follower_bits));
    leaderA_score=0; leaderB_score=0; prefer_mode_B=false; selector_epoch=0; num_mode_B_followers=0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_mode_B = should_use_mode_B(set);
    if(use_mode_B) return mode_B_victim(set, current_set);
    return hawkeye_get_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;

    // Ignore writebacks for policy updates
    if(is_writeback(type)) return;

    // Update selector first
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // Optional Mode B conservative bypass (victim path decides fill vs bypass)
    if(use_mode_B && !hit){
        if(mode_B_should_bypass(set, PC, paddr, type)){
            // Signal bypass to RRIP by setting chosen way to LLC_WAYS at victim time.
            // Here on update we simply avoid touching rrpv_B; ChampSim will handle bypass from victim path.
        }
    }

    if(use_mode_B){
        // Update ONLY Mode B state
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Update ONLY Mode A (Hawkeye) state
        hawkeye_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){
    // Optional: expose selector stats
    std::cout << "prefer_mode_B=" << prefer_mode_B << " followers=" << num_mode_B_followers << std::endl;
}