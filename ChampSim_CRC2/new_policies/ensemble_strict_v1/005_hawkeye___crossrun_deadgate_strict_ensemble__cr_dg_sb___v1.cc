/*
 * Hawkeye + CrossRun-DeadGate (CR-DG-SB)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Targeted refinements:
 * - Forward-only +1/+2 run detector (arm after 2).
 * - Unique-set run tracking per PC; long runs -> hard-tail and soft-bypass.
 * - Prefetches always quarantined deep.
 * - TinyLFU + 2-bit dead-gate with epochic decay to throttle noisy PCs.
 * - Stream-aware rescue: non-stream promotes on 2 hits, stream needs 3 hits.
 * - Conservative gating: high BIAS and per-set THRESHOLD; defaults to Hawkeye.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>
#include <iostream>

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

//------------------------------- Sampling -------------------------------
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

//------------------------------- Selector (conservative gating) -------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 33;  // strong bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 5;   // local per-set threshold

// Pack per-set confidence in 4-bit signed nibbles (-8..+7) stored with +8 bias
static uint8_t mode_conf_nibbles[LLC_SETS/2]; // 2048 sets -> 1024 bytes
static inline int8_t get_mode_conf(uint32_t set){
    uint8_t b = mode_conf_nibbles[set>>1];
    uint8_t nib = (set & 1) ? (b>>4) : (b & 0xF);
    return (int8_t)nib - 8;
}
static inline void set_mode_conf(uint32_t set, int8_t val){
    if(val > 7) val = 7;
    if(val < -8) val = -8;
    uint8_t enc = (uint8_t)(val + 8) & 0xF;
    uint8_t &b = mode_conf_nibbles[set>>1];
    if(set & 1) { b = (uint8_t)((b & 0x0F) | (enc<<4)); }
    else        { b = (uint8_t)((b & 0xF0) | enc); }
}

// Forward declaration of Mode B epoch-decay hook
static void mode_B_epoch_decay();

// Update selector and also trigger Mode B epochic decay
static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // drift local confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            int8_t c = get_mode_conf(s);
            if(prefer_mode_B){ if(c < 7) c++; }
            else             { if(c > -8) c--; }
            set_mode_conf(s, c);
        }
        leaderA_score = 0; leaderB_score = 0;

        // Decay Mode B per-PC telemetry each epoch
        mode_B_epoch_decay();
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;
    if(!prefer_mode_B) return false;
    if(get_mode_conf(set) < MODE_B_THRESHOLD) return false;
    return true;
}

//------------------------------- Mode A: Hawkeye-lite (EXCLUSIVE) -------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // logically 3-bit

// Tiny SHCT (single table, 512 entries, 5-bit counters)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31 stored in 5 bits
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

// Minimal Hawkeye-ish insertion using SHCT
static inline void hawkeye_update(uint32_t set, uint32_t way, uint64_t PC, uint8_t hit){
    hawkeye_train(PC, hit);
    if(!hawkeye_is_friendly(PC)){
        rrpv_A[set][way] = maxRRPV; // averse -> tail
    }else{
        // friendly -> near-MRU with light aging to control stack pressure
        for(uint32_t i=0;i<LLC_WAYS;i++){
            if(rrpv_A[set][i] < maxRRPV-1) rrpv_A[set][i]++;
        }
        rrpv_A[set][way] = 0;
    }
}

//------------------------------- Mode B: CR-DG-SB (EXCLUSIVE) -------------------------------
// Per-line (Mode B only)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];         // 3-bit logical
static uint8_t mode_B_hitcnt[LLC_SETS][LLC_WAYS];  // 0..3
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS]; // 0/1

// PC table (512 entries): stride/freq/dead/run/bypass
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit TinyLFU
static uint16_t mode_B_pc_last_set[MODE_B_PC_TABLE_SIZE];    // 11-bit
static uint8_t  mode_B_pc_run_uniq[MODE_B_PC_TABLE_SIZE];    // 3-bit (unique-set run length)
static uint8_t  mode_B_pc_dead2[MODE_B_PC_TABLE_SIZE];       // 2-bit dead-gate
static uint8_t  mode_B_pc_bypass_flip[MODE_B_PC_TABLE_SIZE]; // 1-bit soft-bypass toggler

// Tunables
static constexpr uint8_t MODE_B_HOT_THRESHOLD     = 8; // TinyLFU
static constexpr uint8_t MODE_B_INSERT_WARM       = 2;
static constexpr uint8_t MODE_B_INSERT_COLD       = 6;
static constexpr uint8_t MODE_B_STREAM_TAIL       = 7;
static constexpr uint8_t MODE_B_STREAM_ARM        = 2; // +1/+2 forward steps to arm
static constexpr uint8_t MODE_B_PROMOTE_HITS_BASE = 2; // non-stream rescue on 2nd hit
static constexpr uint8_t MODE_B_PROMOTE_HITS_STRM = 3; // stream needs 3 hits
static constexpr uint8_t MODE_B_LONG_RUN_UNIQSETS = 6; // long unique-set run threshold
static constexpr uint8_t MODE_B_BYPASS_RATE_SHIFT = 1; // ~50% via flip&1

// helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if(x<7) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }

static inline uint32_t mode_B_pc_index(uint64_t pc){ return CRC32(pc) & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10b
static inline bool mode_B_forward12(uint16_t last, uint16_t curr){
    uint16_t exp1 = (uint16_t)(last + 1), exp2 = (uint16_t)(last + 2);
    return (curr == exp1) || (curr == exp2);
}

// Forward-only +1/+2 run detector
static bool mode_B_detect_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool forward = (last != 0xFFFF) && mode_B_forward12(last, line_id);
    if(forward) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else        mode_B_pc_stride_conf[idx] = 0;

    mode_B_pc_last_line[idx] = line_id;
    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM);
}

// Update unique-set run and dead-gate telemetry
static inline void mode_B_run_dead_update(uint32_t pc_idx, uint32_t set, bool is_stream, bool pc_hot, bool is_hit, bool is_pref){
    uint16_t last_set = mode_B_pc_last_set[pc_idx];
    bool new_set = (last_set != 0x7FF) && (last_set != (uint16_t)set);
    if(is_stream && new_set) sat_inc_u3(mode_B_pc_run_uniq[pc_idx]);
    mode_B_pc_last_set[pc_idx] = (uint16_t)set;

    bool long_run = (mode_B_pc_run_uniq[pc_idx] >= MODE_B_LONG_RUN_UNIQSETS);

    // Dead-gate: penalize cold long-run demand traffic; relax on hits
    if(is_demand(!is_pref) && !is_hit && long_run && !pc_hot) sat_inc_u2(mode_B_pc_dead2[pc_idx]);
    if(is_hit) sat_dec_u2(mode_B_pc_dead2[pc_idx]);

    // Flip soft-bypass bit periodically (cheap pseudo-random)
    mode_B_pc_bypass_flip[pc_idx] ^= 1;
}

static void mode_B_epoch_decay(){
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_freq[i]      >>= 1; // TinyLFU decay
        mode_B_pc_dead2[i]     >>= 1; // dead-gate decay
        mode_B_pc_run_uniq[i]  >>= 1; // run-length decay
        // keep last_line/last_set as is; bypass_flip keeps toggling on use
    }
}

// Mode B insertion/update (EXCLUSIVE to rrpv_B)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU on demand
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Detect forward stream
    bool is_stream = mode_B_detect_stream(PC, paddr);
    mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Update run/dead telemetry (using current access outcome)
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    mode_B_run_dead_update(pc_idx, set, is_stream, pc_hot, hit, is_prefetch(type));

    // Hit counter
    if(hit) sat_inc_u2(mode_B_hitcnt[set][way]);
    else    mode_B_hitcnt[set][way] = 0;

    // Decide insertion depth
    bool long_run = (mode_B_pc_run_uniq[pc_idx] >= MODE_B_LONG_RUN_UNIQSETS);
    bool noisy_pc = (mode_B_pc_dead2[pc_idx] >= 2);
    bool soft_bypass = (long_run && !pc_hot && ((mode_B_pc_bypass_flip[pc_idx] & ((1u<<MODE_B_BYPASS_RATE_SHIFT)-1))==0));

    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL; // quarantine prefetches
    }else if(is_stream){
        if(long_run && (!pc_hot || noisy_pc)){
            rrpv_B[set][way] = MODE_B_STREAM_TAIL; // deep quarantine for scans
        }else{
            rrpv_B[set][way] = MODE_B_INSERT_COLD; // allow potential short reuse
        }
    }else{
        if(pc_hot)           rrpv_B[set][way] = MODE_B_INSERT_WARM;
        else if(noisy_pc)    rrpv_B[set][way] = MODE_B_STREAM_TAIL;
        else                 rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Soft-bypass: reinforce eviction pressure on long-run streams
    if(soft_bypass){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }

    // Stream-aware rescue
    if(hit){
        uint8_t need = mode_B_stream_tag[set][way] ? MODE_B_PROMOTE_HITS_STRM : MODE_B_PROMOTE_HITS_BASE;
        if(mode_B_hitcnt[set][way] >= need){
            rrpv_B[set][way] = 0; // MRU promote on sufficient hits
        }
    }
}

// Mode B victim (EXCLUSIVE to rrpv_B)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

//------------------------------- Minimal Hawkeye Sampler (for Mode A training only) -------------------------------
#define OPTGEN_VECTOR_SIZE 128
struct OPTgen {
    std::vector<unsigned int> liveness_history;
    uint64_t access=0, CACHE_SIZE=0;
    void init(uint64_t size){ CACHE_SIZE=size; liveness_history.assign(OPTGEN_VECTOR_SIZE,0); }
    void add_access(uint64_t curr_quanta){ access++; (void)curr_quanta; }
    bool should_cache(uint64_t, uint64_t){ return true; }
    uint64_t get_num_opt_hits(){ return 0; }
};
static OPTgen perset_optgen[LLC_SETS];
static uint64_t perset_mytimer[LLC_SETS];
#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
static std::vector<std::map<uint64_t, uint64_t>> addr_history; // tag->PC
static inline uint64_t CRC(uint64_t x){ return CRC32(x); }

//------------------------------- ChampSim hooks -------------------------------
void InitReplacementState(){
    // Init Mode A and Mode B arrays
    for(uint32_t s=0;s<LLC_SETS;s++){
        for(uint32_t w=0;w<LLC_WAYS;w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_hitcnt[s][w] = 0;
            mode_B_stream_tag[s][w] = 0;
        }
        perset_mytimer[s] = 0;
        perset_optgen[s].init(LLC_WAYS-2);
        // mode_conf defaults 0 (encoded as +8 bias later)
        if((s & 1)==0) mode_conf_nibbles[s>>1] = 0x88; // both nibbles = 8 (i.e., 0 signed)
    }

    // Init Mode B PC table
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i]=0;
        mode_B_pc_freq[i]=0;
        mode_B_pc_last_set[i]=0x7FF;
        mode_B_pc_run_uniq[i]=0;
        mode_B_pc_dead2[i]=0;
        mode_B_pc_bypass_flip[i]=0;
    }

    // Init sampler
    addr_history.resize(SAMPLER_SETS);
    for(uint32_t i=0;i<SAMPLER_SETS;i++) addr_history[i].clear();

    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    if(should_use_mode_B(set)) return mode_B_victim(set, current_set);
    return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Selector update (leaders score, epochic decay hooked here)
    update_selector(set, hit);

    // Minimal sampler bookkeeping to keep Mode A stable (sampled only)
    if(SAMPLED_SET(set)){
        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS;
        uint64_t sampler_tag = CRC(paddr >> 12) % 4096;
        addr_history[sampler_set][sampler_tag] = PC;
        perset_optgen[set].add_access(perset_mytimer[set] % OPTGEN_VECTOR_SIZE);
        perset_mytimer[set] = (perset_mytimer[set]+1) & 1023u;
    }

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates
    if(use_mode_B){
        mode_B_update(set, way, paddr, PC, type, hit);      // ONLY rrpv_B
    }else{
        hawkeye_update(set, way, PC, hit);                  // ONLY rrpv_A
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}