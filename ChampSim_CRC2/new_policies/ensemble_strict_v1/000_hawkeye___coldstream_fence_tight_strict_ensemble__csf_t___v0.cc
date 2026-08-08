/*
 * Hawkeye + ColdStream-Fence Tight (CSF-T)
 * STRICT DYNAMIC ENSEMBLE with EXCLUSIVE STATE PER MODE
 *
 * - Mode A: Hawkeye-lite (PC-SHCT friendly/averse) using its own RRIP (rrpv_A)
 * - Mode B: HST-E derived ColdStream-Fence (stride/run detector + TinyLFU + dead fence)
 *           using its own RRIP (rrpv_B) and compact PC table
 * - Selector: Very conservative gating, BIAS=50, THRESHOLD=7, 12.5% follower cap
 *   B Leader Safety: B-leaders mirror Hawkeye unless prefer_mode_B==true
 *
 * Tunables (CSF-T within allowed ranges):
 *   MODE_B_INSERT_WARM=0, MODE_B_INSERT_COLD=3, MODE_B_STREAM_TAIL=7
 *   MODE_B_STREAM_ARM=2 (±1/±2 strides, bidirectional)
 *   MODE_B_HOT_THRESHOLD=6 (TinyLFU ≥6 → MRU)
 *   MODE_B_DEAD_THRESHOLD=2 (dead fence)
 *   MODE_B_LONGRUN_UNIQUESETS=4 (≥4 unique sets → long stream)
 *
 * Telemetry hooks (epochic): TinyLFU freq decay (>>=1); leader scores per epoch
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define maxRRPV 7

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// -----------------------------------------------------------------------------
// Leader-follower sampling (match Hawkeye's macro)
// -----------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1ULL << (l))-1ULL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)std::log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// -----------------------------------------------------------------------------
// Simple CRC32-ish mixer for indices
// -----------------------------------------------------------------------------
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// -----------------------------------------------------------------------------
// Selector (very conservative gating + B leader safety)
// -----------------------------------------------------------------------------
static int32_t  leaderA_score = 0, leaderB_score = 0;
static bool     prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // huge bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict local threshold

// Per-set confidence: store two 4-bit signed nibbles per byte (-8..+7 encoded +8)
static uint8_t mode_conf_nibbles[LLC_SETS/2];
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

// Follower cap (~12.5% of sets)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Forward decl for Mode B decay
static void mode_B_epoch_decay();

// Update selector each access; decay TinyLFU each epoch
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
        // Mode B epochic decay (TinyLFU)
        mode_B_epoch_decay();

        leaderA_score = 0; leaderB_score = 0;
    }
}

// Decide mode per access (B-leader safety: conditional)
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return prefer_mode_B; // mirror Hawkeye unless B is globally winning
    if(!prefer_mode_B) return false;
    if(get_mode_conf(set) < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Mode A: Hawkeye-lite (exclusive RRIP + tiny SHCT)
// -----------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

// Tiny SHCT (512 entries, 5-bit counters)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // store 0..31
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static inline uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

static inline void hawkeye_update(uint32_t set, uint32_t way, uint64_t /*paddr*/, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;

    // Train SHCT on outcome
    hawkeye_train(PC, hit);

    if(hit){
        // Promote on hit
        rrpv_A[set][way] = 0;
        return;
    }

    // Miss + fill: friendly PCs insert MRU, averse at tail
    if(hawkeye_is_friendly(PC)) rrpv_A[set][way] = 0;
    else                        rrpv_A[set][way] = maxRRPV;
}

// -----------------------------------------------------------------------------
// Mode B: HST-E derived ColdStream-Fence (exclusive RRIP + compact PC table)
// -----------------------------------------------------------------------------

// Per-line state
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];            // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS]; // 1-bit first-hit flag

// PC table (compact: 128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static inline uint32_t mode_B_pc_index(uint64_t pc){ return CRC32(pc) & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FFu); } // 10-bit line ID
static inline uint8_t  popcount8(uint8_t x){ return (uint8_t)__builtin_popcount((unsigned)x); }
static inline void     sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void     sat_inc_u4(uint8_t &x){ if(x<15) x++; }

// PC metadata
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit (stored in 16)
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit (0..3)
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // TinyLFU 4-bit (0..15)
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit (0..3)
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];   // low 8 of last set (telemetry)
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];    // unique-set mask (8-bit)
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];     // 4-bit (0..15)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];   // 2-bit (0..3)

// Tunables (tight spec)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // LRU for streams
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >=6
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >=2
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // >=4 unique sets

// Detect stride/stream with run tracking (bidirectional ±1/±2)
static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set){
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];

    bool step = false;
    if(last != 0xFFFF){
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }

    if(step){
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if(mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM){
            // Run growth + unique-set tracking
            if(mode_B_pc_run_len[pc_idx] < 15) mode_B_pc_run_len[pc_idx]++;
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    }else{
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        mode_B_pc_setmask8[pc_idx] = 0;
    }

    mode_B_pc_last_line[pc_idx] = line_id;
    mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 0xFF);
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx){
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

// Conservative bypass: long stream AND cold AND dead; never bypass prefetches
static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    if(is_prefetch(type)) return false;
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot  = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);
    return (is_stream && long_run && !pc_hot && pc_dead);
}

// Mode B update (exclusive)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU frequency gate
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead tracking + hit streak (per-PC)
    if(hit){
        if(mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if(mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    }else{
        if(is_demand(type) && mode_B_pc_hitstreak[pc_idx] == 0){
            if(mode_B_pc_dead[pc_idx] < 3) mode_B_pc_dead[pc_idx]++;
        }
        if(mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run  = mode_B_is_long_run(pc_idx);
    bool pc_hot    = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead   = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion policy
    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(is_stream && long_run){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(pc_dead){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(pc_hot){
        rrpv_B[set][way] = MODE_B_INSERT_WARM;  // MRU
    }else{
        rrpv_B[set][way] = MODE_B_INSERT_COLD;  // mid-age
    }

    // Multi-hit rescue (stream aware)
    if(hit){
        if(mode_B_line_touched[set][way]){
            if(!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2){
                rrpv_B[set][way] = 0; // MRU promote
            }
        }else{
            mode_B_line_touched[set][way] = 1;
        }
    }else{
        mode_B_line_touched[set][way] = 0;
    }
}

// Mode B victim (RRIP, exclusive)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

static void mode_B_epoch_decay(){
    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_freq[i] >>= 1;
        // leave dead/stride/run state intact (conservative)
    }
}

// -----------------------------------------------------------------------------
// ChampSim interface
// -----------------------------------------------------------------------------
void InitReplacementState(){
    std::memset(rrpv_A, maxRRPV, sizeof(rrpv_A));
    std::memset(rrpv_B, maxRRPV, sizeof(rrpv_B));
    std::memset(mode_B_line_touched, 0, sizeof(mode_B_line_touched));
    std::memset(mode_conf_nibbles, 0x88, sizeof(mode_conf_nibbles)); // encode 0 as 8
    std::memset(follower_bits, 0, sizeof(follower_bits));
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
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
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_mode_B = should_use_mode_B(set);
    if(use_mode_B){
        return mode_B_victim(set, current_set);
    }else{
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;

    if(is_writeback(type)) return;

    // Update selector gates (leaders accumulate outcomes)
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // OPTIONAL bypass in Mode B only (exclusive)
    if(use_mode_B){
        if(!hit && mode_B_should_bypass(set, PC, paddr, type)){
            // On bypass, do not touch either mode's RRIP state
            return;
        }
        mode_B_update(set, way, paddr, PC, type, hit); // EXCLUSIVE: rrpv_B only
    }else{
        hawkeye_update(set, way, paddr, PC, type, hit); // EXCLUSIVE: rrpv_A only
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}