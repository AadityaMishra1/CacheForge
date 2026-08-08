/*
 * Hawkeye + HST-E TightGate Strict Ensemble (HST-TG)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 * - Mode A: Hawkeye-lite (SHCT-style friendliness) with exclusive rrpv_A
 * - Mode B: HST-E (stride/run + TinyLFU + dead-gate + quarantine + rescue) with exclusive rrpv_B
 * - Selector: Very conservative (BIAS=50, THRESHOLD=7), follower cap=256, B-leader safety (conditional)
 *
 * Storage telemetry (bit-packed intent):
 * - Mode A: rrpv_A: 3b/line; SHCT: 512x5b
 * - Mode B: rrpv_B: 3b/line; first_hit_flag: 1b/line; PC-table: 128 tiny entries
 * - Selector: per-set confidence 4b/set, follower bitmap 1b/set, leader scores
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
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

// Leader-follower sampling: 64 sampled sets (low-6 == high-6)
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// Lightweight CRC32 hash for indices
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// ============================================================================
// Selector state (conservative gating)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50; // ultra-conservative towards Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;  // strict per-set confidence

// Per-set confidence counters (-8..+7)
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5% of sets)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // 1 bit per set
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta; // B-leaders mirror A until prefer_mode_B, but we still score them
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Global decision: Mode B must beat Mode A by a large margin
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Drift local confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 7) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -8) mode_confidence[s]--; }
        }

        // Reset leader scores
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    // B LEADER SAFETY: B-leaders conditional on global win
    if(LEADER_B(set)) return prefer_mode_B;
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// Mode A: Hawkeye-lite (exclusive RRIP + tiny SHCT)
// ============================================================================
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

// Tiny SHCT (512 entries, 5-bit counters)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31
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

static inline void hawkeye_update(uint32_t set, uint32_t way, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;
    hawkeye_train(PC, hit);
    bool friendly = hawkeye_is_friendly(PC);
    rrpv_A[set][way] = friendly ? 0 : maxRRPV;
}

// ============================================================================
// Mode B: HST-E (exclusive RRIP + stride/run + TinyLFU + dead-gate)
// REQUIRED LOGIC preserved; tuned constants noted below.
// ============================================================================
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];                 // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS];    // 1-bit first-hit flag

// PC table (compact, 128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];    // 10-bit (stored in 16-bit)
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE];  // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];         // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];         // 2-bit dead score
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];    // last set low-8 (telemetry)
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];     // 8-bit unique-set mask
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];      // 4-bit (0-15)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];    // 2-bit (0-3)

// Tunables (fixed)
// Insertion depths and thresholds: tightened for scans/pointer-chasing lift
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age for cold
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // hard tail for streams/prefetch
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // arm after 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >= 6
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >= 2
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // long run if >=4 unique sets

// Helpers
static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit line id
static inline void sat_inc_u2(uint8_t &x){ if(x < 3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x < 15) x++; }
static inline uint8_t popcount8(uint8_t x){
    x = x - ((x>>1) & 0x55); x = (x & 0x33) + ((x>>2) & 0x33); return (uint8_t)((x + (x>>4)) & 0x0F);
}

// Detect stride/stream with run tracking
static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set){
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];

    bool step = false;
    if(last != 0xFFFF){
        // Bidirectional ±1/±2
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }

    if(step){
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if(mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM){
            sat_inc_u4(mode_B_pc_run_len[pc_idx]);
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    }else{
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        mode_B_pc_setmask8[pc_idx] = 0;
    }

    mode_B_pc_last_line[pc_idx] = line_id;
    mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 0xFFu);
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx){
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

// ONLY bypass if: long stream AND cold AND dead; never bypass prefetches
static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    (void)set;
    if(is_prefetch(type)) return false;
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);
    if(is_stream && long_run && !pc_hot && pc_dead) return true;
    return false;
}

// Mode B update (updates ONLY rrpv_B and Mode B metadata)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU frequency
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead block tracking
    if(hit){
        if(mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if(mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    }else{
        if(is_demand(type) && mode_B_pc_hitstreak[pc_idx] == 0) sat_inc_u2(mode_B_pc_dead[pc_idx]);
        if(mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion policy
    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(is_stream && long_run){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(pc_dead){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(pc_hot){
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    }else{
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue with stream awareness
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

// Mode B victim (uses ONLY rrpv_B)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

// ============================================================================
// ChampSim interface
// ============================================================================
void InitReplacementState(){
    // Initialize Mode A and B RRIPs and metadata
    for(uint32_t s=0; s<LLC_SETS; s++){
        mode_confidence[s] = 0;
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_line_touched[s][w] = 0;
        }
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));
    std::memset(follower_bits, 0, sizeof(follower_bits));

    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_last_set8[i] = 0;
        mode_B_pc_setmask8[i] = 0;
        mode_B_pc_run_len[i] = 0;
        mode_B_pc_hitstreak[i] = 0;
    }

    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_mode_B = should_use_mode_B(set);
    if(use_mode_B){
        if(mode_B_should_bypass(set, PC, paddr, type))
            return LLC_WAYS; // bypass
        return mode_B_victim(set, current_set);
    }else{
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Selector update on every access
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE updates: update only the chosen mode
    if(use_mode_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    }else{
        hawkeye_update(set, way, PC, type, hit);
    }

    // Epochic decay for Mode B PC stats (at epoch boundary in update_selector)
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i] >>= 1;
            mode_B_pc_dead[i] >>= 1;
            mode_B_pc_run_len[i] >>= 1;
            mode_B_pc_setmask8[i] >>= 1;
            mode_B_pc_hitstreak[i] >>= 1;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}