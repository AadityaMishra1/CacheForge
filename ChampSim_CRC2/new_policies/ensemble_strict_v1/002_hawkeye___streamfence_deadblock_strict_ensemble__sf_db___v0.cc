/*
 * Hawkeye + StreamFence-DeadBlock (SF-DB)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Tunables (telemetry hooks):
 * - Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, SELECTOR_EPOCH_SIZE=4096, MAX_MODE_B_FOLLOWERS=256
 * - Mode B: HOT_THRESHOLD=8, INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7, STREAM_ARM=2 (±1/±2),
 *           BYPASS_RUN_LEN=8 (cross-set), PROB_BYPASS≈50% via per-PC flip, DEAD_STRONG=3 (2-bit max),
 *           stream rescue gate: non-stream promote on 2nd hit; stream promotes only if (seen_once && pc_hitstreak>=2).
 * - Decay each epoch: pc_freq>>=1, pc_dead>>=1, pc_crossrun>>=1.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11

// Access types (CRC2)
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

// Small PC hash
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// ============================================================================
// Selector (conservative)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;     // Hawkeye favored
static constexpr int8_t MODE_B_THRESHOLD = 5;  // per-set confidence (need strong)
static int8_t mode_confidence[LLC_SETS];       // -8..+7 (we budget as 4 bits signed)

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1 tag per follower set

static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Global decision with strong bias toward Mode A
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Drift follower confidence towards global preference
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
    if(LEADER_B(set)) return true;
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    // Enforce follower cap lazily
    if(!follower_is_B[set]){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_is_B[set] = 1;
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// Mode A: Hawkeye-lite (EXCLUSIVE rrpv_A + tiny SHCT predictors)
// ============================================================================
#define maxRRPV 7
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // logically 3-bit

// Tiny SHCT (budgeted)
#define SHCT_ENTRIES 512
static uint8_t SHCT_demand[SHCT_ENTRIES];   // 0-15
static uint8_t SHCT_prefetch[SHCT_ENTRIES]; // 0-15
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC, bool is_pf){
    uint8_t c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    return (c >= 8);
}
static inline void hawkeye_train(uint64_t PC, bool is_pf, bool hit){
    uint8_t &c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    if(hit){ if(c<15) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    // Find maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    // Age all
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    // Retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

// ============================================================================
// Mode B: StreamFence-DeadBlock (EXCLUSIVE rrpv_B)
// - Bidirectional ±1/±2 run detector (arm on 2 runs)
// - TinyLFU per-PC (4-bit) with epoch decay
// - Deadness per-PC (2-bit) with epoch decay
// - Cross-set run counter (4-bit) + last_set8 (8-bit) with epoch decay
// - Prefetch/stream quarantine at tail; probabilistic bypass on long runs
// - Stream-aware multi-hit rescue using 1-bit per-line + per-PC hitstreak
// ============================================================================
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];     // 3-bit logical
static uint8_t mode_B_hit_seen[LLC_SETS][LLC_WAYS]; // 0/1 (per-line rescue latch)

// PC table (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE]; // 10-bit used, 0xFFFF=unset
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 0-3
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];   // 0-15 (TinyLFU)
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];   // 0-3 (dead-on-fill tendency)
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE]; // 8-bit lossy set id
static uint8_t  mode_B_pc_crossrun[MODE_B_PC_TABLE_SIZE];  // 0-15
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE]; // 0-3
static uint8_t  mode_B_pc_bflip[MODE_B_PC_TABLE_SIZE];     // 0/1 toggler for ~50% bypass

// Tunables
static constexpr uint8_t MODE_B_INSERT_WARM      = 2;
static constexpr uint8_t MODE_B_INSERT_COLD      = 6;
static constexpr uint8_t MODE_B_STREAM_TAIL      = 7;
static constexpr uint8_t MODE_B_STREAM_ARM_RUNS  = 2;  // need 2 consecutive runs
static constexpr uint8_t MODE_B_HOT_THRESHOLD    = 8;  // TinyLFU hot
static constexpr uint8_t MODE_B_BYPASS_RUN_LEN   = 8;  // long cross-set run
static constexpr uint8_t MODE_B_DEAD_STRONG      = 3;  // strong dead PC

static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit
static inline uint8_t  mode_B_set8(uint32_t set){ return (uint8_t)(set & 0xFF); }

static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

// Preview whether this access will arm stream and grow cross-run (for bypass decision in GetVictim)
static inline void mode_B_preview(uint32_t set, uint64_t PC, uint64_t paddr, bool &will_stream, uint8_t &next_crossrun){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool run = false;
    if(last != 0xFFFF){
        uint16_t exp1f = (uint16_t)(last + 1), exp2f = (uint16_t)(last + 2);
        uint16_t exp1b = (uint16_t)(last - 1), exp2b = (uint16_t)(last - 2);
        run = (line_id==exp1f)||(line_id==exp2f)||(line_id==exp1b)||(line_id==exp2b);
    }
    uint8_t conf = mode_B_pc_stride_conf[idx];
    if(run){ if(conf<3) conf++; } else { conf = 0; }
    will_stream = (conf >= MODE_B_STREAM_ARM_RUNS);

    uint8_t set8 = mode_B_set8(set);
    uint8_t last_set = mode_B_pc_last_set8[idx];
    uint8_t cr = mode_B_pc_crossrun[idx];
    if(set8 != last_set){ if(cr<15) cr++; } else { cr = 0; }
    next_crossrun = cr;
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    if(is_prefetch(type)) return false; // prefetches quarantined, not bypassed
    uint32_t idx = mode_B_pc_index(PC);
    bool will_stream=false; uint8_t cr=0; mode_B_preview(set, PC, paddr, will_stream, cr);
    bool pc_hot = (mode_B_pc_freq[idx] >= MODE_B_HOT_THRESHOLD);
    bool long_run = (cr >= MODE_B_BYPASS_RUN_LEN);
    bool very_dead = (mode_B_pc_dead[idx] >= MODE_B_DEAD_STRONG);

    if((will_stream && long_run && !pc_hot) || (very_dead && !pc_hot)){
        // ~50% probabilistic bypass via toggler
        mode_B_pc_bflip[idx] ^= 1;
        return (mode_B_pc_bflip[idx] != 0);
    }
    return false;
}

// Detect stride/stream and update PC table (called in UpdateReplacementState)
static inline bool mode_B_detect_stream_and_update(uint32_t set, uint64_t PC, uint64_t paddr){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool run = false;
    if(last != 0xFFFF){
        uint16_t exp1f = (uint16_t)(last + 1), exp2f = (uint16_t)(last + 2);
        uint16_t exp1b = (uint16_t)(last - 1), exp2b = (uint16_t)(last - 2);
        run = (line_id==exp1f)||(line_id==exp2f)||(line_id==exp1b)||(line_id==exp2b);
    }

    if(run) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else    mode_B_pc_stride_conf[idx] = 0;

    mode_B_pc_last_line[idx] = line_id;

    // Cross-set run
    uint8_t set8 = mode_B_set8(set);
    if(set8 != mode_B_pc_last_set8[idx]) sat_inc_u4(mode_B_pc_crossrun[idx]); // 0-15 reuse
    else                                 mode_B_pc_crossrun[idx] = 0;
    mode_B_pc_last_set8[idx] = set8;

    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM_RUNS);
}

// Mode B insertion/update (updates ONLY rrpv_B and Mode B metadata)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t idx = mode_B_pc_index(PC);

    // Frequency and deadness on demand
    if(is_demand(type)){
        sat_inc_u4(mode_B_pc_freq[idx]);
        if(!hit) sat_inc_u2(mode_B_pc_dead[idx]); // dead-on-fill tendency for misses
        else     sat_dec_u2(mode_B_pc_dead[idx]); // reduce deadness on useful hits
    }

    bool is_stream = mode_B_detect_stream_and_update(set, PC, paddr);
    bool pc_hot = (mode_B_pc_freq[idx] >= MODE_B_HOT_THRESHOLD);

    // Update PC hitstreak (for stream-aware rescue gating)
    if(hit) { if(mode_B_pc_hitstreak[idx] < 3) mode_B_pc_hitstreak[idx]++; }
    else    { if(mode_B_pc_hitstreak[idx] > 0) mode_B_pc_hitstreak[idx]--; }

    // Update per-line rescue latch
    if(hit){
        if(!is_stream){
            if(mode_B_hit_seen[set][way]) rrpv_B[set][way] = 0; // promote on 2nd hit (non-stream)
            mode_B_hit_seen[set][way] = 1;
        }else{
            // Stream: require "seen_once" + short hitstreak evidence
            if(mode_B_hit_seen[set][way] && mode_B_pc_hitstreak[idx] >= 2) rrpv_B[set][way] = 0;
            mode_B_hit_seen[set][way] = 1;
        }
        return; // on hits, only promotion/latch update; don't re-insert
    }else{
        mode_B_hit_seen[set][way] = 0; // reset on miss/fill
    }

    // Insertion (fills): quarantine prefetches and streams; otherwise hot/warm vs cold/tail
    if(is_prefetch(type) || is_stream){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    }else if(pc_hot && mode_B_pc_dead[idx] <= 1){
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    }else{
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }
}

// Mode B victim selection (uses ONLY rrpv_B)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
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
    // Clear arrays
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_hit_seen[s][w] = 0;
        }
        mode_confidence[s] = 0;
        follower_is_B[s] = 0;
    }
    num_mode_B_followers = 0;
    leaderA_score = 0; leaderB_score = 0; prefer_mode_B = false; selector_epoch = 0;

    for(uint32_t i=0; i<SHCT_ENTRIES; i++){
        SHCT_demand[i] = 8;
        SHCT_prefetch[i] = 8;
    }

    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_last_set8[i] = 0;
        mode_B_pc_crossrun[i] = 0;
        mode_B_pc_hitstreak[i] = 0;
        mode_B_pc_bflip[i] = 0;
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_B = should_use_mode_B(set);

    // Optional bypass under Mode B
    if(use_B && is_demand(type)){
        if(mode_B_should_bypass(set, PC, paddr, type)){
            return 16; // bypass fill
        }
    }

    if(use_B){
        return mode_B_victim(set, current_set);
    }else{
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    // Ignore writebacks
    if(is_writeback(type)) return;

    // Selector feedback
    update_selector(set, hit);

    bool use_B = should_use_mode_B(set);

    // Epochic decay for Mode B PC tables and keep selector cadence
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i] >>= 1;
            mode_B_pc_dead[i] >>= 1;
            mode_B_pc_crossrun[i] >>= 1;
            if(mode_B_pc_hitstreak[i] > 0) mode_B_pc_hitstreak[i]--;
        }
    }

    // TRAIN Mode A tiny SHCT (always)
    hawkeye_train(PC, is_prefetch(type), hit != 0);

    // EXCLUSIVE updates
    if(use_B){
        // Mode B update ONLY rrpv_B
        mode_B_update(set, way, paddr, PC, type, hit);
    }else{
        // Mode A update ONLY rrpv_A
        if(hit){
            rrpv_A[set][way] = 0;
        }else{
            bool friendly = hawkeye_is_friendly(PC, is_prefetch(type));
            rrpv_A[set][way] = friendly ? 0 : maxRRPV;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}