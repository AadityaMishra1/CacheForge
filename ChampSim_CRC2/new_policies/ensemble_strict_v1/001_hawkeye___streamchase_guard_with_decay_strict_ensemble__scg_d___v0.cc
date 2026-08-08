/*
 * Hawkeye + StreamChase-Guard with Decay (SCG-D)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP state.
 *
 * Tunables (telemetry hooks):
 * - MODE_A_BIAS (default 33), MODE_B_THRESHOLD (default 5), SELECTOR_EPOCH_SIZE (default 4096)
 * - Mode B: HOT_THRESHOLD=8, INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7
 * - STREAM_ARM_RUNS=2 (±1/±2), BYPASS_RUN_LEN=8 (cross-set), PROMOTE_NONSTREAM=2 hits, PROMOTE_STREAM≈3 (via hit_streak+hot gating)
 * - TinyLFU decay (shift/2 each epoch), deadness decay (shift/2 each epoch)
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// Sample 64 sets: low 6 bits == high 6 bits
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// CRC hash (small) for PC signatures
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
static constexpr int8_t MODE_B_THRESHOLD = 5;  // per-set confidence

// Per-set confidence (we count as 3-bit in budget; implemented as int8_t)
static int8_t mode_confidence[LLC_SETS];
// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1 tag per follower set

// ============================================================================
// Mode A: Hawkeye-lite (EXCLUSIVE rrpv_A + tiny SHCT predictors)
// - Budgeted SHCT: 2 tables × 512 entries × 4-bit counters
// ============================================================================
#define maxRRPV 7
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

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
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    // Retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

// ============================================================================
// Mode B: StreamChase-Guard with Decay (EXCLUSIVE rrpv_B)
// - Bidirectional ±1/±2 run detector (arm on 2 runs)
// - TinyLFU per-PC (4-bit), periodic decay
// - Deadness per-PC (2-bit), periodic decay
// - Cross-set run counter (8-bit last_set lossy, 4-bit run length)
// - Stream quarantine & bypass
// - Stream-aware multi-hit rescue (line 1-bit hit_seen + per-PC hit_streak)
// ============================================================================

static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];     // 3-bit logical
static uint8_t mode_B_hit_seen[LLC_SETS][LLC_WAYS]; // 0/1

// PC table (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE]; // 10-bit used, 0xFFFF = unset
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 0-3
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];   // 0-15
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];   // 0-3
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE]; // 8-bit lossy last set id
static uint8_t  mode_B_pc_crossrun[MODE_B_PC_TABLE_SIZE];  // 0-15 (saturating)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE]; // 0/1 (approximate)

static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

// Tunables
static constexpr uint8_t MODE_B_INSERT_WARM      = 2;
static constexpr uint8_t MODE_B_INSERT_COLD      = 6;
static constexpr uint8_t MODE_B_STREAM_TAIL      = 7;
static constexpr uint8_t MODE_B_STREAM_ARM_RUNS  = 2;  // need 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD    = 8;  // TinyLFU
static constexpr uint8_t MODE_B_BYPASS_RUN_LEN   = 8;  // cross-set runs to bypass

// Peek stream (no state change)
static bool mode_B_peek_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];
    bool near_stride = false;
    if(last != 0xFFFF){
        int16_t d = (int16_t)line_id - (int16_t)last;
        near_stride = (d==1)||(d==2)||(d==-1)||(d==-2);
    }
    return near_stride && (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM_RUNS);
}

// Update PC table on access (decide stride/stream; update freq/dead/hitstreak)
static bool mode_B_train_pc(uint64_t PC, uint32_t set, uint64_t paddr, uint32_t type, uint8_t hit){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);

    // Stride/run detector (±1/±2)
    bool near_stride = false;
    if(mode_B_pc_last_line[idx] != 0xFFFF){
        int16_t d = (int16_t)line_id - (int16_t)mode_B_pc_last_line[idx];
        near_stride = (d==1)||(d==2)||(d==-1)||(d==-2);
    }
    if(near_stride) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else mode_B_pc_stride_conf[idx] = 0;
    mode_B_pc_last_line[idx] = line_id;

    // Cross-set run (lossy 8-bit set id)
    uint8_t set8 = (uint8_t)(set & 0xFF);
    if(mode_B_pc_last_set8[idx] == set8+1 || mode_B_pc_last_set8[idx] + 1 == set8){
        if(mode_B_pc_crossrun[idx] < 15) mode_B_pc_crossrun[idx]++;
    } else if(mode_B_pc_last_set8[idx] != 0xFF){
        // reset on discontinuity
        mode_B_pc_crossrun[idx] = 0;
    }
    mode_B_pc_last_set8[idx] = set8;

    // TinyLFU and deadness
    if(is_demand(type)){
        if(hit) { sat_inc_u4(mode_B_pc_freq[idx]); sat_dec_u2(mode_B_pc_dead[idx]); mode_B_pc_hitstreak[idx] = 1; }
        else    { sat_inc_u2(mode_B_pc_dead[idx]); sat_inc_u4(mode_B_pc_freq[idx]); mode_B_pc_hitstreak[idx] = 0; }
    }

    bool stream_now = near_stride && (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM_RUNS);
    return stream_now;
}

// Bypass decision (peek-only)
static bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    if(is_writeback(type)) return false;
    uint32_t idx = mode_B_pc_index(PC);
    bool stream_now = mode_B_peek_stream(PC, paddr);
    bool pc_cold = (mode_B_pc_freq[idx] < MODE_B_HOT_THRESHOLD);
    bool pc_deadish = (mode_B_pc_dead[idx] >= 2);
    // Predict next crossrun length (assume stride continues across sets)
    uint8_t next_run = std::min<uint8_t>(15, mode_B_pc_crossrun[idx] + 1);
    if(stream_now && pc_cold && pc_deadish && (next_run >= MODE_B_BYPASS_RUN_LEN))
        return true; // bypass long cold stream/chase
    return false;
}

// Mode B insertion + rescue (EXCLUSIVE rrpv_B)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t idx = mode_B_pc_index(PC);
    bool stream_now = mode_B_train_pc(PC, set, paddr, type, hit);

    // Update per-line hit_seen
    if(hit) mode_B_hit_seen[set][way] = 1;
    else    mode_B_hit_seen[set][way] = 0;

    // Insert position
    bool pc_hot = (mode_B_pc_freq[idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_deadish = (mode_B_pc_dead[idx] >= 2);

    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL; // quarantine prefetches
    } else if(stream_now && (pc_deadish || !pc_hot)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL; // hard tail for cold/likely-dead streams
    } else if(pc_hot){
        rrpv_B[set][way] = MODE_B_INSERT_WARM; // warm insert
    } else {
        rrpv_B[set][way] = MODE_B_INSERT_COLD; // near-tail
    }

    // Stream-aware multi-hit rescue:
    // - Non-stream: promote on 2nd demand hit (line-level)
    // - Stream PCs: require extra evidence (PC hitstreak + hot)
    if(hit && is_demand(type)){
        if(!stream_now){
            if(mode_B_hit_seen[set][way]) rrpv_B[set][way] = 0;
        } else {
            if(mode_B_pc_hitstreak[idx] && pc_hot && mode_B_hit_seen[set][way]) rrpv_B[set][way] = 0;
        }
    }
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

// ============================================================================
// Selector helpers
// ============================================================================
static inline void selector_on_access(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t d = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += d;
        if(LEADER_B(set)) leaderB_score += d;
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > leaderA_score + MODE_A_BIAS);

        // Drift follower confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 3) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -4) mode_confidence[s]--; }
        }

        // Periodic decay for Mode B PC stats (track phase shifts)
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i] >>= 1;     // TinyLFU decay
            mode_B_pc_dead[i] >>= 1;     // deadness decay
            mode_B_pc_crossrun[i] >>= 1; // run-length decay
            mode_B_pc_hitstreak[i] = 0;  // clear streak window
        }

        // Reset scores
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;

    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    // Enforce follower cap
    if(!follower_is_B[set]){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_is_B[set] = 1;
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// ChampSim hooks
// ============================================================================
void InitReplacementState(){
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){ rrpv_A[s][w]=maxRRPV; rrpv_B[s][w]=maxRRPV; mode_B_hit_seen[s][w]=0; }
        mode_confidence[s]=0; follower_is_B[s]=0;
    }
    for(uint32_t i=0; i<SHCT_ENTRIES; i++){ SHCT_demand[i]=8; SHCT_prefetch[i]=8; }

    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i]=0xFFFF;
        mode_B_pc_stride_conf[i]=0;
        mode_B_pc_freq[i]=0;
        mode_B_pc_dead[i]=0;
        mode_B_pc_last_set8[i]=0xFF;
        mode_B_pc_crossrun[i]=0;
        mode_B_pc_hitstreak[i]=0;
    }

    leaderA_score=leaderB_score=0; prefer_mode_B=false; selector_epoch=0; num_mode_B_followers=0;
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)paddr;
    bool useB = should_use_mode_B(set);
    // Mode B can bypass on long cold stream/chase
    if(useB && mode_B_should_bypass(set, PC, paddr, type)) return 16; // bypass fill

    if(useB) return mode_B_victim(set, current_set);
    else     return mode_A_victim(set, current_set);
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr; (void)paddr;

    // Selector update (leaders collect outcomes)
    selector_on_access(set, hit);

    // Ignore writebacks for policy updates
    if(is_writeback(type)) return;

    bool useB = should_use_mode_B(set);
    bool bypass = (useB && way==16);

    // Train Mode A predictors lightly (no shared RRIP state)
    hawkeye_train(PC, is_prefetch(type), hit);

    if(useB){
        if(!bypass){
            // EXCLUSIVE: update ONLY rrpv_B
            mode_B_update(set, way, paddr, PC, type, hit);
        } else {
            // Bypass still trains Mode B PC table (no RRIP update)
            (void)mode_B_train_pc(PC, set, paddr, type, hit);
        }
    } else {
        // Mode A insertion (EXCLUSIVE rrpv_A)
        bool friendly = hawkeye_is_friendly(PC, is_prefetch(type));
        if(friendly){
            // Insert MRU; simple aging to avoid overflow
            for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
            rrpv_A[set][way] = 0;
        } else {
            rrpv_A[set][way] = maxRRPV;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}