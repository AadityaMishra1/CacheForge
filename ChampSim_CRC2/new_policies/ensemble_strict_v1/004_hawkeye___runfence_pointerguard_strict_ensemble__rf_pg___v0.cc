/*
 * Hawkeye + RunFence-PointerGuard (RF-PG)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Tunables (telemetry hooks):
 * - Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, SELECTOR_EPOCH_SIZE=4096, MAX_MODE_B_FOLLOWERS=256
 * - Mode B: HOT_THRESHOLD=8, DEAD_STRONG=3, STREAM_ARM=2 (±1/±2), RUN_BYPASS_LEN=8, BYPASS_PROB=50%,
 *           INSERT_WARM=1 (2-bit RRIP), INSERT_COLD=3, STREAM_TAIL=3,
 *           stream rescue gate: non-stream promotes on 2nd demand hit (per-line seen_once==1),
 *           stream promotes only if pc_hitstreak>=3.
 * - Epoch decay each SELECTOR_EPOCH: pc_freq>>=1, pc_dead>>=1, pc_run_len>>=1, pc_hitstreak>>=1.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <map>
#include <cstring>

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

// Hash
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// PRNG for probabilistic bypass
static uint32_t prng_state = 0xBADC0FFE;
static inline uint32_t fast_rand(){
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

// Sample 64 sets: low 6 bits == high 6 bits
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// ============================================================================
// Selector (conservative gating)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;     // favor Hawkeye
static constexpr int8_t MODE_B_THRESHOLD = 5;  // per-set confidence
static int8_t mode_confidence[LLC_SETS];       // -8..+7 packed into int8

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1 tag per follower set

static inline void selector_decay_epoch();

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

        // Periodic decay for Mode B PC tables and Mode A SHCT (see selector_decay_epoch)
        selector_decay_epoch();

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
// - 2-bit RRIP (maxRRPV=3) to reduce storage
// - SHCT (2K x 5b) for demand/prefetch friendliness
// - Train: increment on hits; weak negative on misses (current PC)
// ============================================================================
#define maxRRPV 3
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 2-bit logical

#define SHCT_ENTRIES 2048
static uint8_t SHCT_demand[SHCT_ENTRIES];   // 0-31 (5-bit logical)
static uint8_t SHCT_prefetch[SHCT_ENTRIES]; // 0-31 (5-bit logical)
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }

static inline bool hawkeye_is_friendly(uint64_t PC, bool is_pf){
    uint8_t c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    // Midpoint threshold
    return (c >= 16);
}
static inline void hawkeye_train(uint64_t PC, bool is_pf, bool hit){
    uint8_t &c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    if(hit){ if(c<31) c++; }
    else    { if(c>0)  c--; } // weak negative on misses
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
// Mode B: RunFence-PointerGuard (EXCLUSIVE rrpv_B)
// - Bidirectional ±1/±2 run detector (arm on 2 runs)
// - TinyLFU per-PC (4-bit) with epoch decay
// - Deadness per-PC (2-bit) with epoch decay
// - Cross-set run counter (4-bit) + last_set8 (8-bit) with epoch decay
// - Prefetch/stream quarantine at tail; probabilistic bypass on long runs
// - Stream-aware multi-hit rescue using per-line seen_once + per-PC hitstreak
// ============================================================================
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];     // 2-bit logical
static uint8_t mode_B_seen_once[LLC_SETS][LLC_WAYS]; // 0/1
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS];  // 0/1

// PC table (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit payload in 16b
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit deadness
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];   // 8-bit last set id
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];     // 4-bit cross-set run
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];   // 3-bit recent hit streak

// Tunables
static constexpr uint8_t MODE_B_INSERT_WARM      = 1;   // Near-MRU (2-bit RRIP)
static constexpr uint8_t MODE_B_INSERT_COLD      = 3;   // Tail
static constexpr uint8_t MODE_B_STREAM_TAIL      = 3;   // Quarantine
static constexpr uint8_t MODE_B_STREAM_ARM       = 2;   // need 2 confirms
static constexpr uint8_t MODE_B_HOT_THRESHOLD    = 8;   // TinyLFU hot (0-15)
static constexpr uint8_t MODE_B_DEAD_STRONG      = 3;   // pc_dead >= 3 => noisy/dead
static constexpr uint8_t MODE_B_PROMOTE_STREAM_HITS = 3; // PC hitstreak needed for streams
static constexpr uint8_t MODE_B_RUN_BYPASS_LEN   = 8;   // Cross-set long run
static constexpr uint8_t MODE_B_BYPASS_PROB_SHIFT= 1;   // 1 => ~50% (use (rand & ((1<<s)-1))==0)

// Helpers
static inline uint32_t mode_B_pc_index(uint64_t pc) { return CRC32(pc) & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if(x<7) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }

// Stream detection (bidirectional ±1/±2). Update on UpdateReplacementState only.
static bool mode_B_detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool step = false;
    if (last != 0xFFFF) {
        uint16_t exp1p = (uint16_t)(last + 1);
        uint16_t exp2p = (uint16_t)(last + 2);
        uint16_t exp1n = (uint16_t)(last - 1);
        uint16_t exp2n = (uint16_t)(last - 2);
        step = (line_id == exp1p) || (line_id == exp2p) || (line_id == exp1n) || (line_id == exp2n);
    }

    if (step) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else      mode_B_pc_stride_conf[idx] = 0;

    mode_B_pc_last_line[idx] = line_id;
    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM);
}

static inline void mode_B_update_pc_crossrun(uint32_t pc_idx, uint32_t set){
    uint8_t set8 = (uint8_t)(set & 0xFF);
    if(mode_B_pc_last_set8[pc_idx] != 0xFF && mode_B_pc_last_set8[pc_idx] != set8){
        if(mode_B_pc_run_len[pc_idx] < 15) mode_B_pc_run_len[pc_idx]++;
    }else{
        mode_B_pc_run_len[pc_idx] = 0;
    }
    mode_B_pc_last_set8[pc_idx] = set8;
}

static inline bool mode_B_pc_hot(uint32_t pc_idx){ return (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD); }
static inline bool mode_B_pc_deadish(uint32_t pc_idx){ return (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_STRONG); }

// Mode B insertion/hit update (updates ONLY rrpv_B + per-line B metadata)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pc_idx = mode_B_pc_index(PC);

    // Update TinyLFU + deadness on demand reference
    if (is_demand(type)) {
        sat_inc_u4(mode_B_pc_freq[pc_idx]);
        if (!hit) sat_inc_u2(mode_B_pc_dead[pc_idx]); // write-once/no-reuse PCs accumulate deadness
    }

    // Update PC cross-set run
    mode_B_update_pc_crossrun(pc_idx, set);

    // Detect stream and tag line
    bool is_stream = mode_B_detect_stream(PC, paddr);
    if (!hit) mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Record per-line seen_once
    if (hit) {
        if (mode_B_seen_once[set][way] == 0) {
            mode_B_seen_once[set][way] = 1; // first hit observed
        } else {
            // Multi-hit rescue
            bool line_is_stream = (mode_B_stream_tag[set][way] != 0);
            if (!line_is_stream) {
                rrpv_B[set][way] = 0; // MRU on 2nd demand hit
            } else {
                // Stream needs stronger PC-level evidence
                if (mode_B_pc_hitstreak[pc_idx] >= MODE_B_PROMOTE_STREAM_HITS) {
                    rrpv_B[set][way] = 0; // MRU
                }
            }
        }
        // Bump PC hitstreak
        sat_inc_u3(mode_B_pc_hitstreak[pc_idx]);
    } else {
        mode_B_seen_once[set][way] = 0; // new fill path
    }

    // Insertion policy on fills (do not re-assign on hits)
    if (!hit) {
        bool pc_hot = mode_B_pc_hot(pc_idx);
        bool pc_dead = mode_B_pc_deadish(pc_idx);

        if (is_prefetch(type) || is_stream) {
            rrpv_B[set][way] = MODE_B_STREAM_TAIL;  // Quarantine
        } else if (pc_hot && !pc_dead) {
            rrpv_B[set][way] = MODE_B_INSERT_WARM;  // Near-MRU
        } else {
            rrpv_B[set][way] = MODE_B_INSERT_COLD;  // Tail
        }
    }
}

// Mode B victim/bypass (uses ONLY rrpv_B and Mode B PC table)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    // Probabilistic bypass for long cross-set runs (streams/prefetches/cold PCs)
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool long_run = (mode_B_pc_run_len[pc_idx] >= MODE_B_RUN_BYPASS_LEN);
    bool pc_hot = mode_B_pc_hot(pc_idx);
    bool likely_stream = (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
    if ((is_prefetch(type) || (likely_stream && !pc_hot) || (long_run && !pc_hot)) && ( (fast_rand() & ((1u<<MODE_B_BYPASS_PROB_SHIFT)-1u)) == 0u )) {
        return LLC_WAYS; // bypass
    }

    // Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (!current_set[w].valid) return w;

    // Find maxRRPV in rrpv_B
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    // Age all and retry
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    return 0;
}

// Epoch decay hook (selector tick)
static inline void selector_decay_epoch(){
    // Decay Mode B PC tables
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_freq[i] >>= 1;
        mode_B_pc_dead[i] >>= 1;
        mode_B_pc_run_len[i] >>= 1;
        mode_B_pc_hitstreak[i] >>= 1;
    }
    // Light decay for Hawkeye-lite SHCT
    for(uint32_t i=0;i<SHCT_ENTRIES;i++){
        SHCT_demand[i] >>= 1;
        SHCT_prefetch[i] >>= 1;
    }
}

// ============================================================================
// CHAMPSIM INTERFACE
// ============================================================================
void InitReplacementState()
{
    for (uint32_t i=0; i<LLC_SETS; i++) {
        for (uint32_t j=0; j<LLC_WAYS; j++) {
            rrpv_A[i][j] = maxRRPV;
            rrpv_B[i][j] = maxRRPV;
            mode_B_seen_once[i][j] = 0;
            mode_B_stream_tag[i][j] = 0;
        }
        mode_confidence[i] = 0;
        follower_is_B[i] = 0;
    }
    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;

    for(uint32_t i=0;i<SHCT_ENTRIES;i++){ SHCT_demand[i]=0; SHCT_prefetch[i]=0; }

    for (uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_last_set8[i] = 0xFF;
        mode_B_pc_run_len[i] = 0;
        mode_B_pc_hitstreak[i] = 0;
    }
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{
    (void)cpu;
    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        uint32_t v = mode_B_victim(set, current_set, PC, paddr, type);
        return v;
    } else {
        // Mode A: Hawkeye-lite victim selection (uses rrpv_A exclusively)
        // Prefer invalid
        for (uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
        // Look for maxRRPV
        for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_A[set][w] == maxRRPV) return w;
        // Age and retry
        for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
        for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_A[set][w] == maxRRPV) return w;
        return 0;
    }
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr,
                             uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6; // line-align

    // Ignore writebacks
    if (is_writeback(type))
        return;

    // Update selector on this observation
    update_selector(set, hit);

    // Always train Hawkeye-lite SHCT (for stable default path)
    hawkeye_train(PC, is_prefetch(type), hit != 0);

    bool use_mode_B = should_use_mode_B(set);

    // EXCLUSIVE UPDATE: Apply only the chosen mode's policy
    if (use_mode_B) {
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A: insertion and light aging (updates ONLY rrpv_A)
        bool friendly = hawkeye_is_friendly(PC, is_prefetch(type));
        if (!hit) {
            // Fill path
            rrpv_A[set][way] = friendly ? 0 : maxRRPV; // friendly MRU, averse tail
        } else {
            // On hit, gently protect friendly lines
            if (friendly && rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}