/*
 * Hawkeye + CrossRun-TinyFence (CR-TF) Strict Ensemble
 *
 * GOALS
 * - Never worse than Hawkeye: very conservative selector (high bias/threshold, small follower cap)
 * - Boost stream/scan phases: ±1/±2 run detector + quarantine, prefetch hard-tail
 * - Avoid noisy fills: TinyLFU + 2-bit dead-gate
 * - Allow short reuse: multi-hit rescue (stream 3 hits, non-stream 2 hits)
 *
 * SELECTOR TUNABLES (very conservative by default)
 * - SELECTOR_EPOCH_SIZE = 4096
 * - MODE_A_BIAS         = 48  (global preference needs big win to enable Mode B)
 * - MODE_B_THRESHOLD    = 6   (per-set local threshold)
 * - MAX_MODE_B_FOLLOWERS= 256 (~12.5% of sets)
 *
 * MODE B TUNABLES
 * - INSERT_WARM   = 2  (hot)
 * - INSERT_COLD   = 6  (cold)
 * - STREAM_TAIL   = 7  (quarantine tail)
 * - STREAM_ARM    = 2  (consecutive ±1/±2 steps to arm stream)
 * - HOT_THRESHOLD = 8  (TinyLFU hot)
 * - DEAD_MAX      = 3  (2-bit dead fence; promote only if < DEAD_MAX)
 * - PROMOTE_NS    = 2  (non-stream rescue hits)
 * - PROMOTE_STR   = 3  (stream rescue hits)
 * - DEMOTE_TOUCH  = 1  (quarantine demotion on touch)
 *
 * TELEMETRY (decayed each selector epoch)
 * - TinyLFU pc_use>>=1, dead_gate>>=1; leader scores reset; confidence drifts toward prefer_mode_B
 *
 * STORAGE NOTE
 * - Arrays are byte-addressed here for clarity. In hardware, pack bits to meet the budget:
 *   Mode A rrpv_A: 3b/line; Mode B per-line: 6b (rrpv_B 3b + hitcnt 2b + stream 1b).
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return t==ACCESS_LOAD || t==ACCESS_RFO; }
static inline bool is_prefetch(uint32_t t){ return t==ACCESS_PREFETCH; }
static inline bool is_writeback(uint32_t t){ return t==ACCESS_WRITEBACK; }

// Sample 64 sets: low 6 bits == high 6 bits
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// ---------------------------- Selector (conservative) ------------------------
static int32_t  leaderA_score = 0, leaderB_score = 0;
static bool     prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 48; // default to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 6;  // local per-set threshold

// Per-set confidence: -8..+7 stored as int8_t
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Decide Mode B usage (B-leader safety: B leaders only switch if prefer_mode_B==true)
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
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

static inline void selector_epoch_tick(){
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Global preference
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // Drift per-set confidence toward global
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 7) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -8) mode_confidence[s]--; }
        }
        leaderA_score = 0; leaderB_score = 0;
    }
}

// ---------------------------- Mode A: Hawkeye-lite ---------------------------
// Budgeted Hawkeye-style friendly/averse classifier (tiny SHCT)
#define maxRRPV 7
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS];

#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31 (5b logical)
static inline uint32_t shct_idx(uint64_t PC){ // tiny hash
    uint64_t x = PC ^ (PC>>17) ^ (PC<<13);
    return (uint32_t)x & (SHCT_ENTRIES-1);
}
static inline bool hawkeye_is_friendly(uint64_t PC){ return SHCT_cnt[shct_idx(PC)] >= 16; }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // Find invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    // Standard RRIP victim search
    for(;;){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    }
}

static inline void mode_A_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    (void)paddr;
    if(is_writeback(type)) return;

    // Train tiny SHCT on outcome
    hawkeye_train(PC, hit != 0);

    if(hit){
        if(rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        return;
    }

    // Miss+fill insertion (friendly near-MRU, averse near-tail; prefetch quarantined)
    if(is_prefetch(type)){
        rrpv_A[set][way] = maxRRPV; // bypass-like quarantine
    }else{
        rrpv_A[set][way] = hawkeye_is_friendly(PC) ? 2 : 6;
    }
}

// ----------------------- Mode B: CrossRun-TinyFence --------------------------
// Per-line (pack to 6b/line in HW): rrpv_B (3b), hitcnt2 (2b), stream_tag (1b)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt_B[LLC_SETS][LLC_WAYS];
static uint8_t stream_B[LLC_SETS][LLC_WAYS];

// PC table: 512 entries (pack to ~18b/entry in HW)
#define MODEB_PC 512
static uint16_t pc_last_line10[MODEB_PC]; // 10b valid; 0xFFFF means invalid
static uint8_t  pc_stream_conf[MODEB_PC]; // 2b (0..3)
static uint8_t  pc_use4[MODEB_PC];        // 4b TinyLFU (0..15)
static uint8_t  pc_dead2[MODEB_PC];       // 2b dead-gate (0..3)

static inline uint32_t pc_idx(uint64_t PC){ return (uint32_t)((PC ^ (PC>>7) ^ (PC>>17)) & (MODEB_PC-1)); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FFu); }

// Tunables (fixed for this drop)
static constexpr uint8_t INSERT_WARM   = 2;
static constexpr uint8_t INSERT_COLD   = 6;
static constexpr uint8_t STREAM_TAIL   = 7;
static constexpr uint8_t STREAM_ARM    = 2;
static constexpr uint8_t HOT_THRESHOLD = 8;
static constexpr uint8_t DEAD_MAX      = 3;
static constexpr uint8_t PROMOTE_NS    = 2;
static constexpr uint8_t PROMOTE_STR   = 3;
static constexpr uint8_t DEMOTE_TOUCH  = 1;

static inline void sat_inc_u2(uint8_t &x, uint8_t maxv){ if(x<maxv) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }

// Stride/run detector: ±1/±2, arms after 2 consecutive forward/back steps
static inline bool detect_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = pc_idx(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool step = false;

    if(last != 0xFFFFu){
        uint16_t d = (ln > last) ? (ln - last) : (last - ln);
        step = (d == 1) || (d == 2);
    }

    if(step) sat_inc_u2(pc_stream_conf[idx], 3);
    else pc_stream_conf[idx] = 0;

    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM);
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(;;){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    }
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pidx = pc_idx(PC);
    if(is_demand(type)) sat_inc_u4(pc_use4[pidx]); // TinyLFU demand count

    if(hit){
        // Quarantine demotion on touch (keeps streams evictable until proven)
        if(stream_B[set][way] && rrpv_B[set][way] < maxRRPV){
            uint8_t nv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + DEMOTE_TOUCH);
            rrpv_B[set][way] = nv;
        }

        // Multi-hit rescue
        if(hitcnt_B[set][way] < 3) hitcnt_B[set][way]++;
        bool is_stream_line = (stream_B[set][way] != 0);
        uint8_t need = is_stream_line ? PROMOTE_STR : PROMOTE_NS;

        if(hitcnt_B[set][way] >= need){
            stream_B[set][way] = 0;         // escape quarantine if any
            rrpv_B[set][way] = 0;           // promote to MRU
            if(pc_dead2[pidx] > 0) pc_dead2[pidx]--; // this PC proved reuse
        }else{
            if(rrpv_B[set][way] > 0) rrpv_B[set][way]--; // light promotion
        }
        return;
    }

    // Miss + fill
    bool stream_armed = is_demand(type) ? detect_stream(PC, paddr) : false;
    bool hot_pc = (pc_use4[pidx] >= HOT_THRESHOLD);
    bool deadish = (pc_dead2[pidx] >= DEAD_MAX);

    if(is_prefetch(type) || stream_armed){
        // quarantine streams and prefetches
        rrpv_B[set][way] = STREAM_TAIL;
        stream_B[set][way] = 1;
        hitcnt_B[set][way] = 0;
    }else{
        // hot gated warm insert unless dead-fenced
        uint8_t ins = (hot_pc && !deadish) ? INSERT_WARM : INSERT_COLD;
        rrpv_B[set][way] = ins;
        stream_B[set][way] = 0;
        hitcnt_B[set][way] = 0;
    }

    // If the last fill for this PC didn't see reuse, strengthen dead gate
    // (approximate: count fills as dead until a rescue occurs)
    if(!hot_pc) sat_inc_u2(pc_dead2[pidx], DEAD_MAX);
}

// ------------------------------ ChampSim API --------------------------------
void InitReplacementState(){
    // Mode A
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++) rrpv_A[s][w] = maxRRPV;
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_B[s][w] = maxRRPV;
            hitcnt_B[s][w] = 0;
            stream_B[s][w] = 0;
        }
    }
    for(uint32_t i=0; i<MODEB_PC; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_dead2[i] = 0;
    }

    // Selector
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    num_mode_B_followers = 0;
    std::memset(follower_bits, 0, sizeof(follower_bits));
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_B = should_use_mode_B(set);
    return use_B ? mode_B_victim(set, current_set) : mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Leader forcing with B-leader safety
    bool use_B = should_use_mode_B(set);
    if(LEADER_A(set)) use_B = false;
    if(LEADER_B(set)) use_B = prefer_mode_B;

    // Train leader scores
    if(SAMPLED_SET(set)){
        int delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    // Exclusive mode update
    if(use_B) mode_B_update(set, way, paddr, PC, type, hit);
    else      mode_A_update(set, way, paddr, PC, type, hit);

    // Epochic selector + Mode B decays
    selector_epoch_tick();
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Decay TinyLFU and dead gates to adapt across phases
        for(uint32_t i=0; i<MODEB_PC; i++){
            pc_use4[i] >>= 1;
            pc_dead2[i] >>= 1;
            // pc_stream_conf naturally resets by detector on misses
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}