/*
 * Hawkeye + CrossRun-DeadGate (CR-DG) Lite Ensemble
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; never update both on the same access.
 *
 * Mode A: Hawkeye-lite (SRRIP + tiny SHCT)
 * Mode B: CR-DG Lite (±1/±2 stride/run detector + TinyLFU + stream quarantine + multi-hit rescue)
 *
 * Selector: Very conservative gating with B-leader safety
 *   - MODE_A_BIAS=50 (prefer Hawkeye)
 *   - MODE_B_THRESHOLD=7 (strict per-set threshold)
 *   - MAX_MODE_B_FOLLOWERS=256 (~12.5%)
 *   - SELECTOR_EPOCH_SIZE=4096
 *
 * Tunables (knobs):
 *   - Mode B HOT_THRESHOLD (TinyLFU 4-bit): 8
 *   - Mode B STREAM_ARM (consecutive ±1/±2): 2
 *   - Mode B INSERT_WARM/INSERT_COLD: 2/6
 *   - Mode B STREAM_TAIL: 7
 *   - Mode B PROMOTE_NS/PROMOTE_STR (hits to MRU): 2/2
 *   - Mode B DEMOTE_TOUCH (quarantine demotion on touch): 1
 *
 * Telemetry (lightweight):
 *   - Leader A/B scores per epoch
 *   - Per-set confidence drifts toward global preference each epoch
 *   - Optional epochal TinyLFU decay hook (commented)
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define maxRRPV 7

// ChampSim access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// -----------------------------------------------------------------------------
// Leader-follower sampling (64 sampled sets: low 6 bits == high 6 bits)
// -----------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// CRC-ish hash for PC indexing
static inline uint32_t fast_hash(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// -----------------------------------------------------------------------------
// Selector (very conservative gating + B-leader safety)
// -----------------------------------------------------------------------------
static int32_t  leaderA_score      = 0;
static int32_t  leaderB_score      = 0;
static bool     prefer_mode_B      = false;
static uint64_t selector_epoch     = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096; // knob
static constexpr int32_t  MODE_A_BIAS         = 50;   // knob
static constexpr int8_t   MODE_B_THRESHOLD    = 7;    // knob

// Per-set confidence: -8..+7 (stored directly)
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256; // knob
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Selector decision (with B-leader safety)
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;                   // A leaders always A
    if(LEADER_B(set)) return prefer_mode_B;           // B leaders only B when global prefers B
    if(!prefer_mode_B) return false;                  // default Hawkeye unless B decisively wins
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false; // local threshold
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

static inline void update_selector_on_access(uint32_t set, uint8_t hit){
    // Track leader scores
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    // Epoch roll
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // Drift local confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){
                if(mode_confidence[s] < 7) mode_confidence[s]++;
            }else{
                if(mode_confidence[s] > -8) mode_confidence[s]--;
            }
        }
        // Optional TinyLFU global decay hook (disabled by default)
        // for(uint32_t i=0;i<MODEB_PC_SIZE;i++) modeB_pc_use4[i] >>= 1;

        leaderA_score = 0;
        leaderB_score = 0;
    }
}

// -----------------------------------------------------------------------------
// Mode A: Hawkeye-lite (EXCLUSIVE rrpv_A + tiny SHCT friendly/averse)
//   - Not full OPTgen; PC-friendly classifier steers insertion depth
// -----------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

// Tiny SHCT (512 entries, 5-bit counters in uint8_t)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES];
static inline uint32_t shct_idx(uint64_t PC){ return fast_hash(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return SHCT_cnt[shct_idx(PC)] >= 16; }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    }
    // Fallback (should not happen)
    uint32_t victim = 0; uint8_t best=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_A[set][w] >= best){ best=rrpv_A[set][w]; victim=w; } }
    return victim;
}

static void mode_A_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    (void)paddr;
    if(is_writeback(type)) return;

    if(hit){
        // Promote on hit
        if(rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        if(is_demand(type)) hawkeye_train(PC, true);
        return;
    }

    // Miss/fill: set insertion based on friendly/averse
    bool friendly = hawkeye_is_friendly(PC);
    uint8_t ins = friendly ? 2 : 6; // near-MRU vs near-LRU
    rrpv_A[set][way] = ins;
    if(is_demand(type)) hawkeye_train(PC, false);
}

// -----------------------------------------------------------------------------
// Mode B: CR-DG Lite (EXCLUSIVE rrpv_B + per-line hitcnt and stream_tag)
//   - ±1/±2 stride/run detector per-PC with 2-bit confidence
//   - TinyLFU 4-bit per-PC
//   - Hard-tail quarantine for streams/prefetches
//   - Multi-hit rescue to MRU on 2nd demand hit; light promote on 1st hit
// -----------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];       // 3-bit logical
static uint8_t modeB_hitcnt[LLC_SETS][LLC_WAYS]; // 2-bit logical (0..3)
static uint8_t modeB_stream[LLC_SETS][LLC_WAYS]; // 1-bit logical

// PC table (512 entries)
#define MODEB_PC_SIZE 512
static inline uint32_t modeB_pc_idx(uint64_t PC){ return fast_hash(PC) & (MODEB_PC_SIZE-1); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t modeB_pc_last_line10[MODEB_PC_SIZE]; // 0xFFFF means invalid
static uint8_t  modeB_pc_stream_conf[MODEB_PC_SIZE]; // 2-bit (0..3)
static uint8_t  modeB_pc_use4[MODEB_PC_SIZE];        // 4-bit (0..15)

// Tunables (Mode B)
static constexpr uint8_t MODEB_INSERT_WARM   = 2; // near-MRU
static constexpr uint8_t MODEB_INSERT_COLD   = 6; // near-LRU
static constexpr uint8_t MODEB_STREAM_TAIL   = 7; // hard tail for streams
static constexpr uint8_t MODEB_STREAM_ARM    = 2; // need 2 consecutive ±1/±2
static constexpr uint8_t MODEB_HOT_THRESHOLD = 8; // TinyLFU hot bound
static constexpr uint8_t MODEB_PROMOTE_NS    = 2; // non-stream rescue
static constexpr uint8_t MODEB_PROMOTE_STR   = 2; // stream rescue
static constexpr uint8_t MODEB_DEMOTE_TOUCH  = 1; // quarantine demotion

static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }

static inline bool modeB_detect_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = modeB_pc_idx(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = modeB_pc_last_line10[idx];
    bool stride = false;

    if(last != 0xFFFFu){
        // Bidirectional ±1/±2
        uint16_t e1 = (uint16_t)(last + 1);
        uint16_t e2 = (uint16_t)(last + 2);
        uint16_t e3 = (uint16_t)(last - 1);
        uint16_t e4 = (uint16_t)(last - 2);
        stride = (ln==e1)||(ln==e2)||(ln==e3)||(ln==e4);
    }
    if(stride) sat_inc_u2(modeB_pc_stream_conf[idx]);
    else modeB_pc_stream_conf[idx] = 0;

    modeB_pc_last_line10[idx] = ln;
    return (modeB_pc_stream_conf[idx] >= MODEB_STREAM_ARM);
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    }
    // Fallback
    uint32_t victim = 0; uint8_t best=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_B[set][w] >= best){ best=rrpv_B[set][w]; victim=w; } }
    return victim;
}

static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pidx = modeB_pc_idx(PC);
    if(is_demand(type)) sat_inc_u4(modeB_pc_use4[pidx]);

    bool stream_armed = is_demand(type) ? modeB_detect_stream(PC, paddr) : false;

    if(hit){
        // Demote quarantined on touch to keep evictable (stream aging)
        if(modeB_stream[set][way] && rrpv_B[set][way] < maxRRPV){
            uint8_t nv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + MODEB_DEMOTE_TOUCH);
            rrpv_B[set][way] = nv;
        }

        // Multi-hit rescue (demand only)
        if(is_demand(type)){
            if(modeB_hitcnt[set][way] < 3) modeB_hitcnt[set][way]++;
            bool is_stream_line = (modeB_stream[set][way] != 0);
            uint8_t need = is_stream_line ? MODEB_PROMOTE_STR : MODEB_PROMOTE_NS;

            if(modeB_hitcnt[set][way] >= need){
                modeB_stream[set][way] = 0; // escape quarantine
                rrpv_B[set][way] = 0;       // MRU
            }else{
                if(rrpv_B[set][way] > 0) rrpv_B[set][way]--; // light promote
            }
        }
        return;
    }

    // Miss + fill
    bool hot_pc = (modeB_pc_use4[pidx] >= MODEB_HOT_THRESHOLD);
    bool quarantine = is_prefetch(type) || stream_armed;

    if(quarantine){
        rrpv_B[set][way] = MODEB_STREAM_TAIL;
        modeB_stream[set][way] = 1;
    }else{
        rrpv_B[set][way] = hot_pc ? MODEB_INSERT_WARM : MODEB_INSERT_COLD;
        modeB_stream[set][way] = 0;
    }
    modeB_hitcnt[set][way] = 0;
}

// -----------------------------------------------------------------------------
// ChampSim interface
// -----------------------------------------------------------------------------
void InitReplacementState(){
    // Mode A init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++) rrpv_A[s][w] = maxRRPV;
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_B[s][w]      = maxRRPV;
            modeB_hitcnt[s][w]= 0;
            modeB_stream[s][w]= 0;
        }
    }
    for(uint32_t i=0;i<MODEB_PC_SIZE;i++){
        modeB_pc_last_line10[i] = 0xFFFFu;
        modeB_pc_stream_conf[i] = 0;
        modeB_pc_use4[i]        = 0;
    }

    // Selector init
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
    std::memset(follower_bits, 0, sizeof(follower_bits));
    for(uint32_t s=0;s<LLC_SETS;s++) mode_confidence[s] = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_B = should_use_mode_B(set);
    if(use_B) return mode_B_victim(set, current_set);
    else      return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;

    // Update selector scores/epochs first
    update_selector_on_access(set, hit);

    // Decide mode with B-leader safety at the time of update
    bool use_B = should_use_mode_B(set);

    // Exclusive updates: only one mode per access
    if(use_B) mode_B_update(set, way, paddr, PC, type, hit);
    else      mode_A_update(set, way, paddr, PC, type, hit);
}

void PrintStats_Heartbeat(){}
void PrintStats(){}