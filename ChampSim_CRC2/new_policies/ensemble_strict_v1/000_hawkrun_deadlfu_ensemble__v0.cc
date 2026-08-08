/*
 * HawkRun-DeadLFU Ensemble (Hawkeye default + CrossRun-DeadLFU Mode B)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; never update both on a single access.
 *
 * Goals:
 *  - Default to Hawkeye (Mode A) for irregular/mixed reuse.
 *  - Enable Mode B on scan-heavy phases: ±1/±2 run detection + TinyLFU(4b) + 2b dead-gate.
 *  - Quarantine streams/prefetches at hard tail; allow bypass only if long-run && cold && dead.
 *  - Multi-hit rescue: promote on 2nd (non-stream) / 3rd (stream) demand hit.
 *  - Conservative selector: high bias to Hawkeye, strict local threshold, 12.5% B follower cap.
 *
 * Telemetry / Tunables (adjust in-place):
 *  - Selector: MODE_A_BIAS=50, MODE_B_THRESHOLD=7, MAX_MODE_B_FOLLOWERS=256, SELECTOR_EPOCH_SIZE=4096
 *  - Mode B: HOT_THRESHOLD=8, DEAD_THRESHOLD=2, STREAM_ARM=2, LONGRUN_UNIQUESETS=6
 *            INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7, PROMOTE_NS=2, PROMOTE_STR=3, DEMOTE_TOUCH=1
 *  - Epoch decay (each selector epoch): pc_use4>>=1; pc_dead2>>=1; pc_stream_conf dec; pc_setmask=0
 *
 * ChampSim integration: InitReplacementState, GetVictimInSet, UpdateReplacementState, PrintStats.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

#define maxRRPV 7

//------------------------------------------------------------------------------
// Sampling: 64 leader sets (low 6 bits == high 6 bits). Even=A, odd=B.
//------------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); }

// Simple 32-bit hash for PC/signatures
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

//------------------------------------------------------------------------------
// Selector (conservative gating + per-set confidence + follower cap)
//------------------------------------------------------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // strong Hawkeye bias
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict local threshold

// Per-set confidence in Mode B: -8..+7 (int8_t)
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5% of sets)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // bitset: 1 if set is permanently assigned to Mode B (follower)

static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){
    if(!follower_is_B(set)){
        follower_bits[set>>3] |= (uint8_t)(1u << (set & 7));
        num_mode_B_followers++;
    }
}

// Update selector state using observed hit/miss on the active mode in leader sets.
// Also handles epoch rollovers: global preference, per-set drift, and PC-table decays.
static inline void update_selector_train(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        leaderA_score = 0; leaderB_score = 0;

        // Drift local per-set confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){
                if(mode_confidence[s] < 7) mode_confidence[s]++;
            }else{
                if(mode_confidence[s] > -8) mode_confidence[s]--;
            }
        }

        // Epochic decay of PC stats (Mode B)
        for(uint32_t i=0;i<512;i++){
            modeB_pc_use4[i] >>= 1;
            modeB_pc_dead2[i] >>= 1;
            if(modeB_pc_stream_conf[i] > 0) modeB_pc_stream_conf[i]--;
            modeB_pc_setmask[i] = 0; // new run window
        }
    }
}

// Decide if a set should use Mode B. B-leaders always duel (to let Mode B prove itself).
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true; // B leaders duel to build evidence

    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
    }
    return true;
}

//------------------------------------------------------------------------------
// Mode A: Hawkeye-lite (exclusive rrpv_A + tiny SHCT + leader-only sampler)
//------------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

// Tiny SHCT (512 entries, 5-bit counters; friendly if >=16)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES];
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return SHCT_cnt[shct_idx(PC)] >= 16; }
static inline void hawkeye_train(uint64_t PC, bool positive){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(positive){ if(c<31) c++; } else { if(c>0) c--; }
}

// Leader-only sampler metadata: 12-bit PC signature + used bit per line
static uint16_t hawk_sig12[64][LLC_WAYS];
static uint8_t  hawk_used[64][LLC_WAYS]; // 0=dead, 1=reused

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

static void mode_A_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    (void)paddr;

    if(hit){
        if(rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        if(SAMPLED_SET(set)){ hawk_used[LEADER_SLOT(set)][way] = 1; }
        // reward PC on reuse
        if(is_demand(type)) hawkeye_train(PC, true);
        return;
    }

    // Miss + fill training for leaders: penalize dead-on-fill, reward reused lines
    if(SAMPLED_SET(set)){
        uint32_t slot = LEADER_SLOT(set);
        // At fill, the "way" is the victim being replaced
        uint16_t old_sig = hawk_sig12[slot][way];
        if(hawk_used[slot][way] == 0) hawkeye_train(old_sig, false);
        else                          hawkeye_train(old_sig, true);
        // Record new sig and reset used
        hawk_sig12[slot][way] = (uint16_t)(CRC32(PC) & 0x0FFFu);
        hawk_used[slot][way]  = 0;
    }

    // Insertion policy
    if(is_prefetch(type)){
        rrpv_A[set][way] = 7; // quarantine prefetches
    }else{
        bool friendly = hawkeye_is_friendly(PC);
        rrpv_A[set][way] = friendly ? 2 : 6;
    }
}

//------------------------------------------------------------------------------
// Mode B: CrossRun-DeadLFU (exclusive rrpv_B + hitcnt + stream_tag + tiny PC table)
//------------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];      // 3-bit logical
static uint8_t modeB_hitcnt[LLC_SETS][LLC_WAYS];// 2-bit (0..3)
static uint8_t modeB_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit
static uint8_t modeB_pc_sig8[LLC_SETS][LLC_WAYS];    // compact per-line PC sig for dead-gate training

// PC table (512 entries): last_line10, stream_conf(2b), use4(4b), dead2(2b), setmask(8b)
#define MODEB_PC_ENTRIES 512
static inline uint32_t pc_idx(uint64_t PC){ return CRC32(PC) & (MODEB_PC_ENTRIES-1); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t modeB_pc_last_line10[MODEB_PC_ENTRIES];
static uint8_t  modeB_pc_stream_conf[MODEB_PC_ENTRIES]; // 0..3
static uint8_t  modeB_pc_use4[MODEB_PC_ENTRIES];        // 0..15
static uint8_t  modeB_pc_dead2[MODEB_PC_ENTRIES];       // 0..3
static uint8_t  modeB_pc_setmask[MODEB_PC_ENTRIES];     // 8-bit unique-set window

// Tunables (Mode B)
static constexpr uint8_t MODEB_INSERT_WARM    = 2;
static constexpr uint8_t MODEB_INSERT_COLD    = 6;
static constexpr uint8_t MODEB_STREAM_TAIL    = 7;
static constexpr uint8_t MODEB_STREAM_ARM     = 2;  // need 2 stride steps to arm
static constexpr uint8_t MODEB_HOT_THRESHOLD  = 8;  // use4 >= 8 => hot PC
static constexpr uint8_t MODEB_DEAD_THRESHOLD = 2;  // dead2 >= 2 => deadish PC
static constexpr uint8_t MODEB_LONGRUN_UNIQ   = 6;  // unique-set span threshold
static constexpr uint8_t MODEB_PROMOTE_NS     = 2;  // non-stream rescue on 2nd hit
static constexpr uint8_t MODEB_PROMOTE_STR    = 3;  // stream rescue on 3rd hit
static constexpr uint8_t MODEB_DEMOTE_TOUCH   = 1;  // demote quarantined on touch

static inline uint8_t popcount8(uint8_t x){
#if defined(__GNUG__) || defined(__clang__)
    return (uint8_t)__builtin_popcount((unsigned)x);
#else
    // portable
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (uint8_t)((x + (x >> 4)) & 0x0F);
#endif
}

// ±1/±2 run detection + unique-set span build-up
static inline bool modeB_detect_stream_and_track(uint64_t PC, uint64_t paddr){
    uint32_t idx = pc_idx(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = modeB_pc_last_line10[idx];
    bool stride12 = false;

    if(last != 0xFFFFu){
        int16_t d = (int16_t)ln - (int16_t)last;
        if(d == 1 || d == 2 || d == -1 || d == -2) stride12 = true;
    }

    if(stride12){
        if(modeB_pc_stream_conf[idx] < 3) modeB_pc_stream_conf[idx]++;
    }else{
        modeB_pc_stream_conf[idx] = 0;
        modeB_pc_setmask[idx] = 0;
    }

    // update unique-set mask (8-way coarse)
    uint32_t set = (uint32_t)((paddr >> 6) % LLC_SETS);
    uint8_t bit = (uint8_t)(1u << (set & 7));
    modeB_pc_setmask[idx] |= bit;

    modeB_pc_last_line10[idx] = ln;
    return (modeB_pc_stream_conf[idx] >= MODEB_STREAM_ARM);
}

static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    }
    // fallback
    uint8_t best=0; uint32_t vic=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_B[set][w] >= best){ best=rrpv_B[set][w]; vic=w; } }
    return vic;
}

// Bypass decision (checked in GetVictimInSet before allocating a victim)
static inline bool mode_B_should_bypass(uint64_t PC, uint64_t paddr, uint32_t type){
    if(!is_demand(type)) return false; // never bypass writebacks/prefetch (prefetch quarantined)
    uint32_t idx = pc_idx(PC);
    bool run = modeB_detect_stream_and_track(PC, paddr);
    uint8_t uniq = popcount8(modeB_pc_setmask[idx]);
    bool longrun = run && (uniq >= MODEB_LONGRUN_UNIQ);
    bool cold = (modeB_pc_use4[idx] < MODEB_HOT_THRESHOLD);
    bool deadish = (modeB_pc_dead2[idx] >= MODEB_DEAD_THRESHOLD);
    return (longrun && cold && deadish);
}

static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t idx = pc_idx(PC);

    // Update TinyLFU on demand
    if(is_demand(type) && modeB_pc_use4[idx] < 15) modeB_pc_use4[idx]++;

    // Stream detection for this access (used for insertion/quarantine)
    bool stream_armed = is_demand(type) ? modeB_detect_stream_and_track(PC, paddr) : false;

    if(hit){
        // Quarantine demotion on touch
        if(modeB_stream_tag[set][way] && rrpv_B[set][way] < maxRRPV){
            uint8_t nv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + MODEB_DEMOTE_TOUCH);
            rrpv_B[set][way] = nv;
        }

        // Multi-hit rescue
        if(is_demand(type)){
            if(modeB_hitcnt[set][way] < 3) modeB_hitcnt[set][way]++;
            bool is_stream_line = (modeB_stream_tag[set][way] != 0);
            uint8_t need = is_stream_line ? MODEB_PROMOTE_STR : MODEB_PROMOTE_NS;
            if(modeB_hitcnt[set][way] >= need){
                modeB_stream_tag[set][way] = 0; // escape quarantine if any
                rrpv_B[set][way] = 0;          // MRU
            }else{
                if(rrpv_B[set][way] > 0) rrpv_B[set][way]--;
            }
        }
        return;
    }

    // Miss + fill: train dead-gate on victim (was it reused?)
    // Use hitcnt>0 as "used" proxy; map to PC via stored pc_sig8.
    uint8_t old_sig = modeB_pc_sig8[set][way];
    uint32_t old_idx = old_sig; // 8-bit maps into first 256 entries; mix to 512 via XOR
    old_idx ^= (old_idx << 1) & (MODEB_PC_ENTRIES-1);
    if(modeB_hitcnt[set][way] == 0){
        if(modeB_pc_dead2[old_idx] < 3) modeB_pc_dead2[old_idx]++; // dead on fill
    }else{
        if(modeB_pc_dead2[old_idx] > 0) modeB_pc_dead2[old_idx]--;
    }

    // Insert new line
    bool quarantine = is_prefetch(type) || stream_armed;
    if(quarantine){
        rrpv_B[set][way] = MODEB_STREAM_TAIL;
        modeB_stream_tag[set][way] = 1;
    }else{
        bool hot = (modeB_pc_use4[idx] >= MODEB_HOT_THRESHOLD);
        rrpv_B[set][way] = hot ? MODEB_INSERT_WARM : MODEB_INSERT_COLD;
        modeB_stream_tag[set][way] = 0;
    }
    modeB_hitcnt[set][way] = 0;
    modeB_pc_sig8[set][way] = (uint8_t)(CRC32(PC) & 0xFFu);
}

//------------------------------------------------------------------------------
// Entry points
//------------------------------------------------------------------------------
void InitReplacementState(){
    // Mode A
    for(uint32_t s=0;s<LLC_SETS;s++){
        for(uint32_t w=0;w<LLC_WAYS;w++){
            rrpv_A[s][w] = maxRRPV;
        }
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));
    std::memset(hawk_sig12, 0, sizeof(hawk_sig12));
    std::memset(hawk_used,  0, sizeof(hawk_used));

    // Mode B
    for(uint32_t s=0;s<LLC_SETS;s++){
        for(uint32_t w=0;w<LLC_WAYS;w++){
            rrpv_B[s][w] = maxRRPV;
            modeB_hitcnt[s][w] = 0;
            modeB_stream_tag[s][w] = 0;
            modeB_pc_sig8[s][w] = 0;
        }
    }
    for(uint32_t i=0;i<MODEB_PC_ENTRIES;i++){
        modeB_pc_last_line10[i] = 0xFFFFu;
        modeB_pc_stream_conf[i] = 0;
        modeB_pc_use4[i] = 0;
        modeB_pc_dead2[i] = 0;
        modeB_pc_setmask[i] = 0;
    }

    // Selector
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    std::memset(follower_bits, 0, sizeof(follower_bits));
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;

    bool use_B = should_use_mode_B(set);
    if(use_B){
        // Optional bypass if long-run && cold && deadish
        if(mode_B_should_bypass(PC, paddr, type)) return LLC_WAYS; // bypass
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

    bool use_B = should_use_mode_B(set);

    // Train selector on observed outcome within leaders
    update_selector_train(set, hit);

    if(use_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    }else{
        mode_A_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}