/*
 * Hawkeye + CrossRun-DeadGate (CR-DG) Tight-Gate Ensemble
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays or metadata.
 *
 * Goals:
 *   - Never worse than Hawkeye: high Hawkeye bias, strict per-set threshold, B-leader safety.
 *   - Target scans/streams: run detector (±1/±2), stream quarantine, optional bypass on long runs.
 *   - Rescue reuse: multi-hit promotion pathway escapes quarantine on genuine reuse.
 *   - Fence noise: TinyLFU hot/cold, 2-bit dead-gate per PC.
 *   - Storage-constrained: ~30 KB total (packed bit estimates).
 *
 * Tunables (telemetry in comments):
 *   Selector: MODE_A_BIAS=55, MODE_B_THRESHOLD=7, SELECTOR_EPOCH_SIZE=4096, MAX_MODE_B_FOLLOWERS=256
 *   Mode A (Hawkeye-lite): SHCT_ENTRIES=512, friendly>=16, INSERT_FRIENDLY=2, INSERT_VERSE=6
 *   Mode B (CR-DG):
 *     STREAM_ARM=2, LONGRUN_UNIQUESETS=6, HOT_THRESHOLD=8, DEAD_THRESHOLD=2,
 *     INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7, PROMOTE_NS=2, PROMOTE_STR=2, DEMOTE_TOUCH=1,
 *     BYPASS_PROB≈50% via per-PC LFSR bit when (longrun && cold && dead && stream_armed && !prefetch)
 *
 * Storage breakdown (packed):
 *   - Mode A per-line rrpv_A: 3b/line  -> ~12 KB
 *   - Mode B per-line: rrpv_B 3b + hitcnt 2b + stream_tag 1b -> 6b/line -> ~24 KB (never co-active per access)
 *     (Note: arrays are exclusive by mode; accounted per active mode path)
 *   - PC tables (Mode B): 512 entries × (last_line10 10b + stream_conf 2b + use4 4b + dead2 2b + runmask 8b + lfsr 1b) ≈ ~1.3 KB
 *   - Selector state (nibbles + bitsets): ~1.5 KB
 *   Total active metadata ≤ ~30 KB (packed). Code uses bytes; treat as conceptual bit packing.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>

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

//------------------------------------------------------------------------------
// Sampling: 64 sampled sets (low 6 bits == high 6 bits)
//------------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

//------------------------------------------------------------------------------
// Selector (tight gating + B-leader safety)
//   - Global prefer_mode_B flips only when B >> A by MODE_A_BIAS
//   - Per-set signed nibble confidence must exceed MODE_B_THRESHOLD
//   - Follower cap limits exposure; B leaders only use Mode B if prefer_mode_B==true
//------------------------------------------------------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

// Tunables
static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 55;  // tighter bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict local threshold
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;

// Per-set confidence stored as 4-bit signed nibbles (-8..+7) with +8 bias
static uint8_t mode_conf_nibbles[LLC_SETS/2];
static inline int8_t get_mode_conf(uint32_t set){
    uint8_t b = mode_conf_nibbles[set>>1];
    uint8_t nib = (set & 1u) ? (b>>4) : (b & 0xF);
    return (int8_t)nib - 8;
}
static inline void set_mode_conf(uint32_t set, int8_t val){
    if(val > 7) val = 7;
    if(val < -8) val = -8;
    uint8_t enc = (uint8_t)(val + 8) & 0xF;
    uint8_t &b = mode_conf_nibbles[set>>1];
    if(set & 1u) b = (uint8_t)((b & 0x0F) | (enc<<4));
    else         b = (uint8_t)((b & 0xF0) | enc);
}

// Follower cap and assignment bitset
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8];
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Selector update called on each access
static inline void selector_on_access(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Global preference
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Nudge local per-set confidence toward global pref
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            int8_t c = get_mode_conf(s);
            if(prefer_mode_B){ if(c < 7) c++; }
            else             { if(c > -8) c--; }
            set_mode_conf(s, c);
        }

        // Reset epoch stats
        leaderA_score = 0; leaderB_score = 0;
        num_mode_B_followers = 0; // followers will be re-granted on demand under cap
    }
}

// Decide if this set should use Mode B (B-leader safety enforced)
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return prefer_mode_B; // B leaders only if global preference flipped
    if(!prefer_mode_B) return false;
    if(get_mode_conf(set) < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

//------------------------------------------------------------------------------
// Mode A: Hawkeye-lite (exclusive rrpv_A + tiny SHCT)
//   - Tiny SHCT[512] 5-bit counters; friendly>=16
//   - Friendly insert near-MRU (2), averse deep (6), prefetch deep (6)
//   - RRIP promotion on hits
//------------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31 stored in 5 bits
static inline uint32_t shct_idx(uint64_t PC){ return (uint32_t)((PC ^ (PC>>11) ^ (PC>>17)) & (SHCT_ENTRIES-1)); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static inline uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    }
    // fallback
    uint8_t best=0; uint32_t vic=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_A[set][w] >= best){ best=rrpv_A[set][w]; vic=w; } }
    return vic;
}

static inline void mode_A_update(uint32_t set, uint32_t way, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;
    if(hit){
        if(rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        hawkeye_train(PC, true);
        return;
    }
    // Miss + fill: train negative on the PC that missed
    hawkeye_train(PC, false);
    // Insert policy
    if(is_prefetch(type)) rrpv_A[set][way] = 6;
    else                  rrpv_A[set][way] = hawkeye_is_friendly(PC) ? 2 : 6;
}

//------------------------------------------------------------------------------
// Mode B: CrossRun-DeadGate (exclusive rrpv_B + TinyLFU + dead-gate + quarantine)
//------------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];    // 3-bit
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];    // 2-bit (0..3)
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];// 1-bit

// PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)((pc ^ (pc>>13) ^ (pc>>27)) & (PC_TBL_SIZE-1)); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); }
static inline uint8_t set_hash8(uint32_t set){ return (uint8_t)(set ^ (set>>3) ^ (set>>6)); }

static uint16_t pc_last_line10[PC_TBL_SIZE];  // 10 bits used
static uint8_t  pc_stream_conf[PC_TBL_SIZE];  // 2-bit 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];         // TinyLFU 4-bit 0..15
static uint8_t  pc_dead2[PC_TBL_SIZE];        // 2-bit dead gate 0..3
static uint8_t  pc_runmask8[PC_TBL_SIZE];     // 8-bit unique-set mask
static uint8_t  pc_lfsr1[PC_TBL_SIZE];        // 1-bit pseudo-rand flip for bypass

// Tunables
static constexpr uint8_t INSERT_WARM = 2;
static constexpr uint8_t INSERT_COLD = 6;
static constexpr uint8_t STREAM_TAIL = 7;
static constexpr uint8_t STREAM_ARM  = 2;
static constexpr uint8_t HOT_THRESHOLD  = 8;
static constexpr uint8_t DEAD_THRESHOLD = 2;
static constexpr uint8_t LONGRUN_UNIQUESETS = 6; // require ≥6 distinct sets touched in run
static constexpr uint8_t PROMOTE_NS  = 2; // hits to promote non-stream
static constexpr uint8_t PROMOTE_STR = 2; // hits to rescue stream
static constexpr uint8_t DEMOTE_TOUCH = 1;

// Helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if(x<7) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }

static inline uint8_t popcount8(uint8_t x){
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (((x + (x >> 4)) & 0x0F) * 0x01);
}

static inline bool detect_and_update_run(uint64_t PC, uint64_t paddr, uint32_t set){
    uint32_t idx = pc_index(PC);
    uint16_t ln = line10(paddr);
    bool forward = false;
    if(pc_last_line10[idx] != 0xFFFFu){
        uint16_t exp1 = (uint16_t)(pc_last_line10[idx] + 1);
        uint16_t exp2 = (uint16_t)(pc_last_line10[idx] + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if(forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;

    // Update cross-set run mask
    uint8_t bit = (uint8_t)(1u << (set_hash8(set) & 7u));
    pc_runmask8[idx] |= bit;

    return (pc_stream_conf[idx] >= STREAM_ARM);
}

static inline void rrpvB_age_all(uint32_t set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
}

// Bypass predicate: long run across unique sets AND cold AND dead-gated AND stream armed AND demand
static inline bool should_bypass(uint64_t PC, bool stream_armed, bool hot, bool deadish, uint8_t uniques, uint32_t type){
    if(!is_demand(type)) return false;
    if(!stream_armed) return false;
    if(hot) return false;
    if(!deadish) return false;
    if(uniques < LONGRUN_UNIQUESETS) return false;
    // 50% via 1-bit LFSR flip per PC
    return (pc_lfsr1[pc_index(PC)] ^= 1u) & 1u;
}

// Mode B victim: includes bypass decision (returns LLC_WAYS to bypass)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set, uint64_t PC, uint64_t paddr, uint32_t type){
    // Evaluate run/bypass opportunity early
    uint32_t idx = pc_index(PC);
    bool hot_pc   = (pc_use4[idx] >= HOT_THRESHOLD);
    bool deadish  = (pc_dead2[idx] >= DEAD_THRESHOLD);
    uint8_t uniques = popcount8(pc_runmask8[idx]);
    bool stream_armed = (pc_stream_conf[idx] >= STREAM_ARM);
    if(should_bypass(PC, stream_armed, hot_pc, deadish, uniques, type))
        return LLC_WAYS; // bypass fill

    // Normal RRIP victim
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
        rrpvB_age_all(set);
    }
    uint8_t best=0; uint32_t vic=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_B[set][w] >= best){ best=rrpv_B[set][w]; vic=w; } }
    return vic;
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;

    uint32_t idx = pc_index(PC);
    if(is_demand(type)) sat_inc_u4(pc_use4[idx]);

    bool stream_armed = false;
    if(is_demand(type)) stream_armed = detect_and_update_run(PC, paddr, set);

    if(hit){
        // Quarantine demotion on touch keeps stream lines evictable
        if(stream_tag[set][way] && (rrpv_B[set][way] < maxRRPV)){
            uint8_t nv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + DEMOTE_TOUCH);
            rrpv_B[set][way] = nv;
        }

        // Multi-hit rescue
        if(hitcnt[set][way] < 3) hitcnt[set][way]++;
        bool is_stream_line = (stream_tag[set][way] != 0);
        uint8_t need = is_stream_line ? PROMOTE_STR : PROMOTE_NS;
        if(hitcnt[set][way] >= need){
            stream_tag[set][way] = 0;
            rrpv_B[set][way] = 0; // promote to MRU
            // Reward PC as not-dead when rescued
            if(pc_dead2[idx]>0) pc_dead2[idx]--;
        }else{
            if(rrpv_B[set][way] > 0) rrpv_B[set][way]--;
        }
        return;
    }

    // Miss + fill path
    // Update dead-gate negatively when we miss again from the same PC
    if(pc_dead2[idx] < 3) pc_dead2[idx]++;

    bool quarantine = is_prefetch(type) || stream_armed;
    if(quarantine){
        rrpv_B[set][way] = STREAM_TAIL;
        stream_tag[set][way] = 1;
    }else{
        bool hot_pc = (pc_use4[idx] >= HOT_THRESHOLD);
        rrpv_B[set][way] = hot_pc ? INSERT_WARM : INSERT_COLD;
        stream_tag[set][way] = 0;
    }
    hitcnt[set][way] = 0;

    // Decay runmask slowly (epochic decay in selector could also clear it; keep bounded here)
    pc_runmask8[idx] &= 0xFEu | (pc_runmask8[idx]>>1); // cheap bit spread to retain recent sets
}

//------------------------------------------------------------------------------
// Main ChampSim interface
//------------------------------------------------------------------------------
void InitReplacementState(){
    // Mode A init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++) rrpv_A[s][w] = maxRRPV;
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_B[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    for(uint32_t i=0; i<PC_TBL_SIZE; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_dead2[i] = 0;
        pc_runmask8[i] = 0;
        pc_lfsr1[i] = (uint8_t)(i & 1u);
    }

    // Selector init
    std::memset(mode_conf_nibbles, 0x88, sizeof(mode_conf_nibbles)); // encode zero (-8..+7) as 8
    std::memset(follower_bits, 0, sizeof(follower_bits));
    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_B = should_use_mode_B(set);
    if(use_B) return mode_B_victim(set, current_set, PC, paddr, type);
    else      return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Leader/follower scoring and epoch handling
    selector_on_access(set, hit);

    // Exclusive per-mode update
    if(should_use_mode_B(set)) mode_B_update(set, way, paddr, PC, type, hit);
    else                       mode_A_update(set, way, PC, type, hit);
}

void PrintStats_Heartbeat(){}
void PrintStats(){}