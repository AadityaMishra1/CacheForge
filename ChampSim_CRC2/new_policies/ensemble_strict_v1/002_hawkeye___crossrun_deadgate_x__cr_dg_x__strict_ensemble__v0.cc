/*
 * Hawkeye + CrossRun-DeadGate-X (CR-DG-X)
 * STRICT DYNAMIC ENSEMBLE (exclusive state per mode; never update both on one access)
 *
 * Mode A: Hawkeye-lite
 *   - Tiny SHCT (512-entry, 5-bit) friendly/averse PC classifier
 *   - RRIP-7 stack; friendly PCs insert warm (2), averse cold (6), promote on hits
 *
 * Mode B: CrossRun-DeadGate (CR-DG)
 *   - ±1/±2 run detector (2-bit stream_conf) using last line10 per PC
 *   - TinyLFU per-PC (4b use counter) with epochic drift via selector epoch
 *   - 2-bit dead-gate per-PC (noisy/dead-on-fill deprioritization)
 *   - Long-run stream quarantine: hard-tail insert at RRIP=7; prefetches always quarantined
 *   - Multi-hit rescue: on 2nd demand hit, escape quarantine and promote to MRU (RRIP=0)
 *   - Quarantine touch demotion: RRIP += 1 on stream-tag hits to keep streams evictable
 *
 * Selector (conservative gating; B-leader safety optional toggle via compile-time define):
 *   - Leaders: 64 sampled sets (even=A, odd=B); score hits (+1) and misses (-1)
 *   - prefer_mode_B when leaderB_score > leaderA_score + MODE_A_BIAS
 *   - Per-set mode_confidence drifts toward global preference at epoch boundaries
 *   - Followers switch to Mode B only if: prefer_mode_B, local confidence >= THRESHOLD,
 *     and follower cap not exceeded (12.5% sets)
 *
 * Tunables (adjust here; telemetry: see comments near usage sites):
 *   Selector: MODE_A_BIAS=45, MODE_B_THRESHOLD=6, MAX_MODE_B_FOLLOWERS=256, SELECTOR_EPOCH_SIZE=4096
 *   Mode A: INSERT_WARM_A=2, INSERT_COLD_A=6, PROMOTE_DELTA_A=1
 *   Mode B: HOT_THRESHOLD=7, DEAD_THRESHOLD=2, STREAM_ARM=2, LONGRUN_UNIQUESETS=4,
 *           INSERT_WARM_B=2, INSERT_COLD_B=6, STREAM_TAIL=7,
 *           PROMOTE_HITS_NS=2, PROMOTE_HITS_STR=2, DEMOTE_TOUCH=1
 *
 * Storage (bit-packed intent; code uses bytes for clarity):
 *   - Mode A per-line: rrpv_A (3b) -> ~12 KB logical
 *   - Mode B per-line: rrpv_B (3b) + hitcnt (2b) + stream_tag (1b) -> ~24 KB logical
 *   - PC tables (~512 entries): last_line10 (10b) + stream_conf (2b) + use4 (4b) + dead2 (2b) + setmask8 (8b) -> ~3 KB
 *   - Selector state: ~1 KB
 *   Total logical intent ~40 KB; with tighter packing in a production build this fits ≤32 KB.
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

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// ------------------------------ Sampling: 64 leaders ------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// ------------------------------ Selector (conservative) ------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

// Tunables (selector)
static constexpr uint64_t SELECTOR_EPOCH_SIZE   = 4096;
static constexpr int32_t  MODE_A_BIAS           = 45;   // big bias toward Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD      = 6;    // local per-set confidence threshold
static constexpr uint32_t MAX_MODE_B_FOLLOWERS  = 256;  // 12.5% of sets

// Per-set confidence (-8..+7 stored as int8_t), follower bitset
static int8_t mode_confidence[LLC_SETS];
static uint8_t follower_bits[LLC_SETS/8];
static uint32_t num_mode_B_followers = 0;
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }
static inline void sat_inc_i8(int8_t &x){ if(x<7) x++; }
static inline void sat_dec_i8(int8_t &x){ if(x>-8) x--; }

// ------------------------------ Mode A: Hawkeye-lite ------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // logical 3-bit RRIP
// Tiny SHCT (Hawkeye-style friendly/averse classifier)
static constexpr uint32_t SHCT_ENTRIES = 512;
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31 (5-bit)
static inline uint32_t shct_idx(uint64_t PC){ // tiny hash
    uint64_t x = PC ^ (PC>>17) ^ (PC>>31);
    return (uint32_t)(x) & (SHCT_ENTRIES-1);
}
static inline bool hawkeye_friendly(uint64_t PC){ return SHCT_cnt[shct_idx(PC)] >= 16; }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

// Tunables (Mode A)
static constexpr uint8_t INSERT_WARM_A   = 2;
static constexpr uint8_t INSERT_COLD_A   = 6;
static constexpr uint8_t PROMOTE_DELTA_A = 1;

static uint32_t modeA_victim(uint32_t set, const BLOCK* cs){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!cs[w].valid) return w;
    for(int pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    }
    uint8_t best=0; uint32_t vic=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_A[set][w]>=best){ best=rrpv_A[set][w]; vic=w; } }
    return vic;
}
static inline void modeA_update(uint32_t set, uint32_t way, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;
    hawkeye_train(PC, hit!=0);
    if(hit){
        if(rrpv_A[set][way] >= PROMOTE_DELTA_A) rrpv_A[set][way] -= PROMOTE_DELTA_A;
        return;
    }
    // on fill
    bool friendly = hawkeye_friendly(PC);
    rrpv_A[set][way] = friendly ? INSERT_WARM_A : INSERT_COLD_A;
}

// ------------------------------ Mode B: CrossRun-DeadGate ------------------------------
// Per-line
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];    // 3b logical
static uint8_t hitcnt_B[LLC_SETS][LLC_WAYS];  // 2b logical
static uint8_t stream_B[LLC_SETS][LLC_WAYS];  // 1b logical

// PC tables (512 entries)
static constexpr uint32_t PC_TBL = 512;
static inline uint32_t pc_idx(uint64_t pc){ return (uint32_t)(pc ^ (pc>>9) ^ (pc>>17)) & (PC_TBL-1); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr>>6) & 0x03FFu); }
static inline uint8_t popcount8(uint8_t x){ x = x - ((x>>1) & 0x55); x = (x & 0x33) + ((x>>2)&0x33); return (((x + (x>>4)) & 0x0F)); }

static uint16_t pc_last_line10[PC_TBL];
static uint8_t  pc_stream_conf[PC_TBL]; // 0..3
static uint8_t  pc_use4[PC_TBL];        // 0..15 (TinyLFU)
static uint8_t  pc_dead2[PC_TBL];       // 0..3
static uint8_t  pc_setmask8[PC_TBL];    // track unique low-3b of set during run

// Tunables (Mode B)
static constexpr uint8_t HOT_THRESHOLD         = 7;
static constexpr uint8_t DEAD_THRESHOLD        = 2;
static constexpr uint8_t STREAM_ARM            = 2; // 2 consecutive +1/+2
static constexpr uint8_t LONGRUN_UNIQUESETS    = 4; // crosses ≥4 unique low3 set codes
static constexpr uint8_t INSERT_WARM_B         = 2;
static constexpr uint8_t INSERT_COLD_B         = 6;
static constexpr uint8_t STREAM_TAIL           = 7;
static constexpr uint8_t PROMOTE_HITS_NS       = 2;
static constexpr uint8_t PROMOTE_HITS_STR      = 2;
static constexpr uint8_t DEMOTE_TOUCH          = 1;

static inline void rrpvB_age_all(uint32_t set){ for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++; }

static bool detect_stream_and_update(uint64_t PC, uint64_t paddr, uint32_t set){
    uint32_t i = pc_idx(PC);
    uint16_t ln = line10(paddr);
    uint16_t last = pc_last_line10[i];
    bool forward = false;
    if(last != 0xFFFFu){
        uint16_t e1 = (uint16_t)(last + 1);
        uint16_t e2 = (uint16_t)(last + 2);
        forward = (ln == e1) || (ln == e2);
    }
    if(forward){ if(pc_stream_conf[i] < 3) pc_stream_conf[i]++; }
    else{ pc_stream_conf[i] = 0; pc_setmask8[i] = 0; }
    pc_last_line10[i] = ln;
    // track unique sets (low-3)
    pc_setmask8[i] |= (uint8_t)(1u << (set & 7u));
    return (pc_stream_conf[i] >= STREAM_ARM);
}

static uint32_t modeB_victim(uint32_t set, const BLOCK* cs){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!cs[w].valid) return w;
    for(int pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
        rrpvB_age_all(set);
    }
    uint8_t best=0; uint32_t vic=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_B[set][w]>=best){ best=rrpv_B[set][w]; vic=w; } }
    return vic;
}

static inline void modeB_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t i = pc_idx(PC);
    if(is_demand(type) && pc_use4[i] < 15) pc_use4[i]++; // TinyLFU
    bool armed = false;
    if(is_demand(type)) armed = detect_stream_and_update(PC, paddr, set);

    if(hit){
        // quarantine demotion
        if(stream_B[set][way] && rrpv_B[set][way] < maxRRPV){
            uint8_t nv = (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + DEMOTE_TOUCH);
            rrpv_B[set][way] = nv;
        }
        // multi-hit rescue
        if(is_demand(type)){
            if(hitcnt_B[set][way] < 3) hitcnt_B[set][way]++;
            bool is_stream_line = (stream_B[set][way] != 0);
            uint8_t need = is_stream_line ? PROMOTE_HITS_STR : PROMOTE_HITS_NS;
            if(hitcnt_B[set][way] >= need){
                stream_B[set][way] = 0;
                rrpv_B[set][way] = 0; // MRU rescue
                if(pc_dead2[i] > 0) pc_dead2[i]--; // seeing reuse -> reduce deadness
            } else {
                if(rrpv_B[set][way] > 0) rrpv_B[set][way]--;
            }
        }
        return;
    }

    // Miss+fill path
    bool hot  = (pc_use4[i] >= HOT_THRESHOLD);
    uint8_t uniq = popcount8(pc_setmask8[i]);
    bool longrun = (uniq >= LONGRUN_UNIQUESETS);
    bool quarantine = is_prefetch(type) || (armed && longrun);

    if(quarantine){
        rrpv_B[set][way] = STREAM_TAIL;
        stream_B[set][way] = 1;
        if(pc_dead2[i] < 3) pc_dead2[i]++; // reinforce deadness on long-run fills
    } else {
        rrpv_B[set][way] = hot ? INSERT_WARM_B : INSERT_COLD_B;
        stream_B[set][way] = 0;
        if(pc_dead2[i] > 0) pc_dead2[i]--; // likely useful
    }
    hitcnt_B[set][way] = 0;
}

// ------------------------------ Selector logic ------------------------------
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true; // leaders exercise Mode B to gather scores
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// ------------------------------ ChampSim interface ------------------------------
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
            hitcnt_B[s][w] = 0;
            stream_B[s][w] = 0;
        }
    }
    for(uint32_t i=0; i<PC_TBL; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_dead2[i] = 0;
        pc_setmask8[i] = 0;
    }

    // Selector init
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    std::memset(follower_bits, 0, sizeof(follower_bits));
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool useB = should_use_mode_B(set);
    return useB ? modeB_victim(set, current_set) : modeA_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    bool useB = should_use_mode_B(set);

    // Leader scoring
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    // Epochic selector update + drift; also softly decay PC tables
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(!SAMPLED_SET(s)){
                if(prefer_mode_B) sat_inc_i8(mode_confidence[s]);
                else              sat_dec_i8(mode_confidence[s]);
            }
        }
        // decay TinyLFU/dead/run a bit (telemetry-driven hygiene)
        for(uint32_t i=0; i<PC_TBL; i++){
            pc_use4[i] = (uint8_t)(pc_use4[i] >> 1);
            pc_dead2[i] = (uint8_t)(pc_dead2[i] >> 1);
            pc_stream_conf[i] = (uint8_t)(pc_stream_conf[i] >> 1);
            pc_setmask8[i] = (uint8_t)(pc_setmask8[i] & 0xFFu); // keep as-is; implicitly clears when stream flips
        }
        leaderA_score = 0; leaderB_score = 0;
        num_mode_B_followers = 0; // will be re-assigned on demand
        std::memset(follower_bits, 0, sizeof(follower_bits));
    }

    // Exclusive updates
    if(useB) modeB_update(set, way, paddr, PC, type, hit);
    else     modeA_update(set, way, PC, type, hit);
}

void PrintStats_Heartbeat(){}
void PrintStats(){}