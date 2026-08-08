/*
 * Hawkeye + CrossRun-DeadGate (CR-DG) Lite Ensemble
 * STRICT ENSEMBLE: EXCLUSIVE per-mode state; never update both modes on the same access.
 *
 * Mode A (Hawkeye-lite):
 *   - 3-bit RRIP per line + tiny SHCT (512 entries, 5-bit) for PC-friendly/averse.
 *   - Sampled signatures only for 64 leader/follower sets (no global per-line signatures).
 *   - Insert friendly near-MRU, averse near-tail; promote on hits; negative train on evict in sampled sets.
 *
 * Mode B (CR-DG Lite):
 *   - ±1/±2 run detector (2-bit stream conf) using last line10 per PC.
 *   - TinyLFU per-PC (4-bit) with epochic decay; hot PCs insert warm, cold PCs insert cold.
 *   - Dead-PC fence: per-PC 2-bit dead score increased on consecutive fills without intervening hits; decays each epoch; reduced on hits.
 *   - Prefetch and detected runs quarantined at hard tail; 2-hit rescue (escape quarantine, MRU promote).
 *   - Quarantine demotion on touches to keep streams evictable.
 *
 * Selector (Very conservative gating + B-leader safety):
 *   - MODE_A_BIAS=50, MODE_B_THRESHOLD=7, MAX_MODE_B_FOLLOWERS=256 (12.5%), SELECTOR_EPOCH_SIZE=4096.
 *   - B leaders only use Mode B if prefer_mode_B==true (safety).
 *
 * Telemetry hooks (in code comments):
 *   - Epochic decay of TinyLFU/dead scores.
 *   - Leader scores and per-set confidence.
 *
 * Storage (conceptual, bit-packed):
 *   - Mode A: rrpv_A 3b/line (~12 KB) + SHCT 1.2 KB + sampled signatures (64 sets × 16 ways × 8b ≈ 1 KB) ≈ ~14.2 KB
 *   - Mode B: rrpv_B 3b/line (~12 KB) + hitcnt 2b/line (~8 KB) + stream_tag 1b/line (~4 KB) + PC tables ~1 KB ≈ ~25 KB
 *   - Selector: <1 KB
 *   - Combined target: ~30–31 KB (bit-packing assumed; arrays below are byte-wide for clarity).
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cassert>

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

// --------------------------- Sampling (64 sets) ------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); }

// ----------------------------- Hash helpers ---------------------------------
static inline uint32_t fast_hash64(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

// ============================================================================
// Selector (very conservative, follower cap, B-leader safety)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // strong default to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict per-set threshold

static int8_t mode_confidence[LLC_SETS]; // -128..127 (we clamp to [-8..+7])

// follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t assigned_B[LLC_SETS/8]; // bitset of sets permanently allowed to use B
static inline bool is_assigned_B(uint32_t set){ return (assigned_B[set>>3] >> (set & 7)) & 1u; }
static inline void assign_B(uint32_t set){ assigned_B[set>>3] |= (uint8_t)(1u << (set & 7)); }

// saturating up/down in [-8,+7]
static inline void sat_inc_i8_7(int8_t &x){ if(x < 7) x++; }
static inline void sat_dec_i8_8(int8_t &x){ if(x > -8) x--; }

static inline bool should_use_mode_B(uint32_t set){
    // Leaders: A-leaders never B; B-leaders only B if prefer_mode_B==true (safety)
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return prefer_mode_B;

    // Followers: require global preference, per-set confidence, and follower cap
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    if(!is_assigned_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        assign_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// Mode A: Hawkeye-lite (EXCLUSIVE state)
// ============================================================================
#define maxRRPV 7

// Per-line RRIP (3-bit conceptually; byte for simplicity)
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS];

// Tiny SHCT (512 entries, 5-bit counters conceptually)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31
static inline uint32_t shct_idx(uint64_t PC){ return fast_hash64(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit) { if(c<31) c++; }
    else    { if(c>0)  c--; }
}

// Sampled signatures only for sampled sets: 64 slots × 16 ways × 8b
static uint8_t sigA_sampled[64][LLC_WAYS];

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // look for invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    // look for maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
    // age once
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    // pick any now at max
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
    return 0;
}

static void mode_A_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    (void)paddr;
    if(is_writeback(type)) return;

    // Train SHCT on sampled sets: negative on eviction (done in GetVictimInSet), positive on hits
    if(SAMPLED_SET(set) && is_demand(type) && hit){
        hawkeye_train(PC, true);
    }

    if(hit){
        if(rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        return;
    }

    // Miss + fill: insertion depth
    if(is_prefetch(type)){
        rrpv_A[set][way] = 6; // quarantine prefetches in A as well
    } else {
        bool friendly = hawkeye_is_friendly(PC);
        rrpv_A[set][way] = friendly ? 2 : 6;
    }

    // Record signature only for sampled sets for A-negative training later
    if(SAMPLED_SET(set)){
        sigA_sampled[LEADER_SLOT(set)][way] = (uint8_t)(shct_idx(PC) & 0xFFu);
    }
}

// ============================================================================
// Mode B: CrossRun-DeadGate Lite (EXCLUSIVE state)
// ============================================================================

// Per-line state (conceptually packed: rrpv 3b + hitcnt 2b + stream 1b)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt_B[LLC_SETS][LLC_WAYS];   // 0..3
static uint8_t stream_tag_B[LLC_SETS][LLC_WAYS];

// PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return fast_hash64(pc) & (PC_TBL_SIZE-1); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); }

// PC features
static uint16_t pc_last_line10[PC_TBL_SIZE]; // last line index (10-bit), 0xFFFF=invalid
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit stream conf
static uint8_t  pc_use4[PC_TBL_SIZE];       // 4-bit TinyLFU
static uint8_t  pc_dead2[PC_TBL_SIZE];      // 2-bit dead score
static uint8_t  pc_last_fill_nohit[PC_TBL_SIZE]; // sticky: 1 if last demand fill had no subsequent hit

// Tunables (exposed)
static constexpr uint8_t INSERT_WARM = 2;   // near-MRU for hot PCs
static constexpr uint8_t INSERT_COLD = 6;   // near-tail for cold PCs
static constexpr uint8_t STREAM_TAIL = 7;   // hard tail quarantine
static constexpr uint8_t STREAM_ARM  = 2;   // need 2 consecutive ±1/±2 steps
static constexpr uint8_t HOT_THRESHOLD  = 8; // TinyLFU hot when >= 8
static constexpr uint8_t DEAD_THRESHOLD = 2; // dead fence when >= 2
static constexpr uint8_t PROMOTE_NS  = 2;   // multi-hit rescue (non-stream)
static constexpr uint8_t PROMOTE_STR = 2;   // multi-hit rescue (stream)
static constexpr uint8_t DEMOTE_TOUCH = 1;  // quarantine demotion on touch

// Helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if(x<7) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void rrpv_set_B(uint32_t set, uint32_t way, uint8_t val){ rrpv_B[set][way] = (val>maxRRPV)?maxRRPV:val; }
static inline void rrpv_age_all_B(uint32_t set){ for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++; }

static inline bool detect_stream_and_update(uint64_t PC, uint64_t paddr){
    uint32_t pi = pc_index(PC);
    uint16_t ln = line10(paddr);
    uint16_t last = pc_last_line10[pi];
    bool step = false;
    if(last != 0xFFFFu){
        // bidirectional ±1/±2
        int16_t d = (int16_t)ln - (int16_t)last;
        step = (d==1)||(d==2)||(d==-1)||(d==-2);
    }
    if(step) { if(pc_stream_conf[pi] < 3) pc_stream_conf[pi]++; }
    else     { pc_stream_conf[pi] = 0; }
    pc_last_line10[pi] = ln;
    return (pc_stream_conf[pi] >= STREAM_ARM);
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(int pass=0; pass<8; pass++){
        for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
        rrpv_age_all_B(set);
    }
    // fallback
    uint32_t v=0; uint8_t best=0;
    for(uint32_t w=0; w<LLC_WAYS; w++){ if(rrpv_B[set][w] >= best){ best=rrpv_B[set][w]; v=w; } }
    return v;
}

static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;

    uint32_t pi = pc_index(PC);
    if(is_demand(type)) sat_inc_u4(pc_use4[pi]);
    bool stream_now = false;
    if(is_demand(type) || is_prefetch(type)){
        stream_now = detect_stream_and_update(PC, paddr);
    }

    if(hit){
        // quarantine demotion on touches
        if(stream_tag_B[set][way] && rrpv_B[set][way] < maxRRPV){
            rrpv_set_B(set, way, (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + DEMOTE_TOUCH));
        }
        // multi-hit rescue
        if(is_demand(type)){
            if(hitcnt_B[set][way] < 3) hitcnt_B[set][way]++;
            bool is_stream_line = (stream_tag_B[set][way] != 0);
            uint8_t need = is_stream_line ? PROMOTE_STR : PROMOTE_NS;
            if(hitcnt_B[set][way] >= need){
                stream_tag_B[set][way] = 0;
                rrpv_set_B(set, way, 0);
            } else {
                if(rrpv_B[set][way] > 0) rrpv_set_B(set, way, rrpv_B[set][way]-1);
            }
        }
        // dead fence relief on hits
        if(is_demand(type)){
            pc_last_fill_nohit[pi] = 0;
            if(pc_dead2[pi] > 0) pc_dead2[pi]--;
        }
        return;
    }

    // Miss + fill
    bool hot  = (pc_use4[pi] >= HOT_THRESHOLD);
    bool dead = (pc_dead2[pi] >= DEAD_THRESHOLD);
    bool quarantine = is_prefetch(type) || stream_now;

    // Dead fence: consecutive dead fills detection
    if(is_demand(type)){
        if(pc_last_fill_nohit[pi]) { if(pc_dead2[pi] < 3) pc_dead2[pi]++; }
        pc_last_fill_nohit[pi] = 1; // until a hit occurs
    }

    if(quarantine){
        rrpv_set_B(set, way, STREAM_TAIL);
        stream_tag_B[set][way] = 1;
    } else {
        // dead+not-hot => colder insertion
        uint8_t ins = (hot && !dead) ? INSERT_WARM : INSERT_COLD;
        rrpv_set_B(set, way, ins);
        stream_tag_B[set][way] = 0;
    }
    hitcnt_B[set][way] = 0;
}

// ============================================================================
// ChampSim interface
// ============================================================================
void InitReplacementState(){
    // Mode A init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++) rrpv_A[s][w] = maxRRPV;
    }
    for(uint32_t i=0; i<SHCT_ENTRIES; i++) SHCT_cnt[i] = 16; // neutral
    std::memset(sigA_sampled, 0, sizeof(sigA_sampled));

    // Mode B init
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_B[s][w] = maxRRPV;
            hitcnt_B[s][w] = 0;
            stream_tag_B[s][w] = 0;
        }
    }
    for(uint32_t i=0; i<PC_TBL_SIZE; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_dead2[i] = 0;
        pc_last_fill_nohit[i] = 0;
    }

    // Selector init
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    for(uint32_t s=0; s<LLC_SETS; s++) mode_confidence[s] = 0;
    num_mode_B_followers = 0;
    std::memset(assigned_B, 0, sizeof(assigned_B));
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set,
                         uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;

    bool use_B = should_use_mode_B(set);

    if(use_B){
        return mode_B_victim(set, current_set);
    } else {
        // Negative training for Mode A on sampled sets when we actually evict
        if(SAMPLED_SET(set)){
            // Identify impending victim and train its signature as averse
            // Note: we can only train here (before fill) reliably
            uint32_t v = mode_A_victim(set, current_set);
            uint32_t slot = LEADER_SLOT(set);
            uint8_t sig = sigA_sampled[slot][v];
            // Recover a pseudo-PC from sig (we only need index for SHCT); train down
            // We don't know the original PC, but SHCT index is 0..511; map back via sig
            // Use the 8-bit we stored directly as lower bits of SHCT index
            uint32_t idx = sig; // coarse; good enough for sampled negative train
            if(SHCT_cnt[idx] > 0) SHCT_cnt[idx]--;
        }
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way,
                             uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                             uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;

    if(is_writeback(type)) return;

    // Update leader scores for gating (based on actual outcome)
    if(SAMPLED_SET(set)){
        int delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    // Epochic selector update + PC decays
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B) sat_inc_i8_7(mode_confidence[s]);
            else              sat_dec_i8_8(mode_confidence[s]);
        }

        // decay TinyLFU and dead fence
        for(uint32_t i=0; i<PC_TBL_SIZE; i++){
            pc_use4[i]  >>= 1;
            if(pc_dead2[i] > 0) pc_dead2[i]--;
        }

        leaderA_score = leaderB_score = 0;
        // keep assigned_B sticky; cap already enforced when assigning
    }

    // Decide mode and update exclusively
    bool use_B = should_use_mode_B(set);
    if(use_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        mode_A_update(set, way, paddr, PC, type, hit);
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}