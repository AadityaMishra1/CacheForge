/*
 * Hawkeye + RunFence-LFU (RF-LFU) - Strict Ensemble
 * - EXCLUSIVE state: rrpv_A (Mode A) vs rrpv_B (Mode B) never mix
 * - Conservative selector: high bias, per-set threshold, follower cap
 * - Compact metadata budget (≈24 KB)
 *
 * Tunables (Knobs):
 *   Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, EPOCH=4096, MAX_MODE_B_FOLLOWERS=256
 *   RF-LFU: HOT_TH=8 (4-bit LFU), STREAM_ARM=2, INSERT_WARM=1, INSERT_COLD=2, STREAM_TAIL=3,
 *           BYPASS_RUN_LEN=8, STREAM_HITS_TO_PROMOTE=3 (gated by pc_hitstreak), DECAY_EPOCH=4096
 *
 * Useful telemetry (optional): track leaderA_score/leaderB_score, #bypasses, pc_run_len histogram.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <algorithm>
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

// 2-bit RRIP (to fit budget)
#define maxRRPV 3

// =============================== Sampling (64 sets) ===============================
static inline bool SAMPLED_SET(uint32_t set){
    return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// =============================== Selector (conservative) ===============================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;    // strong bias to Hawkeye
static constexpr int8_t MODE_B_THRESHOLD = 5; // local per-set confidence

// Per-set confidence: -8..+7 (packed as int8)
static int8_t mode_confidence[LLC_SETS];
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256; // ~12.5%
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1

static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    // Local confidence drift toward chosen mode outcome
    if(!SAMPLED_SET(set)){
        if(hit){ if(mode_confidence[set] < 7) mode_confidence[set]++; }
        else    { if(mode_confidence[set] > -8) mode_confidence[set]--; }
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // Gentle drift (guard against oscillations)
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s] < 7) mode_confidence[s]++; }
            else             { if(mode_confidence[s] > -8) mode_confidence[s]--; }
        }
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if(!follower_is_B[set]){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_is_B[set] = 1;
        num_mode_B_followers++;
    }
    return true;
}

// =============================== Mode A: Hawkeye-lite (budgeted) ===============================
// EXCLUSIVE RRIP for Mode A
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 2-bit logical

// Tiny friendly predictor (budgeted PC-based LFU, 512 entries x4b)
#define A_SHCT_ENTRIES 512
static uint8_t A_pc_cnt[A_SHCT_ENTRIES]; // 0..15
static inline uint32_t A_idx(uint64_t pc){ return (uint32_t)pc & (A_SHCT_ENTRIES-1); }
static inline bool A_is_friendly(uint64_t pc){
    return A_pc_cnt[A_idx(pc)] >= 8;
}
static inline void A_train(uint64_t pc, bool hit, bool is_pf){
    (void)is_pf;
    uint8_t &c = A_pc_cnt[A_idx(pc)];
    if(hit){ if(c<15) c++; }
    else    { if(c>0) c--; }
}

static inline uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
    return 0;
}

// =============================== Mode B: RunFence-LFU (budgeted) ===============================
// EXCLUSIVE RRIP for Mode B
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS]; // 2-bit logical

// Per-line: 1-bit "ever-hit since fill" (for dead training and 2nd-hit rescue)
static uint8_t B_hit_seen[LLC_SETS][LLC_WAYS]; // 0/1

// PC table (512 entries): last_line(10b), stride_conf(2b), freq(4b), dead(2b), run_len(4b),
// last_set(8b), hitstreak(2b), bypass_flip(1b)
#define B_PC_TABLE_SIZE 512
static uint16_t B_pc_last_line[B_PC_TABLE_SIZE];   // 0..1023, 0xFFFF means invalid
static uint8_t  B_pc_stride_conf[B_PC_TABLE_SIZE]; // 0..3
static uint8_t  B_pc_freq[B_PC_TABLE_SIZE];        // 0..15
static uint8_t  B_pc_dead[B_PC_TABLE_SIZE];        // 0..3
static uint8_t  B_pc_run_len[B_PC_TABLE_SIZE];     // 0..15
static uint8_t  B_pc_last_set[B_PC_TABLE_SIZE];    // 0..255
static uint8_t  B_pc_hitstreak[B_PC_TABLE_SIZE];   // 0..3
static uint8_t  B_pc_flip[B_PC_TABLE_SIZE];        // 0/1

// RF-LFU Tunables
static constexpr uint8_t B_INSERT_WARM  = 1;  // near-MRU (2-bit RRIP)
static constexpr uint8_t B_INSERT_COLD  = 2;  // near-tail
static constexpr uint8_t B_STREAM_TAIL  = 3;  // hard tail quarantine
static constexpr uint8_t B_STREAM_ARM   = 2;  // need 2 stride confirmations
static constexpr uint8_t B_HOT_TH       = 8;  // TinyLFU hot threshold
static constexpr uint8_t B_BYPASS_RUN_LEN = 8;// cross-set run length to start bypass
static constexpr uint64_t DECAY_EPOCH   = 4096;

static inline uint32_t B_idx(uint64_t pc){ return (uint32_t)pc & (B_PC_TABLE_SIZE-1); }
static inline uint16_t B_line_id(uint64_t paddr){ return (uint16_t)((paddr>>6)&0x3FF); }
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

static uint64_t global_epoch = 0;

static inline bool B_detect_stream(uint64_t PC, uint64_t paddr){
    uint32_t i = B_idx(PC);
    uint16_t line = B_line_id(paddr);
    uint16_t last = B_pc_last_line[i];
    bool stride = false;
    if(last != 0xFFFF){
        uint16_t lp1 = (uint16_t)(last + 1);
        uint16_t lp2 = (uint16_t)(last + 2);
        uint16_t lm1 = (uint16_t)(last - 1);
        uint16_t lm2 = (uint16_t)(last - 2);
        stride = (line==lp1)||(line==lp2)||(line==lm1)||(line==lm2);
    }
    if(stride) sat_inc_u2(B_pc_stride_conf[i]);
    else       B_pc_stride_conf[i] = 0;
    B_pc_last_line[i] = line;
    return (B_pc_stride_conf[i] >= B_STREAM_ARM);
}

static inline void B_decay_epoch(){
    if(global_epoch % DECAY_EPOCH == 0){
        for(uint32_t i=0;i<B_PC_TABLE_SIZE;i++){
            B_pc_freq[i] >>= 1;
            B_pc_dead[i] >>= 1;
            B_pc_run_len[i] >>= 1;
            if(B_pc_hitstreak[i]>0) B_pc_hitstreak[i]--; // mild decay
            B_pc_flip[i] ^= 1; // flip bypass coin each epoch for diversity
        }
    }
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t i = B_idx(PC);

    // TinyLFU update on demand
    if(is_demand(type)) sat_inc_u4(B_pc_freq[i]);

    bool is_stream = B_detect_stream(PC, paddr);

    // Cross-set run tracking (only when stream-like)
    uint8_t curr_set8 = (uint8_t)(set & 0xFF);
    if(is_stream){
        if(B_pc_last_set[i] != curr_set8) sat_inc_u4(B_pc_run_len[i]);
        else B_pc_run_len[i] = 0;
    }else{
        B_pc_run_len[i] = 0;
    }
    B_pc_last_set[i] = curr_set8;

    // Update hitstreak and per-line "ever-hit"
    if(hit){
        if(B_pc_hitstreak[i] < 3) B_pc_hitstreak[i]++;
        if(B_hit_seen[set][way]==0) B_hit_seen[set][way]=1;
    }else{
        // On fill/miss, slightly decay hitstreak (handled by epoch decay too)
        if(B_pc_hitstreak[i]>0) B_pc_hitstreak[i]--;
        B_hit_seen[set][way] = 0;
    }

    // Train deadness on fills that never saw a hit before eviction (approx via miss with prior no-hit)
    // Note: ChampSim doesn't pass eviction PC here; approximate: if last access was miss (no prior hit_seen), bump dead
    if(!hit && is_demand(type)){
        sat_inc_u2(B_pc_dead[i]);
    }

    // Insertion policy (EXCLUSIVE to rrpv_B)
    bool pc_hot = (B_pc_freq[i] >= B_HOT_TH);
    bool quarantine = is_stream || is_prefetch(type);
    if(quarantine){
        rrpv_B[set][way] = B_STREAM_TAIL;
    }else if(pc_hot){
        rrpv_B[set][way] = B_INSERT_WARM;
    }else{
        rrpv_B[set][way] = B_INSERT_COLD;
    }

    // Multi-hit rescue:
    // - Non-stream: promote on 2nd demand hit (detected via per-line B_hit_seen==1 and this is another hit)
    // - Stream PCs: require stronger evidence -> pc_hitstreak>=2 to promote
    if(hit && is_demand(type)){
        if(!is_stream){
            if(B_hit_seen[set][way]) rrpv_B[set][way] = 0; // MRU
        }else{
            if(B_pc_hitstreak[i] >= 2) rrpv_B[set][way] = 0;
        }
    }

    // Periodic decay
    B_decay_epoch();
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t paddr, uint64_t PC, uint32_t type){
    (void)set; (void)paddr;
    if(is_prefetch(type)) return true; // strict quarantine/bypass for prefetch noise
    uint32_t i = B_idx(PC);
    bool is_stream = (B_pc_stride_conf[i] >= B_STREAM_ARM);
    bool pc_cold = (B_pc_freq[i] < B_HOT_TH);
    bool long_run = (B_pc_run_len[i] >= B_BYPASS_RUN_LEN);
    bool pc_very_dead = (B_pc_dead[i] >= 3);
    // Probabilistic bypass on long runs from cold or dead PCs
    if(is_stream && long_run && pc_cold && B_pc_flip[i]) return true;
    if(is_stream && pc_very_dead && B_pc_flip[i]) return true;
    return false;
}

static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;
    return 0;
}

// =============================== ChampSim Hooks ===============================
void InitReplacementState(){
    for(uint32_t s=0; s<LLC_SETS; s++){
        mode_confidence[s]=0; follower_is_B[s]=0;
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w]=maxRRPV; rrpv_B[s][w]=maxRRPV;
            B_hit_seen[s][w]=0;
        }
    }
    for(uint32_t i=0;i<A_SHCT_ENTRIES;i++) A_pc_cnt[i]=0;

    for(uint32_t i=0;i<B_PC_TABLE_SIZE;i++){
        B_pc_last_line[i]=0xFFFF; B_pc_stride_conf[i]=0;
        B_pc_freq[i]=0; B_pc_dead[i]=0; B_pc_run_len[i]=0;
        B_pc_last_set[i]=0xFF; B_pc_hitstreak[i]=0; B_pc_flip[i]=i&1;
    }
    leaderA_score=leaderB_score=0; prefer_mode_B=false;
    selector_epoch=0; num_mode_B_followers=0; global_epoch=0;
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_B = should_use_mode_B(set);

    // Optional bypass (Mode B only). Return LLC_WAYS (16) to signal bypass in ChampSim.
    if(use_B && mode_B_should_bypass(set, paddr, PC, type)){
        return LLC_WAYS; // bypass fill
    }

    if(use_B) return mode_B_victim(set, current_set);
    else      return mode_A_victim(set, current_set);
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Update selector on outcome
    update_selector(set, hit);
    global_epoch++;

    bool use_B = should_use_mode_B(set);

    // EXCLUSIVE updates: only one mode per access
    if(use_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    }else{
        // Mode A: simple friendly insertion via tiny PC counter
        bool friendly = A_is_friendly(PC);
        if(is_prefetch(type)){
            rrpv_A[set][way] = maxRRPV; // prefetches: low priority
        }else if(friendly){
            rrpv_A[set][way] = 0;       // MRU for friendly PCs
        }else{
            rrpv_A[set][way] = maxRRPV; // tail for averse PCs
        }
        // Train tiny friendly predictor on demand
        if(is_demand(type)) A_train(PC, hit!=0, is_prefetch(type));
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}