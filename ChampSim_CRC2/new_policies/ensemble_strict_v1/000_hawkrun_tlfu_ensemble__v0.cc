/*
 * HawkRun-TLFU Ensemble
 * Mode A: Hawkeye-lite (PC-friendly classifier + RRIP)
 * Mode B: Run-Sieve (±1/±2) + TinyLFU + dead-gate + quarantine + 2-hit rescue
 *
 * Selector: conservative gating (BIAS=50, THRESHOLD=7, MAX_MODE_B_FOLLOWERS=256, EPOCH=4096)
 * - Defaults to Hawkeye-lite; Mode B enabled only if leaders show decisive wins
 * - B leaders always execute Mode B to gather score; followers capped at 12.5%
 *
 * Telemetry/tunables (sweepable):
 * - Selector: MODE_A_BIAS, MODE_B_THRESHOLD, SELECTOR_EPOCH_SIZE, MAX_MODE_B_FOLLOWERS
 * - Mode B: HOT_THRESHOLD, STREAM_ARM, LONGRUN_UNIQUESETS, PROMOTE_NS/PROMOTE_STR,
 *           DEMOTE_TOUCH, bypass off (use hard-tail quarantine), epochic decay
 *
 * Storage note: Compact per-PC tables (512 entries) and per-line small arrays; this code favors clarity.
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
// Leader-follower sampling: 64 sampled sets, low-6 == high-6 bits
//------------------------------------------------------------------------------
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

//------------------------------------------------------------------------------
// Selector (very conservative gating)
//------------------------------------------------------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // strong bias to Hawkeye-lite
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict per-set threshold

// Per-set confidence: -8..+7 stored directly in int8_t
static int8_t mode_confidence[LLC_SETS];

// Follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // bitset
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Selector decision
static inline bool should_use_mode_B(uint32_t set){
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true; // leaders execute Mode B to gather scores
    if (!prefer_mode_B) return false;
    if (mode_confidence[set] < MODE_B_THRESHOLD) return false;
    if (!follower_is_B(set)){
        if (num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set); num_mode_B_followers++;
    }
    return true;
}

// Update selector each access (hit=1, miss=0)
static inline void update_selector(uint32_t set, uint8_t hit){
    if (SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if (selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Drift local confidence toward global preference
        for (uint32_t s=0; s<LLC_SETS; s++){
            if (SAMPLED_SET(s)) continue;
            if (prefer_mode_B){ if (mode_confidence[s] < 7) mode_confidence[s]++; }
            else              { if (mode_confidence[s] > -8) mode_confidence[s]--; }
        }

        // Epochal reset of follower assignment count; bitset persists, but cap recomputed lazily
        num_mode_B_followers = 0;
        for (uint32_t s=0; s<LLC_SETS; s++){
            if (!SAMPLED_SET(s) && follower_is_B(s)) num_mode_B_followers++;
        }

        // Decay Mode B PC tables (TinyLFU/dead/run); see Mode B epoch block in UpdateReplacementState
        leaderA_score = 0; leaderB_score = 0;
    }
}

//------------------------------------------------------------------------------
// Mode A: Hawkeye-lite (RRIP + tiny SHCT)
//------------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS];      // 3-bit RRIP (stored in 8-bit)
static uint8_t line_sig_A[LLC_SETS][LLC_WAYS];  // 8-bit PC signature for negative training

// Tiny SHCT (PC-friendly): 512 entries, 5-bit counters (stored in 8-bit)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31
static inline uint32_t shct_idx(uint64_t PC){ // simple hash
    uint64_t x = PC ^ (PC>>13) ^ (PC<<7);
    return (uint32_t)x & (SHCT_ENTRIES-1);
}
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void hawkeye_train_pos(uint64_t PC){ uint8_t &c = SHCT_cnt[shct_idx(PC)]; if (c<31) c++; }
static inline void hawkeye_train_neg(uint8_t sig){ uint8_t &c = SHCT_cnt[sig & (SHCT_ENTRIES-1)]; if (c>0) c--; }

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for (uint32_t w=0; w<LLC_WAYS; w++) if (!current_set[w].valid) return w;
    // Standard RRIP victim search
    for (uint32_t pass=0; pass<8; pass++){
        for (uint32_t w=0; w<LLC_WAYS; w++){
            if (rrpv_A[set][w] == maxRRPV){
                // Negative training for evicted line's sig
                hawkeye_train_neg(line_sig_A[set][w]);
                return w;
            }
        }
        // Age
        for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    }
    // Fallback
    uint32_t vict = 0; uint8_t best = 0;
    for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_A[set][w] >= best){ best = rrpv_A[set][w]; vict = w; }
    hawkeye_train_neg(line_sig_A[set][vict]);
    return vict;
}

static inline void mode_A_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    (void)paddr;
    if (is_writeback(type)) return;

    if (hit){
        // Promotion on hit
        if (rrpv_A[set][way] > 0) rrpv_A[set][way]--;
        hawkeye_train_pos(PC);
        return;
    }

    // Miss + fill: decide insertion depth via PC-friendly
    bool friendly = hawkeye_is_friendly(PC);
    uint8_t ins = is_prefetch(type) ? 6 : (friendly ? 2 : 6);
    rrpv_A[set][way] = ins;
    line_sig_A[set][way] = (uint8_t)shct_idx(PC);
}

//------------------------------------------------------------------------------
// Mode B: Run-Sieve + TinyLFU + Dead-Gate + Quarantine
//------------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];      // 3-bit RRIP (stored in 8-bit)
static uint8_t hitcnt_B[LLC_SETS][LLC_WAYS];    // 2-bit hit counter (stored in 8-bit)
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];  // 1-bit stream quarantine tag
static uint8_t line_pcidx_B[LLC_SETS][LLC_WAYS];// 8-bit pc index at insertion (for dead-gate training)

// PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // line number modulo 1024

// Stride/run detection + TinyLFU + dead-gate
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit conf for forward run
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4-bit TinyLFU
static uint8_t  pc_dead2[PC_TBL_SIZE];       // 2-bit deadness
static uint8_t  pc_run_len4[PC_TBL_SIZE];    // 4-bit run length
static uint8_t  pc_spanmask[PC_TBL_SIZE];    // 8-bit unique-set span mask (set&7)

// Tunables
static constexpr uint8_t INSERT_WARM      = 2;
static constexpr uint8_t INSERT_COLD      = 6;
static constexpr uint8_t STREAM_TAIL      = 7;
static constexpr uint8_t STREAM_ARM       = 2; // consecutive forward hits to arm
static constexpr uint8_t HOT_THRESHOLD    = 6; // TinyLFU hot threshold
static constexpr uint8_t DEAD_THRESHOLD   = 2; // fence noisy/dead PCs
static constexpr uint8_t LONGRUN_UNIQSETS = 4; // run spans >= this many unique sets
static constexpr uint8_t PROMOTE_NS       = 2; // non-stream 2nd hit -> MRU
static constexpr uint8_t PROMOTE_STR      = 2; // stream 2nd hit -> MRU
static constexpr uint8_t DEMOTE_TOUCH     = 1; // quarantined touch demotion
static constexpr uint64_t MODEB_DECAY_EPOCH = SELECTOR_EPOCH_SIZE; // decay period

static inline void sat_inc_u2(uint8_t &x){ if (x < 3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if (x < 15) x++; }
static inline void sat_dec_u2(uint8_t &x){ if (x > 0) x--; }
static inline void rrpv_set_B(uint32_t set, uint32_t way, uint8_t val){ rrpv_B[set][way] = (val > maxRRPV) ? maxRRPV : val; }
static inline void rrpv_age_all_B(uint32_t set){ for(uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++; }

static inline bool forward_run_step(uint16_t last, uint16_t cur){
    uint16_t e1 = (uint16_t)(last + 1);
    uint16_t e2 = (uint16_t)(last + 2);
    return (cur == e1) || (cur == e2);
}

static inline uint8_t popcount8(uint8_t x){
    x = x - ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    return (uint8_t)((x + (x >> 4)) & 0x0F);
}

static inline bool detect_stream_and_update(uint64_t PC, uint64_t paddr, uint32_t set){
    uint32_t idx = pc_index(PC);
    uint16_t cur = line10(paddr);
    uint16_t last = pc_last_line10[idx];

    bool fwd = false;
    if (last != 0xFFFFu){
        fwd = forward_run_step(last, cur);
    }
    pc_last_line10[idx] = cur;

    if (fwd){
        sat_inc_u2(pc_stream_conf[idx]);
        if (pc_run_len4[idx] < 15) pc_run_len4[idx]++;
        pc_spanmask[idx] |= (uint8_t)(1u << (set & 7));
    } else {
        pc_stream_conf[idx] = 0;
        pc_run_len4[idx] = 0;
        pc_spanmask[idx] = (uint8_t)(1u << (set & 7));
    }

    // Arm stream if 2+ forward steps and spans multiple unique sets
    uint8_t uniq = popcount8(pc_spanmask[idx]);
    return (pc_stream_conf[idx] >= STREAM_ARM) && (uniq >= LONGRUN_UNIQSETS);
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for (uint32_t w=0; w<LLC_WAYS; w++) if (!current_set[w].valid) return w;
    for (uint32_t pass=0; pass<8; pass++){
        for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_B[set][w] == maxRRPV) return w;
        rrpv_age_all_B(set);
    }
    uint32_t vict=0; uint8_t best=0;
    for (uint32_t w=0; w<LLC_WAYS; w++) if (rrpv_B[set][w] >= best){ best=rrpv_B[set][w]; vict=w; }
    return vict;
}

static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    if (is_writeback(type)) return;

    uint32_t pidx = pc_index(PC);
    if (is_demand(type)) sat_inc_u4(pc_use4[pidx]);

    bool stream_armed = false;
    if (is_demand(type)) stream_armed = detect_stream_and_update(PC, paddr, set);

    if (hit){
        // Quarantine touch demote
        if (stream_tag[set][way] && (rrpv_B[set][way] < maxRRPV)){
            rrpv_set_B(set, way, (uint8_t)std::min<int>(maxRRPV, rrpv_B[set][way] + DEMOTE_TOUCH));
        }
        // Multi-hit rescue
        if (is_demand(type)){
            if (hitcnt_B[set][way] < 3) hitcnt_B[set][way]++;
            bool is_stream_line = (stream_tag[set][way] != 0);
            uint8_t need = is_stream_line ? PROMOTE_STR : PROMOTE_NS;
            if (hitcnt_B[set][way] >= need){
                stream_tag[set][way] = 0;
                rrpv_set_B(set, way, 0);
            } else {
                if (rrpv_B[set][way] > 0) rrpv_set_B(set, way, (uint8_t)(rrpv_B[set][way]-1));
            }
        }
        return;
    }

    // Miss + fill
    bool quarantine = is_prefetch(type) || stream_armed;
    bool hot_pc = (pc_use4[pidx] >= HOT_THRESHOLD);
    bool deadish = (pc_dead2[pidx] >= DEAD_THRESHOLD);

    if (quarantine){
        rrpv_set_B(set, way, STREAM_TAIL);
        stream_tag[set][way] = 1;
    } else {
        // Dead-gate: cold+dead -> colder insert
        uint8_t ins = hot_pc ? INSERT_WARM : INSERT_COLD;
        if (!hot_pc && deadish) ins = (uint8_t)std::min<int>(maxRRPV, INSERT_COLD + 1);
        rrpv_set_B(set, way, ins);
        stream_tag[set][way] = 0;
    }
    hitcnt_B[set][way] = 0;
    line_pcidx_B[set][way] = (uint8_t)pidx;
}

// Dead-gate training on eviction (call on victim selection or before overwrite)
static inline void mode_B_train_dead_on_eviction(uint32_t set, uint32_t way){
    uint8_t pcidx = line_pcidx_B[set][way];
    if (hitcnt_B[set][way] == 0){
        if (pcidx < PC_TBL_SIZE) sat_inc_u2(pc_dead2[pcidx]);
    } else {
        if (pcidx < PC_TBL_SIZE) sat_dec_u2(pc_dead2[pcidx]);
    }
}

//------------------------------------------------------------------------------
// ChampSim interface
//------------------------------------------------------------------------------
void InitReplacementState(){
    // Selector
    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    std::memset(mode_confidence, 0, sizeof(mode_confidence));
    std::memset(follower_bits, 0, sizeof(follower_bits));
    num_mode_B_followers = 0;

    // Mode A init
    for (uint32_t s=0; s<LLC_SETS; s++){
        for (uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w] = maxRRPV;
            line_sig_A[s][w] = 0;
        }
    }
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B init
    for (uint32_t s=0; s<LLC_SETS; s++){
        for (uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_B[s][w] = maxRRPV;
            hitcnt_B[s][w] = 0;
            stream_tag[s][w] = 0;
            line_pcidx_B[s][w] = 0;
        }
    }
    for (uint32_t i=0; i<PC_TBL_SIZE; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_dead2[i] = 0;
        pc_run_len4[i] = 0;
        pc_spanmask[i] = 0;
    }
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set,
                         uint64_t PC, uint64_t paddr, uint32_t type)
{
    (void)cpu; (void)PC; (void)paddr; (void)type;

    bool use_B = should_use_mode_B(set);
    if (use_B){
        // Pre-train dead-gate for the victim (about to be evicted)
        uint32_t vict = mode_B_victim(set, current_set);
        if (current_set[vict].valid) mode_B_train_dead_on_eviction(set, vict);
        return vict;
    } else {
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way,
                             uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                             uint32_t type, uint8_t hit)
{
    (void)cpu; (void)victim_addr;
    if (is_writeback(type)) return;

    // Update selector scores and epoch
    update_selector(set, hit);

    bool use_B = should_use_mode_B(set);

    if (use_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        mode_A_update(set, way, paddr, PC, type, hit);
    }

    // Epochic decay for Mode B PC tables
    if (selector_epoch % MODEB_DECAY_EPOCH == 0){
        for (uint32_t i=0; i<PC_TBL_SIZE; i++){
            pc_use4[i] >>= 1;
            pc_dead2[i] >>= 1;
            pc_run_len4[i] >>= 1;     // decay run persistence
            pc_spanmask[i] = 0;       // reset span mask
            // stream_conf decays implicitly via detect logic (reset on non-forward)
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}