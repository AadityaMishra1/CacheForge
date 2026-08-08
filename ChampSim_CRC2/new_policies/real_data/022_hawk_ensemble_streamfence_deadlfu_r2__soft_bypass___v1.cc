#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t == ACCESS_WRITEBACK); }

// -------- Leader-set sampling (64 total) ----------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set){
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamFence-DeadLFU
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ---------------- Tunables ------------------------
// RRIP knobs
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail insertion
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for confident streams
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;   // shallower tail for TinyLFU-hot streams
// Stream detection/promotions
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // need two forward steps (+1/+2) to be confident
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;   // stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;   // LFU-hot stream: promote on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH   = 1;   // demote stream lines on touch
// DeadLFU
static constexpr uint8_t  PC_USE_HOT_THRESH     = 9;   // TinyLFU (4b) hot threshold (0..15)
static constexpr uint8_t  PC_DEAD_STRICT_TH     = 2;   // dead if >=2 (0..3)
// Selector + LFU decay
static constexpr uint32_t LFU_DECAY_PERIOD      = 4096; // slower decay for stability
static constexpr uint32_t BANDIT_EPOCH          = 16384; // longer selector epoch (accesses)
static constexpr int8_t   MODEB_ENABLE_THRESH   = 3;     // per-set bias threshold for enabling Mode B

// -------------- Per-line metadata (bit-packed logically) ---------
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1 if inserted as stream-guarded

// -------------- Per-set bandit selector --------------------------
static int8_t bandit_score[LLC_SETS]; // followers enable Mode B only if >= threshold

// -------------- Mode A (Hawkeye-like) ----------------------------
// Tiny SHCT with 12-bit PC signatures; train only on leaders
#define SHCT_ENTRIES 512
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_ENTRIES];   // 5-bit conceptual
static uint8_t shct_prefetch[SHCT_ENTRIES]; // 5-bit conceptual

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_ENTRIES - 1u); }
static inline void shct_inc(uint8_t& x){ if(x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x){ if(x > 0) x--; }

// Leader per-line buffers (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// -------------- Mode B (StreamFence + DeadLFU) -------------------
// 512-entry PC tables (10b last_line, 2b stream_conf, 4b lfu, 2b deadness)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b
static inline uint32_t fold_sig12_to_pcidx(uint16_t sig){
    uint32_t h = sig ^ (sig >> 3) ^ (sig >> 7);
    return h & (PC_TBL_SIZE - 1u);
}

static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15
static uint8_t  pc_dead2[PC_TBL_SIZE];       // 0..3

// ---------------- Stats/selector state ---------------------------
static uint64_t access_count = 0;
static uint32_t leader_req_A = 0, leader_req_B = 0;
static uint32_t leader_hit_A = 0, leader_hit_B = 0;

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x){ if(x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x){ if(x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x){ if(x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x){ if(x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x){ if(x > 0) x--; }
static inline void sat_inc_i8(int8_t& x){ if(x < 7) x++; }
static inline void sat_dec_i8(int8_t& x){ if(x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val){
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way){
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way){
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool modeB_enabled(uint32_t set){
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Deterministic 80% soft-bypass hash (no RNG)
static inline bool soft_bypass_decision(uint64_t PC, uint64_t paddr){
    uint32_t h = (uint32_t)((PC ^ (PC >> 7) ^ paddr ^ (paddr >> 6)) & 0x3FFu);
    return (h % 10u) < 8u; // true 80% of time
}

void InitReplacementState(){
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV; // cold start
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    std::memset(shct_demand,   0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4,        0, sizeof(pc_use4));
    std::memset(pc_dead2,       0, sizeof(pc_dead2));
    std::memset(hawk_sig,       0, sizeof(hawk_sig));
    std::memset(hawk_used,      0, sizeof(hawk_used));
    std::memset(hawk_is_pref,   0, sizeof(hawk_is_pref));
    access_count = 0;
    leader_req_A = leader_req_B = 0;
    leader_hit_A = leader_hit_B = 0;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
){
    // Pick any invalid way first
    for(uint32_t w=0; w<LLC_WAYS; w++){
        if(!current_set[w].valid) return w;
    }
    // SRRIP victim: search for maxRRPV; if none, age all and retry
    while (true){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if (rrpv[set][w] >= maxRRPV) return w;
        }
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Train predictors on eviction for leader sets
static inline void train_on_eviction(uint32_t set, uint32_t way){
    if (!SEL_SAMPLED(set)) return;
    uint32_t slot = LEADER_SLOT(set);
    uint16_t psig = hawk_sig[slot][way];
    uint8_t  used = hawk_used[slot][way];
    uint8_t  was_pref = hawk_is_pref[slot][way];

    // Mode A: SHCT update
    uint32_t idx = shct_idx(psig);
    if (was_pref){
        if (used) shct_inc(shct_prefetch[idx]);
        else      shct_dec(shct_prefetch[idx]);
    }else{
        if (used) shct_inc(shct_demand[idx]);
        else      shct_dec(shct_demand[idx]);
    }

    // Mode B: deadness update using folded sig -> pc idx
    uint32_t pidx = fold_sig12_to_pcidx(psig);
    if (used) { if (pc_dead2[pidx] > 0) pc_dead2[pidx]--; }
    else      { if (pc_dead2[pidx] < 3) pc_dead2[pidx]++; }

    // Clear for safety
    hawk_used[slot][way] = 0;
    hawk_is_pref[slot][way] = 0;
}

void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
){
    access_count++;

    // Periodic TinyLFU decay
    if ((access_count & (LFU_DECAY_PERIOD-1)) == 0){
        for(uint32_t i=0;i<PC_TBL_SIZE;i++) sat_dec_u4(pc_use4[i]);
    }

    // Update PC tables (stream + lfu) on every access
    uint32_t pidx = pc_index(PC);
    uint16_t curr_line = line10(paddr);
    uint16_t last_line = pc_last_line10[pidx];
    uint16_t delta = (uint16_t)((curr_line - last_line) & 0x03FFu);
    bool forward12 = (delta == 1u) || (delta == 2u);
    if (forward12) sat_inc_u2(pc_stream_conf[pidx]);
    else if (delta != 0u) sat_dec_u2(pc_stream_conf[pidx]); // non-stream step reduces confidence
    pc_last_line10[pidx] = curr_line;

    if (is_demand(type)) sat_inc_u4(pc_use4[pidx]); // only demand drives LFU hotness

    // Leader accounting (hit-rate comparison)
    if (LEADER_A(set)) { leader_req_A++; if (hit) leader_hit_A++; }
    if (LEADER_B(set)) { leader_req_B++; if (hit) leader_hit_B++; }

    // Epoch end: update per-set bias and reset tallies
    if ((access_count % BANDIT_EPOCH) == 0){
        // Compare hit-rates: B wins if hitB/reqB > hitA/reqA
        bool b_better = false;
        if (leader_req_A == 0 && leader_req_B > 0) b_better = (leader_hit_B > 0);
        else if (leader_req_B == 0 && leader_req_A > 0) b_better = false;
        else if (leader_req_A > 0 && leader_req_B > 0){
            // cross-multiply to avoid FP
            uint64_t lhs = (uint64_t)leader_hit_B * (uint64_t)leader_req_A;
            uint64_t rhs = (uint64_t)leader_hit_A * (uint64_t)leader_req_B;
            b_better = (lhs > rhs);
        }
        for(uint32_t s=0; s<LLC_SETS; s++){
            if (b_better) sat_inc_i8(bandit_score[s]);
            else          sat_dec_i8(bandit_score[s]);
        }
        leader_req_A = leader_req_B = 0;
        leader_hit_A = leader_hit_B = 0;
    }

    // HIT path: gated promotions + stream demotion
    if (hit){
        // Determine thresholds
        bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        uint8_t needed = HITS_PROMOTE_NS;
        if (stream_tag[set][way]){
            needed = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
        }
        // Only count demand hits toward promotion gate
        if (is_demand(type)){
            uint8_t prev = hitcnt[set][way];
            if (prev + 1 >= needed){
                // Escape quarantine: strong promote and clear stream tag
                rrpv_set(set, way, 0);
                stream_tag[set][way] = 0;
            }else{
                // First demand hit: no promotion
                hitcnt[set][way] = (prev < 3) ? (prev + 1) : prev;
                // Stream demotion to keep near tail
                if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
            }
        }else{
            // Prefetch hit: do not promote; optional stream demotion
            if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
        }
        return;
    }

    // MISS/FILL path
    // Train on eviction for leaders before overwriting metadata
    train_on_eviction(set, way);

    // Decide which mode to apply on this set
    bool use_modeB = modeB_enabled(set);

    // Compute policy signals
    bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
    bool stream_conf = (pc_stream_conf[pidx] >= STREAM_CONF_THRESH);
    bool dead_pc = (pc_dead2[pidx] >= PC_DEAD_STRICT_TH);

    // Mode A (Hawkeye-like): SHCT-driven insert depth
    uint16_t sig = pc_sig12(PC);
    uint32_t sidx = shct_idx(sig);
    bool hawk_friendly = (shct_demand[sidx] >= shct_prefetch[sidx]);

    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t ins_stream_tag = 0;

    if (!use_modeB){
        // Hawkeye-like: warm-insert for friendly demand PCs; cold for others; prefetch always tail
        if (is_prefetch(type)){
            ins_rrpv = STREAM_TAIL_DEPTH;
        }else if (is_writeback(type)){
            ins_rrpv = INSERT_WARM_DEPTH;
        }else{
            if (hawk_friendly && !dead_pc) ins_rrpv = INSERT_WARM_DEPTH;
            else ins_rrpv = INSERT_COLD_DEPTH;
        }
    }else{
        // Mode B: StreamFence + DeadLFU
        if (is_prefetch(type)){
            ins_rrpv = STREAM_TAIL_DEPTH; // quarantine prefetches
        }else if (is_writeback(type)){
            ins_rrpv = INSERT_WARM_DEPTH; // never bypass writebacks
        }else if (stream_conf){
            // Confident stream: soft-bypass + quarantine; shallower if hot
            bool bypass = soft_bypass_decision(PC, paddr);
            ins_rrpv = pc_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            if (bypass) ins_rrpv = STREAM_TAIL_DEPTH;
            ins_stream_tag = 1;
        }else{
            // Non-stream demand: dead PCs cold, others near-MRU
            if (dead_pc) ins_rrpv = INSERT_COLD_DEPTH;
            else         ins_rrpv = INSERT_WARM_DEPTH;
        }
    }

    // Install metadata for the new line
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = ins_stream_tag;

    // Record fill info for leaders (for future eviction training)
    if (SEL_SAMPLED(set)){
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = sig;
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }