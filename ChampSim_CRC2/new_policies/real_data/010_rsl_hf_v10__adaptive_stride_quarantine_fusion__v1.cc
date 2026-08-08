#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return t==ACCESS_LOAD || t==ACCESS_RFO; }
static inline bool is_pref(uint32_t t){ return t==ACCESS_PREFETCH; }
static inline bool is_wb(uint32_t t){ return t==ACCESS_WRITEBACK; }

// RRIP config
static constexpr uint8_t maxRRPV = 7;                    // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH = 2;          // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH = 6;          // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH = 7;          // hard tail

// Stream detector (Mode B)
static constexpr uint8_t STREAM_ARM_THRESH = 2;          // arm at 2 forward steps
static constexpr uint8_t STREAM_LOCK_THRESH = 3;         // lock at 3
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;        // demote quarantined line on touch

// Multi-hit promotion gates
static constexpr uint8_t HITS_PROMOTE_NS = 2;            // non-stream: promote to MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR = 2;           // stream: promote to MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT = 1;       // if PC LFU-hot: 1st demand hit

// PC TinyLFU
static constexpr uint32_t PC_TBL_SIZE = 512;
static constexpr uint8_t  PC_USE_HOT_THRESH = 8;         // 4-bit LFU hotness
static constexpr uint32_t LFU_DECAY_PERIOD = 2048;

// Selector config (bias to Hawkeye)
static constexpr uint32_t BANDIT_EPOCH = 4096;
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;
static constexpr int32_t  GATE_MARGIN = 3;

// Leader set sampling (64 total)
static constexpr uint32_t LLC_SET_BITS = 11;
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// Per-line metadata (conceptually bit-packed): rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// Selector state
static int8_t  bandit_score[LLC_SETS]; // [-8..7] per-set confidence for Mode B
static bool    prefer_B = false;       // global gate
static int32_t leaderA_reward = 0;     // global A leader score
static int32_t leaderB_reward = 0;     // global B leader score
static uint64_t access_count = 0;

// Mode A (Hawkeye-lite): tiny SHCT and leader buffers
#define SHCT_SIZE (1u<<10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // 0/1: reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 0/1: filled by prefetch

// Mode B (PC regional stride + TinyLFU + diversity)
static uint16_t pc_last_region12[PC_TBL_SIZE]; // 0xFFFF uninit
static uint8_t  pc_last_off6[PC_TBL_SIZE];     // 0xFF uninit
static uint16_t pc_last_line10[PC_TBL_SIZE];   // 0xFFFF uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE];   // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];          // 0..15
static uint8_t  pc_last_delta3[PC_TBL_SIZE];   // 0..7 bucketed abs delta
static uint8_t  pc_diversity2[PC_TBL_SIZE];    // 0..3

// Helpers
static inline void sat_inc_i8(int8_t &x){ if(x<7) x++; }
static inline void sat_dec_i8(int8_t &x){ if(x>-8) x--; }
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if(x<7) x++; }
static inline void sat_dec_u3(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc>>3) ^ (pc>>11) ^ (pc>>19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)(sig & (SHCT_SIZE-1u)); }
static inline void shct_inc(uint8_t &x){ if(x<SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x){ if(x>0) x--; }

static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr>>6) & 0x03FFu); }
static inline uint16_t region12(uint64_t paddr){
    uint64_t x = (paddr>>12);
    x ^= (x>>7) ^ (x>>13);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint8_t region_off6(uint64_t paddr){ return (uint8_t)((paddr>>6) & 0x3Fu); }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t v){
    rrpv[set][way] = (v>maxRRPV)? maxRRPV : v;
}

// Decide which mode to use for this set
static inline bool use_modeB(uint32_t set){
    if(LEADER_B(set)) return true;
    if(LEADER_A(set)) return false;
    // followers: prefer Hawkeye unless both gates allow B
    if(!prefer_B) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Update global selector at epoch boundary
static inline void end_epoch_update(){
    bool new_prefer_B = (leaderB_reward > (leaderA_reward + GATE_MARGIN));
    prefer_B = new_prefer_B;
    leaderA_reward = 0;
    leaderB_reward = 0;
}

// Mode B: update PC stride state and return whether stream is armed/locked
static inline bool update_and_is_stream(uint64_t PC, uint64_t paddr, uint32_t type){
    uint32_t idx = (uint32_t)(PC) & (PC_TBL_SIZE-1u);
    if(is_demand(type)) sat_inc_u4(pc_use4[idx]); // count only demand for LFU

    uint16_t reg = region12(paddr);
    uint8_t  off = region_off6(paddr);
    uint16_t ln10 = line10(paddr);

    bool inited = (pc_last_region12[idx] != 0xFFFFu) && (pc_last_off6[idx] != 0xFFu) && (pc_last_line10[idx] != 0xFFFFu);

    // Forward stride +1/+2 detection on both region-off and low line number
    bool fwd = false;
    if(inited && pc_last_region12[idx] == reg){
        int off_delta = (int)off - (int)pc_last_off6[idx];
        int ln_delta  = (int)ln10 - (int)pc_last_line10[idx];
        if( (off_delta==1 || off_delta==2) && (ln_delta==1 || ln_delta==2) ){
            fwd = true;
            if(pc_stream_conf[idx] < 3) pc_stream_conf[idx]++; // build confidence
        }else{
            if(pc_stream_conf[idx] > 0) pc_stream_conf[idx]--; // decay on mismatch
        }
        // Diversity tracking (bucket abs delta into [0..7])
        uint8_t absd = (uint8_t)(std::min(7, std::abs(off_delta)));
        if(absd != pc_last_delta3[idx]) { sat_inc_u2(pc_diversity2[idx]); }
        else { sat_dec_u2(pc_diversity2[idx]); }
        pc_last_delta3[idx] = absd;
    }else{
        // first sighting: no stream confidence
        pc_stream_conf[idx] = 0;
        pc_last_delta3[idx] = 0;
    }

    // Update last positions
    pc_last_region12[idx] = reg;
    pc_last_off6[idx]     = off;
    pc_last_line10[idx]   = ln10;

    // Armed/locked stream if confidence >= 2
    (void)fwd;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

void InitReplacementState() {
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for(uint32_t sl=0; sl<64; sl++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            hawk_sig[sl][w] = 0;
            hawk_used[sl][w] = 0;
            hawk_is_pref[sl][w] = 0;
        }
    }
    for(uint32_t i=0;i<PC_TBL_SIZE;i++){
        pc_last_region12[i] = 0xFFFFu;
        pc_last_off6[i]     = 0xFFu;
        pc_last_line10[i]   = 0xFFFFu;
        pc_stream_conf[i]   = 0;
        pc_use4[i]          = 0;
        pc_last_delta3[i]   = 0;
        pc_diversity2[i]    = 0;
    }
    access_count = 0;
    prefer_B = false;
    leaderA_reward = leaderB_reward = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid, return immediately
    for(uint32_t w=0; w<LLC_WAYS; w++){
        if(!current_set[w].valid) return w;
    }
    // RRIP victim selection
    while(true){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if(rrpv[set][w] == maxRRPV) return w;
        }
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if(rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Epoch maintenance
    access_count++;
    if((access_count % LFU_DECAY_PERIOD) == 0){
        for(uint32_t i=0;i<PC_TBL_SIZE;i++){ pc_use4[i] >>= 1; }
    }
    if((access_count % BANDIT_EPOCH) == 0){
        end_epoch_update();
    }

    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    bool modeB = use_modeB(set);

    // Update leader rewards (hits positive, misses negative)
    if(leaderA){ if(hit) leaderA_reward++; else leaderA_reward--; }
    if(leaderB){ if(hit) leaderB_reward++; else leaderB_reward--; }

    // Train leader-mode A SHCT on eviction (only for sampled sets)
    if(!hit && (leaderA || leaderB)){
        uint32_t slot = LEADER_SLOT(set);
        // Evicted line's training
        uint16_t sig = hawk_sig[slot][way];
        uint8_t  used = hawk_used[slot][way];
        uint8_t  was_pref = hawk_is_pref[slot][way];
        uint32_t idx = shct_idx(sig);
        if(was_pref) { if(used) shct_inc(shct_prefetch[idx]); else shct_dec(shct_prefetch[idx]); }
        else         { if(used) shct_inc(shct_demand[idx]);   else shct_dec(shct_demand[idx]); }
        // Reset for new fill
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_pref(type) ? 1 : 0;
    }

    // On hit: apply promotion rules and selector/bandit updates
    if(hit){
        // Update leader used-bit if this is a leader set
        if(leaderA || leaderB){
            uint32_t slot = LEADER_SLOT(set);
            if(is_demand(type)) hawk_used[slot][way] = 1;
        }

        // Local bandit score: only reinforce when we actually used Mode B on followers
        if(!leaderA && !leaderB){
            if(modeB){
                if(hit) sat_inc_i8(bandit_score[set]);
            }else{
                // bias toward Hawkeye: no positive reinforcement to B here
                // slight decay keeps score conservative
                sat_dec_i8(bandit_score[set]);
            }
        }

        // Apply multi-hit and quarantine behavior
        if(is_demand(type)){
            if(hitcnt[set][way] < 3) hitcnt[set][way]++; // 2-bit saturating assumed (0..3)
        }
        // Demote quarantined stream lines on touch
        if(stream_tag[set][way]){
            if(rrpv[set][way] < maxRRPV) rrpv[set][way] = std::min<uint8_t>(maxRRPV, rrpv[set][way] + STREAM_DEMOTE_TOUCH);
            // Escape quarantine and promote if sufficient reuse
            uint32_t idx = (uint32_t)(PC) & (PC_TBL_SIZE-1u);
            bool lfu_hot = (pc_use4[idx] >= PC_USE_HOT_THRESH);
            uint8_t need = lfu_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
            if(is_demand(type) && hitcnt[set][way] >= need){
                rrpv_set(set, way, 0);      // MRU
                stream_tag[set][way] = 0;   // escaped quarantine
            }
        }else{
            // Non-stream: promote only on 2nd+ demand hit
            if(is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_NS){
                rrpv_set(set, way, 0);
            }
        }
        return;
    }

    // Miss path: decide insertion depth and stream tagging
    uint8_t insert_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;

    if(modeB){
        bool is_stream = update_and_is_stream(PC, paddr, type);
        uint32_t idx = (uint32_t)(PC) & (PC_TBL_SIZE-1u);
        bool lfu_hot = (pc_use4[idx] >= PC_USE_HOT_THRESH);
        uint8_t div = pc_diversity2[idx];

        if(is_pref(type)){
            insert_rrpv = STREAM_TAIL_DEPTH; // prefetch quarantine
            new_stream_tag = 1;
        }else if(is_stream){
            insert_rrpv = STREAM_TAIL_DEPTH; // hard tail for streams
            new_stream_tag = 1;
        }else{
            // Not streaming: rely on LFU and diversity guard
            if(lfu_hot && div <= 1){
                insert_rrpv = INSERT_WARM_DEPTH; // warm insertion for likely reuse
            }else{
                insert_rrpv = INSERT_COLD_DEPTH; // conservative near-tail
            }
        }
    }else{
        // Mode A (Hawkeye-lite): SHCT-guided insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if(is_pref(type)){
            insert_rrpv = STREAM_TAIL_DEPTH; // always tail
        }else{
            // Friendly if counter is high; threshold ~ half
            bool friendly = (shct_demand[idx] >= (SHCT_MAX/2));
            insert_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }

        // For leaders, record metadata for the filled line
        if(LEADER_A(set) || LEADER_B(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = is_pref(type) ? 1 : 0;
        }
        // Also update PC TinyLFU/diversity so Mode B has signal ready
        (void)update_and_is_stream(PC, paddr, type);
    }

    // Writebacks never bypass: insert cold in both modes
    if(is_wb(type)){
        insert_rrpv = INSERT_COLD_DEPTH;
        new_stream_tag = 0;
    }

    // Followers: update bandit score on misses based on chosen mode
    if(!LEADER_A(set) && !LEADER_B(set)){
        if(modeB) sat_dec_i8(bandit_score[set]); // penalize B on miss
        else       sat_inc_i8(bandit_score[set]); // slight reward A on miss (bias toward A)
    }

    // Install new line metadata
    rrpv_set(set, way, insert_rrpv);
    stream_tag[set][way] = new_stream_tag;
    hitcnt[set][way] = 0;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}