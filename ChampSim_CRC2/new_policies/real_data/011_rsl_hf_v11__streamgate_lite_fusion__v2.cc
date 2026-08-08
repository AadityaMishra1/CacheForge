#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return t==ACCESS_LOAD || t==ACCESS_RFO; }
static inline bool is_pref(uint32_t t){ return t==ACCESS_PREFETCH; }
static inline bool is_wb(uint32_t t){ return t==ACCESS_WRITEBACK; }

// RRIP config (3-bit)
static constexpr uint8_t maxRRPV = 7;
static constexpr uint8_t INSERT_WARM_DEPTH = 2;     // near-MRU, reuse-friendly
static constexpr uint8_t INSERT_COLD_DEPTH = 6;     // near-tail, cold PCs
static constexpr uint8_t STREAM_TAIL_DEPTH = 7;     // hard tail for streams

// Stream detector (Mode B)
static constexpr uint8_t STREAM_ARM_THRESH  = 2;    // arm at 2 forward steps
static constexpr uint8_t STREAM_LOCK_THRESH = 3;    // lock at 3
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;   // demote stream on touch

// Promotion gates
static constexpr uint8_t HITS_PROMOTE_NS = 2;       // non-stream: MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_BASE = 3; // stream: MRU at 3rd demand hit (base)

// PC TinyLFU (Mode B assist)
static constexpr uint32_t PC_TBL_SIZE = 256;
static constexpr uint8_t  PC_USE_HOT_THRESH = 6;    // hot if >=6 (4-bit counter)
static constexpr uint32_t LFU_DECAY_PERIOD = 2048;

// Selector config (conservative)
static constexpr uint32_t BANDIT_EPOCH = 4096;
static constexpr int8_t   MODEB_ENABLE_THRESH = 3;
static constexpr int32_t  GATE_MARGIN = 3;

// Leader set sampling (64)
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
static int8_t  bandit_score[LLC_SETS]; // [-8..7], per-set confidence for Mode B
static bool    prefer_B = false;       // global gate
static int32_t leaderA_reward = 0;     // global leader scores
static int32_t leaderB_reward = 0;
static uint64_t access_count = 0;

// Hawkeye-lite: single SHCT + leader metadata
#define SHCT_SIZE (1u<<9)              // 512 entries
#define SHCT_MAX 31
static uint8_t shct[SHCT_SIZE];        // 5-bit conceptual

static uint16_t hawk_sig[64][LLC_WAYS];   // 12-bit sig (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];  // 1-bit: reuse seen
static uint8_t  hawk_valid[64][LLC_WAYS]; // 1-bit: entry valid

// Mode B: PC regional stride + TinyLFU
static uint16_t pc_last_region12[PC_TBL_SIZE]; // 0xFFFF uninit
static uint8_t  pc_last_off6[PC_TBL_SIZE];     // 0xFF uninit
static uint16_t pc_last_line10[PC_TBL_SIZE];   // 0xFFFF uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE];   // 0..3 (2-bit conceptual)
static uint8_t  pc_use4[PC_TBL_SIZE];          // 0..15 (4-bit conceptual)
static uint8_t  pc_last_delta3[PC_TBL_SIZE];   // 0..7 (optional bucket)
static uint8_t  pc_diversity2[PC_TBL_SIZE];    // 0..3 (optional)

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

static inline bool use_modeB(uint32_t set){
    if(LEADER_B(set)) return true;
    if(LEADER_A(set)) return false;
    if(!prefer_B) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline void end_epoch_update(){
    bool new_prefer_B = (leaderB_reward > (leaderA_reward + GATE_MARGIN));
    prefer_B = new_prefer_B;
    leaderA_reward = 0;
    leaderB_reward = 0;
}

// Mode B: update PC stride state and return current confidence (0..3)
static inline uint8_t update_stream_conf(uint64_t PC, uint64_t paddr, uint32_t type){
    uint32_t idx = (uint32_t)PC & (PC_TBL_SIZE-1u);
    if(is_demand(type)) sat_inc_u4(pc_use4[idx]);

    uint16_t reg = region12(paddr);
    uint8_t  off = region_off6(paddr);
    uint16_t ln10 = line10(paddr);

    bool inited = (pc_last_region12[idx] != 0xFFFFu) && (pc_last_off6[idx] != 0xFFu) && (pc_last_line10[idx] != 0xFFFFu);
    bool forward12 = false;
    if(inited && reg == pc_last_region12[idx]) {
        uint8_t prev = pc_last_off6[idx];
        int d = (int)off - (int)prev;
        forward12 = (d == 1) || (d == 2);
        // coarsely track delta bucket and diversity
        uint8_t bucket = (uint8_t)((d>=0? d: (-d)) & 0x7);
        if(bucket != pc_last_delta3[idx]) sat_inc_u2(pc_diversity2[idx]);
        pc_last_delta3[idx] = bucket;
    }

    if(forward12) sat_inc_u2(pc_stream_conf[idx]); else sat_dec_u2(pc_stream_conf[idx]);

    pc_last_region12[idx] = reg;
    pc_last_off6[idx]     = off;
    pc_last_line10[idx]   = ln10;

    return pc_stream_conf[idx];
}

// Initialization
void InitReplacementState() {
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    std::memset(shct, 0, sizeof(shct));
    for(uint32_t ls=0; ls<64; ls++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_valid[ls][w] = 0;
        }
    }
    for(uint32_t i=0; i<PC_TBL_SIZE; i++){
        pc_last_region12[i] = 0xFFFFu;
        pc_last_off6[i]     = 0xFFu;
        pc_last_line10[i]   = 0xFFFFu;
        pc_stream_conf[i]   = 0;
        pc_use4[i]          = 0;
        pc_last_delta3[i]   = 0;
        pc_diversity2[i]    = 0;
    }
    prefer_B = false;
    leaderA_reward = 0;
    leaderB_reward = 0;
    access_count = 0;
}

// Victim selection: SRRIP
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Return first invalid way if present
    for(uint32_t w=0; w<LLC_WAYS; w++){
        if(!current_set[w].valid) return w;
    }
    // Find RRPV==max; if none, age once and retry (bounded)
    for(int iter=0; iter<8; iter++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if(rrpv[set][w] == maxRRPV) return w;
        }
        for(uint32_t w=0; w<LLC_WAYS; w++){
            if(rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback
    return 0;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Epoch housekeeping
    access_count++;
    if((access_count % BANDIT_EPOCH) == 0) end_epoch_update();
    if((access_count % LFU_DECAY_PERIOD) == 0){
        for(uint32_t i=0;i<PC_TBL_SIZE;i++) sat_dec_u4(pc_use4[i]);
    }

    bool modeB = use_modeB(set);
    uint16_t sig = pc_sig12(PC);
    uint8_t sc = 0;
    if(modeB) sc = update_stream_conf(PC, paddr, type);

    // Leader rewards (hits are positive feedback)
    if(hit){
        if(LEADER_A(set)) leaderA_reward++;
        else if(LEADER_B(set)) leaderB_reward++;
        else {
            if(modeB) sat_inc_i8(bandit_score[set]);
        }
    } else {
        if(!SEL_SAMPLED(set) && modeB) sat_dec_i8(bandit_score[set]);
    }

    // Hit handling: multi-hit promotion with stream quarantine
    if(hit){
        // Mark reuse for Hawkeye leaders on demand hits
        if(is_demand(type) && SEL_SAMPLED(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        if(is_demand(type)){
            if(hitcnt[set][way] < 3) hitcnt[set][way]++;

            if(stream_tag[set][way]){
                // Demote on touch to keep streams near tail
                if(rrpv[set][way] <= (uint8_t)(maxRRPV - STREAM_DEMOTE_TOUCH))
                    rrpv[set][way] += STREAM_DEMOTE_TOUCH;
                else
                    rrpv[set][way] = maxRRPV;

                // Allow escape to MRU after enough demand hits (zeusmp-friendly)
                uint32_t idx = (uint32_t)PC & (PC_TBL_SIZE-1u);
                uint8_t thr = HITS_PROMOTE_STR_BASE - (pc_use4[idx] >= PC_USE_HOT_THRESH ? 1 : 0); // 3 -> 2 if hot
                if(hitcnt[set][way] >= thr){
                    rrpv_set(set, way, 0);
                }
            } else {
                // Non-stream: don't promote on 1st hit; on 2nd+ go MRU
                if(hitcnt[set][way] >= HITS_PROMOTE_NS){
                    rrpv_set(set, way, 0);
                } else {
                    // gentle warmup
                    if(rrpv[set][way] > 1) rrpv[set][way] = 1;
                }
            }
        }
        return;
    }

    // Miss/fill handling
    // Train Hawkeye SHCT on leader sets for the evicted line
    if(SEL_SAMPLED(set)){
        uint32_t slot = LEADER_SLOT(set);
        if(hawk_valid[slot][way]){
            uint16_t old_sig = hawk_sig[slot][way];
            if(hawk_used[slot][way]) shct_inc(shct[shct_idx(old_sig)]);
            else                     shct_dec(shct[shct_idx(old_sig)]);
        }
        hawk_sig[slot][way] = sig;
        hawk_used[slot][way] = 0;
        hawk_valid[slot][way] = 1;
    }

    // Decide insertion depth and stream tag
    uint8_t ins_rrpv = INSERT_WARM_DEPTH;
    uint8_t ins_stream = 0;

    // PC coldness
    bool pc_cold = (shct[shct_idx(sig)] == 0);

    if(is_wb(type)){
        // No bypass on writeback; insert fairly cold
        ins_rrpv = INSERT_COLD_DEPTH;
        ins_stream = 0;
    } else if(is_pref(type)){
        // Prefetch quarantine: always tail
        ins_rrpv = STREAM_TAIL_DEPTH;
        // Optional stream mark if detector strong
        if(modeB && sc >= STREAM_LOCK_THRESH) ins_stream = 1;
    } else {
        // Demand fill
        if(modeB){
            if(sc >= STREAM_LOCK_THRESH){
                // Strong stream -> hard tail and stream tag
                ins_rrpv = STREAM_TAIL_DEPTH;
                ins_stream = 1;
            } else if(sc >= STREAM_ARM_THRESH){
                // Armed stream -> cold insertion and tag for quarantine
                ins_rrpv = INSERT_COLD_DEPTH;
                ins_stream = 1;
            } else {
                // Not a stream: use coldness-aware insertion
                ins_rrpv = pc_cold ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            }
        } else {
            // Mode A (Hawkeye-lite): coldness-aware insertion
            ins_rrpv = pc_cold ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
        }
    }

    // Install new metadata
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = ins_stream;
}

void PrintStats_Heartbeat() {}
void PrintStats() {}