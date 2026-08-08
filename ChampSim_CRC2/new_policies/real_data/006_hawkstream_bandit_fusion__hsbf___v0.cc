#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD) || (t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set){
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==0u); } // Hawkeye path
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==1u); } // Stream/TinyLFU path
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ---------------- Tunables ---------------------------------------
static constexpr uint8_t  maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;  // stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_H1 = 1;  // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;  // demote quarantined streams on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 8;  // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Hawkeye

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch (1) or normal (0)

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // followers: local confidence for Mode B
static bool    prefer_B = false;       // global gate from leaders
static int32_t leaderA_score = 0;      // hits - misses on A leaders
static int32_t leaderB_score = 0;      // hits - misses on B leaders
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (leader-only training)
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x){ if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamSentry + TinyLFU) -----------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 0..15

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x){ if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x){ if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x){ if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val){
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set){
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu){
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set){
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Init --------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
    }

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }

    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
    prefer_B = false;
}

// ---------------- Victim selection --------------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
){
    // Return any invalid immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (!current_set[w].valid) return w;
    }

    // RRIP victim search with bounded aging
    while (true){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all (saturating); guarantees termination
        rrpv_age_all(set);
    }
}

// ---------------- Update state ------------------------------------
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
){
    // Ignore writebacks for policy
    if (is_writeback(type)) return;

    access_count++;

    // TinyLFU decay
    if ((access_count % LFU_DECAY_PERIOD) == 0){
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Train/use stream detector (all non-WB accesses)
    bool stream_armed = detect_and_update_stream(PC, paddr);
    uint32_t pc_idx = pc_index(PC);
    bool pc_hot = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);

    // Per-set mode decision
    bool use_modeB = modeB_enabled(set);

    // Leader-set scoring (hits - misses), bias toward Hawkeye via global gate later
    if (LEADER_A(set)){
        if (hit) leaderA_score++;
        else if (is_demand(type)) leaderA_score--;
    } else if (LEADER_B(set)){
        if (hit) leaderB_score++;
        else if (is_demand(type)) leaderB_score--;
    }

    // Epochic selector update
    if ((access_count % BANDIT_EPOCH) == 0){
        prefer_B = ((leaderB_score - leaderA_score) > GATE_MARGIN);
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Local bandit (followers only track when Mode B is used)
    if (!SEL_SAMPLED(set) && use_modeB){
        if (hit) sat_inc_i8(bandit_score[set]);
        else if (is_demand(type)) sat_dec_i8(bandit_score[set]);
    }

    // SHiP-like Hawkeye-lite training (leader sets only)
    if (SEL_SAMPLED(set)){
        uint32_t slot = LEADER_SLOT(set);
        if (!hit){
            // On fill: train the evicted resident of this way
            uint16_t prev_sig = hawk_sig[slot][way];
            uint8_t  prev_used = hawk_used[slot][way];
            uint8_t  prev_pref = hawk_is_pref[slot][way];
            if (prev_sig){
                uint32_t idx = shct_idx(prev_sig);
                if (prev_used){
                    if (prev_pref) shct_inc(shct_prefetch[idx]);
                    else           shct_inc(shct_demand[idx]);
                } else {
                    if (prev_pref) shct_dec(shct_prefetch[idx]);
                    else           shct_dec(shct_demand[idx]);
                }
            }
            // Initialize new tracker
            hawk_sig[slot][way]     = pc_sig12(PC);
            hawk_used[slot][way]    = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        } else {
            // On hit: mark reuse
            hawk_used[slot][way] = 1;
        }
    }

    // Demand usefulness accounting for TinyLFU (Mode B)
    if (is_demand(type) && hit) sat_inc_u4(pc_use4[pc_idx]);

    // Multi-hit promotion and quarantine handling
    if (hit){
        if (is_demand(type)){
            if (stream_tag[set][way]){
                // Quarantined: demote on touch (scan resistance)
                if (STREAM_DEMOTE_TOUCH) sat_inc_u3(rrpv[set][way]);

                // Promote out of quarantine on sufficient reuse
                uint8_t need = pc_hot ? HITS_PROMOTE_STR_H1 : HITS_PROMOTE_STR;
                if (hitcnt[set][way] + 1u >= need){
                    rrpv_set(set, way, 0);
                    stream_tag[set][way] = 0; // escape quarantine
                    hitcnt[set][way] = 0;
                } else {
                    if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                }
            } else {
                // Non-stream: promote only on 2nd demand hit
                if (hitcnt[set][way] + 1u >= HITS_PROMOTE_NS){
                    rrpv_set(set, way, 0);
                    hitcnt[set][way] = 0;
                } else {
                    if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                }
            }
        }
        return; // hits handled
    }

    // Miss fill: choose insertion depth per mode and stream status
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t st_tag = 0;
    if (is_prefetch(type)){
        // All prefetches to hard tail under quarantine
        ins_rrpv = STREAM_TAIL_DEPTH;
        st_tag = 1;
    } else {
        bool pred_stream = use_modeB && stream_armed;
        if (pred_stream){
            // Aggressive stream quarantine: hard tail, no early promotion
            ins_rrpv = STREAM_TAIL_DEPTH;
            st_tag = 1;
        } else {
            if (use_modeB){
                // TinyLFU PC gate
                ins_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                st_tag = 0;
            } else {
                // Mode A: Hawkeye-lite PC usefulness (leader-trained SHCT)
                uint16_t sig = pc_sig12(PC);
                uint32_t idx = shct_idx(sig);
                uint8_t conf = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];
                ins_rrpv = (conf >= (SHCT_MAX/2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                st_tag = 0;
            }
        }
    }

    rrpv_set(set, way, ins_rrpv);
    stream_tag[set][way] = st_tag;
    hitcnt[set][way] = 0;
}

// ---------------- Stats -------------------------------------------
void PrintStats() { }
void PrintStats_Heartbeat() { }