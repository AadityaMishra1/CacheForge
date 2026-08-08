#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
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

// ---------------- Tunables (v2 retune) ---------------------------
static constexpr uint8_t  maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;  // stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_H1 = 1;  // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;  // demote quarantined streams on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15), lowered
static constexpr uint32_t LFU_DECAY_PERIOD    = 1024; // faster decay

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 4;    // global bias toward Hawkeye

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1=quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // local confidence for Mode B (saturating, conceptual 3b)
static bool    prefer_B = false;       // global gate from leaders
static int32_t leaderA_score = 0;      // demand hits - misses on A leaders
static int32_t leaderB_score = 0;      // demand hits - misses on B leaders
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

// ---------------- Mode B (StreamSentry v2 + TinyLFU) --------------
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

// Stream detector: update only on misses to track line-to-line forward runs
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
    if (LEADER_B(set)) return true;     // force Mode B on leader-B
    if (LEADER_A(set)) return false;    // force Mode A on leader-A
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline void tinyLFU_decay(){
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_tag, 0, sizeof(stream_tag));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t s = 0; s < 64; s++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            hawk_sig[s][w] = 0;
            hawk_used[s][w] = 0;
            hawk_is_pref[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++){
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    prefer_B = false;
    leaderA_score = leaderB_score = 0;
    access_count = 0;
}

// ---------------- Victim Selection (RRIP) -------------------------
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // 1) Choose any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) RRIP victim search: look for maxRRPV; if none, age and retry
    for (int iter = 0; iter < 8; iter++) { // bounded to avoid infinite loop
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // Fallback (should not happen): choose way 0
    return 0;
}

// ---------------- State Update ------------------------------------
static uint64_t lfu_tick = 0;

void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    // Epoch and LFU decay housekeeping
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Global gate (bias toward Hawkeye)
        prefer_B = (leaderB_score > (leaderA_score + GATE_MARGIN));
        leaderA_score = leaderB_score = 0;
    }
    if (is_demand(type)) {
        lfu_tick++;
        if ((lfu_tick % LFU_DECAY_PERIOD) == 0) tinyLFU_decay();
    }

    bool use_modeB = modeB_enabled(set);

    // Leader scoring (demand only)
    if (is_demand(type)) {
        if (LEADER_A(set)) { if (hit) leaderA_score++; else leaderA_score--; }
        if (LEADER_B(set)) { if (hit) leaderB_score++; else leaderB_score--; }
    }

    // Local bandit heuristic: if we see an armed stream on misses, raise confidence; penalize noisy demand misses
    bool armed_stream = false;
    if (!hit && !is_writeback(type)) {
        armed_stream = detect_and_update_stream(PC, paddr);
        if (armed_stream) sat_inc_i8(bandit_score[set]);
        else if (is_demand(type)) sat_dec_i8(bandit_score[set]);
    }

    // TinyLFU update on demand hits
    if (hit && is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    // HIT behavior
    if (hit) {
        // Multi-hit promotion gate; never on writeback/prefetch touches
        if (is_demand(type)) {
            // Stream quarantine handling
            if (stream_tag[set][way]) {
                // Demote on touch to keep scan resistance
                if (STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                // Early escape for TinyLFU-hot
                uint32_t pidx = pc_index(PC);
                bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
                if (hot) {
                    // Promote immediately on first demand hit
                    rrpv_set(set, way, 0);
                    stream_tag[set][way] = 0;
                    hitcnt[set][way] = 0;
                } else {
                    // Require 2nd demand hit to escape
                    if (hitcnt[set][way] + 1 >= HITS_PROMOTE_STR) {
                        rrpv_set(set, way, 0);
                        stream_tag[set][way] = 0;
                        hitcnt[set][way] = 0;
                    } else {
                        if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                    }
                }
            } else {
                // Non-stream: promote only on 2nd demand hit
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                    hitcnt[set][way] = 0;
                } else {
                    if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                    // small gentle aging toward MRU is omitted to stay conservative
                }
            }
        }
        // Leader A: mark reuse for SHiP training
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // MISS / FILL behavior (no bypass on writebacks)
    // Train Hawkeye (Mode A leaders) on victim being evicted from leader-A sets
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t psig = hawk_sig[slot][way];
        uint8_t  used = hawk_used[slot][way];
        uint8_t  was_pref = hawk_is_pref[slot][way];
        if (psig) {
            uint32_t idx = shct_idx(psig);
            if (was_pref) {
                if (used) shct_inc(shct_prefetch[idx]); else shct_dec(shct_prefetch[idx]);
            } else {
                if (used) shct_inc(shct_demand[idx]); else shct_dec(shct_demand[idx]);
            }
        }
        // overwrite new fill info below
    }

    // Choose insertion depth
    uint8_t insert_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;

    if (is_writeback(type)) {
        // Writebacks: insert warm-ish to avoid premature eviction of dirty lines
        insert_rrpv = INSERT_WARM_DEPTH;
        new_stream_tag = 0;
    } else if (use_modeB) {
        // Mode B: StreamSentry v2 + TinyLFU
        bool stream_armed_now = armed_stream; // computed on miss above
        if (is_prefetch(type)) {
            insert_rrpv = STREAM_TAIL_DEPTH;
            new_stream_tag = 1; // quarantine
        } else if (stream_armed_now) {
            insert_rrpv = STREAM_TAIL_DEPTH;
            new_stream_tag = 1; // quarantine stream
        } else {
            // Non-stream demand: TinyLFU gate
            uint32_t pidx = pc_index(PC);
            bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            insert_rrpv = hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_tag = 0;
        }
    } else {
        // Mode A: Hawkeye-like (SHiP)
        if (is_prefetch(type)) {
            insert_rrpv = STREAM_TAIL_DEPTH;
            new_stream_tag = 1; // quarantine prefetch
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t conf = shct_demand[idx];
            bool useful = (conf >= 8); // conservative threshold
            insert_rrpv = useful ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_tag = 0;
        }
    }

    // Install new block
    rrpv_set(set, way, insert_rrpv);
    stream_tag[set][way] = new_stream_tag;
    hitcnt[set][way] = 0;

    // Record per-line info for leader-A training
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}