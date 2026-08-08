#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamFence-DeadLFU
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (harmonic-mean focused) ----------------
// RRIP knobs
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail insertion
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for confident streams
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;   // shallower tail for TinyLFU-hot streams
// Stream detection/promotions
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // +1/+2 forward steps needed
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;   // stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;   // hot-PC stream early escape
static constexpr uint8_t  STREAM_DEMOTE_TOUCH   = 1;   // demote stream lines on touch
// DeadLFU
static constexpr uint8_t  PC_USE_HOT_THRESH     = 8;   // TinyLFU 4b hot threshold (0..15)
static constexpr uint8_t  PC_DEAD_STRICT_TH     = 2;   // dead if >=2 (0..3)
// Selector + LFU decay
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024; // fast decay period
static constexpr uint32_t BANDIT_EPOCH          = 8192; // selector epoch (accesses)
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;    // per-set bias threshold

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector -------------------------
static int8_t bandit_score[LLC_SETS]; // followers enable Mode B only if >= MODEB_ENABLE_THRESH

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
#define SHCT_ENTRIES 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_ENTRIES];   // 5-bit conceptual
static uint8_t shct_prefetch[SHCT_ENTRIES]; // 5-bit conceptual

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b PC signature hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_ENTRIES - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A/B per-line training buffers (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint16_t hawk_pcidx[64][LLC_WAYS];   // 9b effective (for Mode B deadness training)
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamFence + DeadLFU) ------------------
// 512-entry PC tables (information-theoretic: 10b last_line, 2b stream_conf, 4b lfu, 2b deadness)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15
static uint8_t  pc_dead2[PC_TBL_SIZE];       // 0..3

// ---------------- Stats/selector state ----------------------------
static uint64_t access_count = 0;
static uint32_t leader_req_A = 0, leader_req_B = 0;
static uint32_t leader_hit_A = 0, leader_hit_B = 0;

// ---------------- Helpers -----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_pcidx, 0, sizeof(hawk_pcidx));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_dead2, 0, sizeof(pc_dead2));
    access_count = 0;
    leader_req_A = leader_req_B = 0;
    leader_hit_A = leader_hit_B = 0;
}

// ---------------- Victim selection (bounded SRRIP) ----------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Age bounded passes (ensure termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback (should not happen)
    uint32_t victim = 0;
    uint8_t  maxv = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= maxv) { maxv = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update replacement state ------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    access_count++;

    bool sampledA = LEADER_A(set);
    bool sampledB = LEADER_B(set);
    bool sampled  = sampledA || sampledB;

    // LFU periodic decay
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
        }
    }

    // Track leader req/hits for selector (ignore writebacks)
    if (!is_writeback(type) && sampled) {
        if (sampledA) {
            leader_req_A++;
            if (hit) leader_hit_A++;
        } else {
            leader_req_B++;
            if (hit) leader_hit_B++;
        }
    }

    // Selector epoch update (bounded per-set bias)
    if ((access_count % BANDIT_EPOCH) == 0) {
        int32_t goodA = (int32_t)(2*leader_hit_A) - (int32_t)leader_req_A;
        int32_t goodB = (int32_t)(2*leader_hit_B) - (int32_t)leader_req_B;
        if (goodB > goodA) {
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_inc_i8(bandit_score[s]);
        } else if (goodB < goodA) {
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_dec_i8(bandit_score[s]);
        }
        leader_req_A = leader_req_B = 0;
        leader_hit_A = leader_hit_B = 0;
    }

    // Ignore writebacks (never bypass)
    if (is_writeback(type)) return;

    // Demand access contributes to TinyLFU
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    // Hit path: multi-hit promotion + stream demotion/escape
    if (hit) {
        // Leader training: mark reuse
        if (sampled) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Only demand hits can promote
        if (is_demand(type)) {
            uint32_t pidx = pc_index(PC);
            bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

            if (stream_tag[set][way]) {
                // Quarantine behavior: demote on touch
                for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);

                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                uint8_t need = hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;

                if (hitcnt[set][way] >= need) {
                    stream_tag[set][way] = 0;      // escape quarantine
                    rrpv_set(set, way, 0);         // MRU
                }
            } else {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);         // MRU on confirmed reuse
                } else {
                    // gentle nudge toward MRU without full promotion
                    rrpv_promote(set, way);
                }
            }
        }
        return;
    }

    // Miss/Fill path: train on evicted line for leaders, then insert new line
    if (sampled) {
        uint32_t slot = LEADER_SLOT(set);
        // Train with previous resident
        uint16_t old_sig   = hawk_sig[slot][way];
        uint16_t old_pidx  = hawk_pcidx[slot][way];
        uint8_t  old_used  = hawk_used[slot][way];
        uint8_t  old_pref  = hawk_is_pref[slot][way];

        if (sampledA) {
            // Mode A training (tiny SHiP-style)
            uint32_t sidx = shct_idx(old_sig);
            if (old_sig != 0) {
                if (old_pref) { if (old_used) shct_inc(shct_prefetch[sidx]); else shct_dec(shct_prefetch[sidx]); }
                else          { if (old_used) shct_inc(shct_demand[sidx]);   else shct_dec(shct_demand[sidx]);   }
            }
        } else {
            // Mode B training: update deadness with reuse outcome
            if (old_sig != 0) {
                if (old_used) { if (old_pidx < PC_TBL_SIZE) { if (pc_dead2[old_pidx] > 0) pc_dead2[old_pidx]--; } }
                else          { if (old_pidx < PC_TBL_SIZE) { if (pc_dead2[old_pidx] < 3) pc_dead2[old_pidx]++; } }
            }
        }
        // Overwrite leader buffers for the incoming line
        hawk_sig[slot][way]    = pc_sig12(PC);
        hawk_pcidx[slot][way]  = (uint16_t)pc_index(PC);
        hawk_used[slot][way]   = 0;
        hawk_is_pref[slot][way]= is_prefetch(type) ? 1 : 0;
    }

    // Compute insertion depth per selected mode
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t stream_bit = 0;

    if (modeB_enabled(set)) {
        // Mode B: StreamFence + DeadLFU
        uint32_t pidx = pc_index(PC);
        bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        bool stream = detect_and_update_stream(PC, paddr);
        bool cold_gate = (pc_dead2[pidx] >= PC_DEAD_STRICT_TH) || !hot;

        if (is_prefetch(type)) {
            // Prefetch quarantine: always tail
            ins_rrpv = STREAM_TAIL_DEPTH;
            stream_bit = 1;
        } else if (stream) {
            ins_rrpv = hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            stream_bit = 1;
        } else {
            ins_rrpv = cold_gate ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            stream_bit = 0;
        }
    } else {
        // Mode A: tiny Hawkeye-like insertion using SHCT
        uint16_t sig = pc_sig12(PC);
        uint32_t sidx = shct_idx(sig);
        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV; // tail for prefetch
        } else {
            uint8_t conf = shct_demand[sidx];
            ins_rrpv = (conf >= (SHCT_MAX/2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
        stream_bit = 0; // no quarantine tag from Mode A
    }

    // Apply insertion
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way]   = 0;
    stream_tag[set][way]= stream_bit;
}

void PrintStats_Heartbeat() {}
void PrintStats() {}