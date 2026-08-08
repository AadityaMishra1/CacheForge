#include <vector>
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

// ---------------- Tunables (v3) ----------------------------------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;   // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_ARM_THRESH   = 2;   // +1/+2 forward steps to arm
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_H1 = 1;   // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;   // demote quarantined streams on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  SHCT_HOT_THRESH     = 8;   // SHCT hot threshold (0..31)
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
static int8_t  bandit_score[LLC_SETS]; // local confidence for Mode B (saturating 3b)
static uint8_t active_modeB[LLC_SETS]; // per-access latch (conceptual 1b)
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
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1) by demand
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamSentry v3 + TinyLFU) --------------
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
static inline void sat_inc_u4(uint8_t &x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_i3(int8_t &x){ if (x < 3) x++; }
static inline void sat_dec_i3(int8_t &x){ if (x > -3) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val){
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set){
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
}

// Stream detector state read (update happens on demand misses)
static inline bool stream_armed(uint64_t PC){
    return pc_stream_conf[pc_index(PC)] >= STREAM_ARM_THRESH;
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        active_modeB[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
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
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// ---------------- Victim selection (SRRIP) ------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Latch mode choice for this access
    bool modeB = false;
    if (LEADER_A(set)) modeB = false;
    else if (LEADER_B(set)) modeB = true;
    else modeB = (prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH));
    active_modeB[set] = modeB ? 1 : 0;

    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP victim selection with guaranteed termination
    for (uint32_t pass = 0; pass <= maxRRPV; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age once if no maxRRPV present
        rrpv_age_all(set);
    }
    return 0; // fallback
}

// ---------------- Update replacement state ------------------------
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    // Periodic TinyLFU decay and bandit epoch handling
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = (leaderB_score > (leaderA_score + GATE_MARGIN));
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // Demand access updates TinyLFU usage
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    // Leader set scoring (demand-only)
    if (SEL_SAMPLED(set) && is_demand(type)) {
        if (LEADER_A(set)) {
            if (hit) leaderA_score++; else leaderA_score--;
        } else if (LEADER_B(set)) {
            if (hit) leaderB_score++; else leaderB_score--;
        }
    }
    // Follower per-set bandit scoring (only when B was chosen)
    if (!SEL_SAMPLED(set) && is_demand(type)) {
        if (active_modeB[set]) {
            if (hit) sat_inc_i3(bandit_score[set]);
            else     sat_dec_i3(bandit_score[set]);
        }
    }

    // On hits: multi-hit promotion and stream quarantine handling
    if (hit) {
        // leader training: mark reuse only for demand hits
        if (SEL_SAMPLED(set) && is_demand(type)) {
            uint32_t ls = LEADER_SLOT(set);
            hawk_used[ls][way] = 1;
        }

        // Promotion policy
        if (is_demand(type)) {
            // increment hit count (max 3)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            bool promote = false;
            uint32_t pidx = pc_index(PC);
            bool lfu_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

            if (stream_tag[set][way]) {
                // quarantined: demote on touch and only escape after gate
                if ((lfu_hot && hitcnt[set][way] >= HITS_PROMOTE_STR_H1) ||
                    (!lfu_hot && hitcnt[set][way] >= HITS_PROMOTE_STR)) {
                    promote = true;
                    stream_tag[set][way] = 0; // escape quarantine
                } else {
                    // demote a bit to hasten eviction if no quick reuse
                    if (rrpv[set][way] < maxRRPV)
                        rrpv[set][way] = std::min<uint8_t>(maxRRPV, rrpv[set][way] + STREAM_DEMOTE_TOUCH);
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) promote = true;
            }

            if (promote) rrpv_set(set, way, 0);
        } else {
            // prefetch hit: no promotion, gently demote if quarantined
            if (stream_tag[set][way] && rrpv[set][way] < maxRRPV)
                rrpv[set][way] = std::min<uint8_t>(maxRRPV, uint8_t(rrpv[set][way] + STREAM_DEMOTE_TOUCH));
        }
        return;
    }

    // From here: miss path (allocation). Never bypass writebacks.
    // Train SHCT on eviction for leader sets (using previous line's signature/state)
    if (SEL_SAMPLED(set)) {
        uint32_t ls = LEADER_SLOT(set);
        uint16_t old_sig = hawk_sig[ls][way];
        uint32_t idx = shct_idx(old_sig);
        if (hawk_is_pref[ls][way]) {
            if (hawk_used[ls][way]) shct_inc(shct_prefetch[idx]);
            else                    shct_dec(shct_prefetch[idx]);
        } else {
            if (hawk_used[ls][way]) shct_inc(shct_demand[idx]);
            else                    shct_dec(shct_demand[idx]);
        }
        // install new per-line leader metadata
        hawk_sig[ls][way] = pc_sig12(PC);
        hawk_is_pref[ls][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[ls][way] = 0;
    }

    // Update stream detector only on demand misses (steady streams advance here)
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        uint16_t ln   = line10(paddr);
        uint16_t last = pc_last_line10[pidx];
        bool forward = false;
        if (last != 0xFFFFu) {
            uint16_t exp1 = (uint16_t)(last + 1);
            uint16_t exp2 = (uint16_t)(last + 2);
            forward = (ln == exp1) || (ln == exp2);
        }
        if (forward) sat_inc_u2(pc_stream_conf[pidx]);
        else         pc_stream_conf[pidx] = 0;
        pc_last_line10[pidx] = ln;
    }

    // Decide insertion depth and flags
    uint8_t new_rrpv = INSERT_WARM_DEPTH;
    uint8_t new_stream_tag = 0;
    uint8_t new_hitcnt = 0;

    // Helper: SHCT usefulness for this PC
    auto pc_shct_useful = [&](uint64_t pc, bool pref)->bool {
        uint16_t sig = pc_sig12(pc);
        uint32_t idx = shct_idx(sig);
        uint8_t val = pref ? shct_prefetch[idx] : shct_demand[idx];
        return (val >= SHCT_HOT_THRESH);
    };

    bool modeB = (active_modeB[set] != 0);

    if (is_writeback(type)) {
        // Writebacks never bypass and get moderate priority
        new_rrpv = INSERT_WARM_DEPTH;
        new_stream_tag = 0;
        new_hitcnt = 0;
    } else if (is_prefetch(type)) {
        // Prefetch quarantine at hard tail
        new_rrpv = STREAM_TAIL_DEPTH;
        new_stream_tag = 1;
        new_hitcnt = 0;
    } else {
        // Demand miss
        uint32_t pidx = pc_index(PC);
        bool lfu_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        bool sarmed  = stream_armed(PC);

        if (modeB) {
            if (sarmed) {
                // Stream: hard tail + quarantine (aggressive near-bypass)
                new_rrpv = STREAM_TAIL_DEPTH;
                new_stream_tag = 1;
                new_hitcnt = 0;
            } else {
                // Non-stream in Mode B: TinyLFU + SHCT gate
                if (lfu_hot) new_rrpv = INSERT_WARM_DEPTH;
                else if (pc_shct_useful(PC, false)) new_rrpv = INSERT_WARM_DEPTH;
                else new_rrpv = INSERT_COLD_DEPTH;
            }
        } else {
            // Mode A (Hawkeye-like via SHCT)
            if (pc_shct_useful(PC, false)) new_rrpv = INSERT_WARM_DEPTH;
            else new_rrpv = INSERT_COLD_DEPTH;
        }
    }

    // Install metadata
    rrpv_set(set, way, new_rrpv);
    stream_tag[set][way] = new_stream_tag;
    hitcnt[set][way] = new_hitcnt;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}