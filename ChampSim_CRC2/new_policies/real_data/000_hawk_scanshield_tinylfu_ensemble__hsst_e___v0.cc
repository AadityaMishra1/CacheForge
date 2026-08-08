#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 types (stable for CRC2)
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
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (scan resistance + irregular reuse) ----
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU (friendly)
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail (averse)
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for TinyLFU-hot streams

static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream/prefetch: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_H  = 1;  // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined streams on touch

static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Mode A

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t   bandit_score[LLC_SETS]; // streaming pressure / confidence
static bool     prefer_B = false;        // global gate from leaders
static int32_t  leaderA_good = 0;        // hits - misses on leader A sets
static int32_t  leaderB_good = 0;        // hits - misses on leader B sets
static uint64_t access_count = 0;

// Keep the chosen victim info for training on fill
static uint8_t pending_victim_way[LLC_SETS];
static uint8_t pending_victim_valid[LLC_SETS];

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b xor-folded hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (ScanShield + TinyLFU) -------------------
// 512-entry PC tables: last line (10b+valid), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Replacement API --------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w]       = maxRRPV;
            hitcnt[s][w]     = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
        pending_victim_way[s]   = 0;
        pending_victim_valid[s] = 0;
    }

    std::memset(shct_demand,   0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));

    for (uint32_t i = 0; i < 64; i++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[i][w]     = 0;
            hawk_used[i][w]    = 0;
            hawk_is_pref[i][w] = 0;
        }
    }

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i]        = 0;
    }

    prefer_B     = false;
    leaderA_good = 0;
    leaderB_good = 0;
    access_count = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            pending_victim_valid[set] = 0; // no eviction training for invalid
            return w;
        }
    }

    // RRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                pending_victim_way[set]   = (uint8_t)w;
                pending_victim_valid[set] = 1;
                return w;
            }
        }
        // Age once and retry (saturates at maxRRPV)
        rrpv_age_all(set);
    }
}

// Helper: Mode A insertion depth based on SHCT
static inline uint8_t hawk_insertion_depth(uint64_t PC, uint32_t type) {
    uint16_t sig  = pc_sig12(PC);
    uint32_t idx  = shct_idx(sig);
    uint8_t  val  = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];
    // Friendly if counter is warm; else averse
    return (val >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Ignore writebacks for policy state (never bypass on WB)
    if (is_writeback(type)) return;

    access_count++;

    // TinyLFU decay (global, cheap: 512 entries)
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Track streaming pressure per-set to bias selector a bit (safe even under Mode A)
    if (is_demand(type)) {
        bool is_stream = detect_and_update_stream(PC, paddr);
        if (is_stream) sat_inc_i8(bandit_score[set]);
        else           sat_dec_i8(bandit_score[set]);
        // Demand access contributes to TinyLFU
        sat_inc_u4(pc_use4[pc_index(PC)]);
    }

    // Leader bookkeeping (reward = +1 hit, -1 miss)
    if (SEL_SAMPLED(set)) {
        int32_t reward = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_good += reward;
        else               leaderB_good += reward;

        // End of epoch: decide global preference
        if ((access_count % BANDIT_EPOCH) == 0) {
            prefer_B = ((leaderB_good - leaderA_good) > GATE_MARGIN);
            leaderA_good = 0;
            leaderB_good = 0;
        }
    }

    // On hit: multi-hit gated promotion; stream quarantine demotion
    if (hit) {
        uint8_t hc = hitcnt[set][way];
        if (stream_tag[set][way]) {
            // quarantine demotion on touch
            if (STREAM_DEMOTE_TOUCH && (rrpv[set][way] < maxRRPV)) sat_inc_u3(rrpv[set][way]);

            // promote only when sufficient reuse appears
            bool hot_pc = (pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH);
            uint8_t need = hot_pc ? HITS_PROMOTE_STR_H : HITS_PROMOTE_STR;
            if (hc + 1 >= need) {
                rrpv_set(set, way, 0);     // escape quarantine
                stream_tag[set][way] = 0;  // clear stream tag after proven reuse
                hitcnt[set][way] = 0;
            } else {
                hitcnt[set][way] = (uint8_t)std::min<uint8_t>(3, hc + 1);
            }
        } else {
            // non-stream: MRU on 2nd demand hit; no promote on prefetch-only touches
            if (hc + 1 >= HITS_PROMOTE_NS) {
                rrpv_set(set, way, 0);
                hitcnt[set][way] = 0;
            } else {
                hitcnt[set][way] = (uint8_t)std::min<uint8_t>(3, hc + 1);
            }
        }

        // Mark reuse for Hawkeye leaders
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1; // reuse seen
        }

        return;
    }

    // Miss path (new fill): train Mode A (leaders) on the evicted line
    if (SEL_SAMPLED(set) && pending_victim_valid[set]) {
        uint32_t slot = LEADER_SLOT(set);
        uint32_t vway = pending_victim_way[set];
        // Decide which SHCT to train
        if (hawk_is_pref[slot][vway]) {
            if (hawk_used[slot][vway]) shct_inc(shct_prefetch[shct_idx(hawk_sig[slot][vway])]);
            else                       shct_dec(shct_prefetch[shct_idx(hawk_sig[slot][vway])]);
        } else {
            if (hawk_used[slot][vway]) shct_inc(shct_demand[shct_idx(hawk_sig[slot][vway])]);
            else                       shct_dec(shct_demand[shct_idx(hawk_sig[slot][vway])]);
        }
        // Clear old record (not strictly necessary)
        hawk_used[slot][vway] = 0;
        pending_victim_valid[set] = 0;
    }

    // Choose ensemble mode for this set
    bool use_modeB = modeB_enabled(set);

    // Decide insertion policy
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t st_tag   = 0;

    if (use_modeB) {
        // ScanShield + TinyLFU
        bool armed_stream = detect_and_update_stream(PC, paddr);
        bool hot_pc = (pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH);

        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH; // quarantine prefetches
            st_tag   = 1;
        } else if (armed_stream) {
            ins_rrpv = hot_pc ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            st_tag   = 1; // mark as stream quarantined
        } else {
            // non-stream: PC gate
            ins_rrpv = hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            st_tag   = 0;
        }
    } else {
        // Mode A: Hawkeye-like PC usefulness (leaders trained)
        ins_rrpv = is_prefetch(type) ? STREAM_TAIL_DEPTH : hawk_insertion_depth(PC, type);
        st_tag   = is_prefetch(type) ? 1 : 0; // quarantine prefetches
    }

    // Install the new line
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way]     = 0;
    stream_tag[set][way] = st_tag;

    // Record per-line training context for leaders
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = (uint8_t)is_prefetch(type);
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}