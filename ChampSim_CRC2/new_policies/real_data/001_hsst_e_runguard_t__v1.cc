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
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (scan + irregular reuse) --------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU (friendly)
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail (averse)
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for TinyLFU-hot streams

static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t STRONG_STREAM_CONF  = 3;  // strong stream → always hard tail
static constexpr uint8_t HITS_PROMOTE_ALL    = 2;  // MRU on 2nd demand hit (non-stream)
static constexpr uint8_t HITS_PROMOTE_STR_H  = 1;  // TinyLFU-hot stream escape on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined streams on touch

static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector (global bandit via leaders)
static constexpr uint32_t BANDIT_EPOCH        = 4096;
static constexpr int32_t  GATE_MARGIN         = 3;    // bias toward Mode A

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Global selector --------------------------------
static bool     prefer_B = false;        // global gate from leaders
static int32_t  leaderA_good = 0;        // hits - misses on leader A sets
static int32_t  leaderB_good = 0;        // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)
static uint8_t  hawk_valid[64][LLC_WAYS];   // valid (0/1)

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

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12-bit xor-folded signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Stream detector: returns current confidence after update
static inline uint8_t stream_update(uint64_t PC, uint64_t paddr) {
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
    return pc_stream_conf[idx];
}

// ---------------- Init -------------------------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_tag, 0, sizeof(stream_tag));

    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }

    for (uint32_t s = 0; s < 64; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[s][w] = 0;
            hawk_used[s][w] = 0;
            hawk_is_pref[s][w] = 0;
            hawk_valid[s][w] = 0;
        }
    }

    prefer_B = false;
    leaderA_good = leaderB_good = 0;
    access_count = 0;
}

// ---------------- RRIP victim selection --------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Prefer any invalid way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP: find a line with RRPV == max; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
}

// ---------------- Update replacement state -----------------------
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
    (void)cpu; (void)victim_addr;

    // TinyLFU update + periodic decay
    uint32_t pidx = pc_index(PC);
    sat_inc_u4(pc_use4[pidx]);
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Stream detector update (on all accesses)
    uint8_t sconf = stream_update(PC, paddr);
    bool stream_armed = (sconf >= STREAM_ARM_THRESH);
    bool stream_strong = (sconf >= STRONG_STREAM_CONF);

    // Leader accounting for mode selector
    if (SEL_SAMPLED(set)) {
        if (LEADER_A(set)) {
            leaderA_good += (hit ? 1 : -1);
        } else { // LEADER_B
            leaderB_good += (hit ? 1 : -1);
        }
        // Epoch gate
        if ((access_count % BANDIT_EPOCH) == 0) {
            int32_t diff = leaderB_good - leaderA_good;
            prefer_B = (diff > GATE_MARGIN);
            leaderA_good = leaderB_good = 0;
        }
    } else {
        // Epoch gate also ticks even if not in leader sets
        if ((access_count % BANDIT_EPOCH) == 0) {
            int32_t diff = leaderB_good - leaderA_good;
            prefer_B = (diff > GATE_MARGIN);
            leaderA_good = leaderB_good = 0;
        }
    }

    // Handle hits: multi-hit gated promotion, demote quarantined streams
    if (hit) {
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            // Stream-aware early escape if PC is hot
            uint8_t promote_thr = HITS_PROMOTE_ALL;
            if (stream_tag[set][way]) {
                if (pc_use4[pidx] >= PC_USE_HOT_THRESH) promote_thr = HITS_PROMOTE_STR_H; // 1
            }

            if (hitcnt[set][way] >= promote_thr) {
                rrpv_set(set, way, 0);      // promote to MRU
                hitcnt[set][way] = 0;       // reset
                stream_tag[set][way] = 0;   // escaped quarantine
            } else {
                // Not enough hits yet: keep near-MRU but not MRU
                if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) {
                    sat_inc_u3(rrpv[set][way]); // demote quarantined streams on touch
                } else {
                    if (rrpv[set][way] > 1) rrpv_set(set, way, 1);
                }
            }
        } else {
            // Prefetch or writeback hit: gentle nudge only
            if (rrpv[set][way] > 1) rrpv_set(set, way, 1);
        }

        // Mark reuse for Hawkeye leaders
        if (LEADER_A(set) || LEADER_B(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // Miss path (fill): train Hawkeye on eviction for leader sets, then insert
    bool modeB = LEADER_B(set) || (!LEADER_A(set) && !LEADER_B(set) && prefer_B);

    // Train SHCT on victim eviction before overwriting leader metadata
    if (LEADER_A(set) || LEADER_B(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hawk_valid[slot][way]) {
            uint32_t idx = shct_idx(hawk_sig[slot][way]);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                else                      shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                else                      shct_dec(shct_demand[idx]);
            }
        }
        // Prepare metadata for the new fill
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = (uint8_t)is_prefetch(type);
        hawk_used[slot][way]    = 0;
        hawk_valid[slot][way]   = 1;
    }

    // Decide insertion depth and quarantine tagging
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t st_tag = 0;

    if (is_writeback(type)) {
        // Never bypass writebacks; treat as likely-reused
        ins_depth = INSERT_WARM_DEPTH;
        st_tag = 0;
    } else if (is_prefetch(type)) {
        // Prefetch quarantine at hard tail
        ins_depth = STREAM_TAIL_DEPTH;
        st_tag = 1;
    } else {
        // Demand fill
        if (modeB && stream_armed) {
            // Stream quarantine; allow shallow tail for TinyLFU-hot PCs when only moderately armed
            st_tag = 1;
            if (stream_strong) {
                ins_depth = STREAM_TAIL_DEPTH; // strong → hard tail
            } else {
                ins_depth = (pc_use4[pidx] >= PC_USE_HOT_THRESH) ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            }
        } else {
            // Hawkeye + LFU coldness gate
            uint16_t sig = pc_sig12(PC);
            uint32_t si  = shct_idx(sig);
            bool warm = (shct_demand[si] >= 15) || (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            ins_depth = warm ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            st_tag = 0;
        }
    }

    // Install
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = st_tag;
}

// Print end-of-simulation statistics
void PrintStats() {
    // intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // intentionally blank
}