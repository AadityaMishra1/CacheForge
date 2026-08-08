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

// ---------------- Tunables (retuned for lbm/zeusmp balance) -------
static constexpr uint8_t maxRRPV               = 7;  // 3-bit RRIP (0..7)
static constexpr uint8_t INSERT_WARM_DEPTH     = 3;  // near-MRU for likely-reuse
static constexpr uint8_t INSERT_COLD_DEPTH     = 6;  // near-tail for cold PCs
static constexpr uint8_t STREAM_TAIL_DEPTH     = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_ARM_THRESH     = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t HITS_PROMOTE_NS       = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH   = 1;  // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH     = 8;  // TinyLFU hot threshold (0..15)
static constexpr uint8_t PC_BYPASS_COLD_THRESH = 6;  // cold PCs (<6) enable strong stream bypass

static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch (accesses)
static constexpr int32_t  GATE_MARGIN         = 3;     // B-leaders must beat A by >=3

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Selector (leader-driven global enable) ----------
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_score = 0;      // hits - misses on leader A sets (demand only)
static int32_t leaderB_score = 0;      // hits - misses on leader B sets (demand only)
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (trained only in leaders)
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
static uint8_t  hawk_valid[64][LLC_WAYS];   // valid slot (0/1)

// ---------------- Mode B (Stride-Sieve + TinyLFU) -----------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // only 10 bits used
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU usefulness)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    // +1/+2 forward-run detector with 2-bit confidence (0..3)
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;

    // Expect last+1 or last+2 mod 1024 (10b wrap naturally with uint16 add)
    uint16_t exp1 = (uint16_t)(last + 1);
    uint16_t exp2 = (uint16_t)(last + 2);
    forward = (ln == exp1) || (ln == exp2);

    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;

    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH); // arms after 2 forward steps
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    return prefer_B;                   // followers: conservative global gate
}

// ---------------- Core interface ----------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(hawk_valid, 0, sizeof(hawk_valid));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));

    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count  = 0;
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP victim selection: find rrpv == max; age until found
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // none at max -> age all (bounded, guarantees termination)
        rrpv_age_all(set);
    }

    // Fallback (should not happen)
    return 0;
}

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

    // Epoch/selector bookkeeping
    access_count++;

    // Leader scoring (demand only)
    if (SEL_SAMPLED(set) && is_demand(type)) {
        if (LEADER_A(set)) {
            leaderA_score += (hit ? 1 : -1);
        } else if (LEADER_B(set)) {
            leaderB_score += (hit ? 1 : -1);
        }
    }

    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = (leaderB_score >= (leaderA_score + GATE_MARGIN));
        leaderA_score = 0;
        leaderB_score = 0;
        // TinyLFU decay
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;
        }
    }

    // Stream detector update (ignore for writebacks)
    bool stream_armed = false;
    if (!is_writeback(type)) {
        stream_armed = detect_and_update_stream(PC, paddr);
    }

    uint32_t pc_idx = pc_index(PC);

    // HIT path
    if (hit) {
        // Count hits (cap at 3 to save bits)
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        // Leader: mark reuse
        if (SEL_SAMPLED(set)) {
            hawk_used[LEADER_SLOT(set)][way] = 1;
        }

        // Useful PCs get TinyLFU credit on demand hits
        if (is_demand(type)) {
            sat_inc_u4(pc_use4[pc_idx]);
        }

        // Promotion policy with quarantine for streams/prefetch
        if (stream_tag[set][way]) {
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_STR) {
                // Escape quarantine and become MRU
                stream_tag[set][way] = 0;
                rrpv_set(set, way, 0);
            } else {
                // Quarantined: gentle demotion to keep tail pressure low
                for (uint8_t k = 0; k < STREAM_DEMOTE_TOUCH; k++) {
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                }
            }
        } else {
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_NS) {
                rrpv_set(set, way, 0); // MRU on 2nd demand hit
            } else {
                // soft promotion
                if (rrpv[set][way] > 1) rrpv[set][way] = 1;
            }
        }
        return;
    }

    // MISS/FILL path ------------------------------------------------

    // Train Hawkeye in leaders on eviction of prior line
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hawk_valid[slot][way]) {
            uint16_t psig = hawk_sig[slot][way];
            uint32_t sidx = shct_idx(psig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[sidx]);
                else                      shct_dec(shct_prefetch[sidx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[sidx]);
                else                      shct_dec(shct_demand[sidx]);
            }
        }
        // Record metadata for the newly inserted line
        hawk_valid[slot][way]    = 1;
        hawk_sig[slot][way]      = pc_sig12(PC);
        hawk_used[slot][way]     = 0;
        hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
    }

    // Decide mode for insertion policy
    bool useB = modeB_enabled(set);

    // Choose insertion depth and quarantine tag
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;

    if (is_writeback(type)) {
        // Never bypass writebacks; insert as warm
        ins_rrpv = INSERT_WARM_DEPTH;
    } else if (useB) {
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH; // prefetches always at tail
            new_stream_tag = 1;
        } else {
            uint8_t conf = pc_stream_conf[pc_idx];
            uint8_t use  = pc_use4[pc_idx];
            bool strong_stream = (conf >= 3) && (use < PC_BYPASS_COLD_THRESH);
            if (strong_stream || (conf >= STREAM_ARM_THRESH)) {
                // Armed streams: hard-tail insertion (bypass-equivalent)
                ins_rrpv = STREAM_TAIL_DEPTH;
                new_stream_tag = 1;
            } else {
                // Non-stream: adaptive insertion by PC usefulness
                ins_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                new_stream_tag = 0;
            }
        }
    } else {
        // Mode A (Hawkeye-like)
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            new_stream_tag = 1;
        } else {
            uint32_t sidx = shct_idx(pc_sig12(PC));
            bool pred_hot = (shct_demand[sidx] > 7); // conservative threshold
            ins_rrpv = pred_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            new_stream_tag = 0;
        }
    }

    rrpv_set(set, way, ins_rrpv);
    stream_tag[set][way] = new_stream_tag;
    hitcnt[set][way] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}