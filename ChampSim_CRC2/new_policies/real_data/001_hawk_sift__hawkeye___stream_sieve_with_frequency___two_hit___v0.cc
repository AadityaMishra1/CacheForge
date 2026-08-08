#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

// ChampSim CRC2 hooks
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ------------ Leader-set sampling: 64 total (low 6 bits == next 6) ------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // Stream-Sieve
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (retuned for lbm + irregulars) ---------------------
static constexpr uint8_t  maxRRPV               = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;  // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;  // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;  // hard tail
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;  // for TinyLFU-hot streams
static constexpr uint8_t  STREAM_ARM_THRESH     = 2;  // +1/+2 strides to arm
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;  // stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;  // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH   = 1;  // demote quarantined on touch
static constexpr uint8_t  PC_USE_HOT_THRESH     = 8;  // TinyLFU hot gate (0..15)
static constexpr uint8_t  SHCT_WARM_THRESH      = 8;  // Hawkeye-like warm threshold (0..31)

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH         = 4096; // accesses
static constexpr int8_t   MODEB_ENABLE_THRESH  = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN          = 3;    // global bias toward Mode A

// TinyLFU decay period
static constexpr uint32_t LFU_DECAY_PERIOD     = 2048;

// ---------------- Per-line metadata (conceptually bit-packed) -----------------
// rrpv:3b, hitcnt:2b, stream_tag:1b  => 6 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1: quarantined stream/prefetch

// ---------------- Per-set selector (bandit) -----------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], only updated when set is in Mode B
static bool prefer_B = false;          // global decision gate from leaders
static int32_t leaderA_good = 0;       // hits - misses on A leaders
static int32_t leaderB_good = 0;       // hits - misses on B leaders
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (trained only in leaders)
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only buffers (64 sampled sets) to train SHCT on eviction
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit effective
static uint8_t  hawk_used[64][LLC_WAYS];    // 1 if line saw reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1 if filled by prefetch

// ---------------- Mode B (Stream-Sieve + TinyLFU) -----------------------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF = uninitialized
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 0..15

// ---------------- Small helpers ------------------------------------------------
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

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    // Detect +1/+2 forward runs; arm after two consecutive forward steps
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
    if (LEADER_B(set)) return true;   // force B on B-leaders
    if (LEADER_A(set)) return false;  // force A on A-leaders
    // Followers: enable B only if globally preferred and local bandit confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Core API -----------------------------------------------------
void InitReplacementState() {
    std::memset(rrpv,         maxRRPV, sizeof(rrpv)); // initialize to max
    std::memset(hitcnt,       0,       sizeof(hitcnt));
    std::memset(stream_tag,   0,       sizeof(stream_tag));
    std::memset(bandit_score, 0,       sizeof(bandit_score));

    std::memset(shct_demand,  0,       sizeof(shct_demand));
    std::memset(shct_prefetch,0,       sizeof(shct_prefetch));
    std::memset(hawk_sig,     0,       sizeof(hawk_sig));
    std::memset(hawk_used,    0,       sizeof(hawk_used));
    std::memset(hawk_is_pref, 0,       sizeof(hawk_is_pref));

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i]        = 0;
    }

    prefer_B = false;
    leaderA_good = leaderB_good = 0;
    access_count = 0;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // First, pick any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP victim selection with bounded aging
    for (;;) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines by +1 (saturate)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
        // Loop will terminate because all counters saturate to maxRRPV
    }
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
) {
    // Periodic selector gate and TinyLFU decay
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = ((leaderB_good - leaderA_good) > GATE_MARGIN);
        leaderA_good = leaderB_good = 0;
    }
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }

    // Update TinyLFU and stream detector on demand/prefetch
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
        (void)detect_and_update_stream(PC, paddr);
    } else if (is_prefetch(type)) {
        (void)detect_and_update_stream(PC, paddr);
    }

    // Leader accounting for selector (reward = +1 hit, -1 miss)
    if (SEL_SAMPLED(set)) {
        int reward = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_good += reward;
        else if (LEADER_B(set)) leaderB_good += reward;
    } else {
        // Followers: only update bandit when actually running Mode B
        if (modeB_enabled(set)) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else     sat_dec_i8(bandit_score[set]);
        }
    }

    uint32_t p_idx = pc_index(PC);
    bool pc_hot = (pc_use4[p_idx] >= PC_USE_HOT_THRESH);
    bool stream_armed = (pc_stream_conf[p_idx] >= STREAM_ARM_THRESH);
    bool use_modeB = modeB_enabled(set);

    // Handle WRITEBACKs: never bypass; insert moderately warm on fills
    if (is_writeback(type)) {
        if (!hit) {
            rrpv_set(set, way, INSERT_WARM_DEPTH);
            hitcnt[set][way] = 0;
            stream_tag[set][way] = 0;
        }
        return;
    }

    // Leader-set SHCT training on fills (train evicted line), then record new fill
    if (SEL_SAMPLED(set) && !hit) {
        uint32_t slot = LEADER_SLOT(set);
        // Train evicted line
        uint16_t old_sig = hawk_sig[slot][way];
        uint8_t  old_used = hawk_used[slot][way];
        uint8_t  old_is_pref = hawk_is_pref[slot][way];
        if (old_sig) {
            uint32_t si = shct_idx(old_sig);
            if (old_is_pref) {
                if (old_used) shct_inc(shct_prefetch[si]);
                else          shct_dec(shct_prefetch[si]);
            } else {
                if (old_used) shct_inc(shct_demand[si]);
                else          shct_dec(shct_demand[si]);
            }
        }
        // Record new fill signature
        uint16_t newsig = pc_sig12(PC);
        hawk_sig[slot][way]     = newsig;
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }

    // Mark reuse on hits in sampled sets
    if (SEL_SAMPLED(set) && hit) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_used[slot][way] = 1;
    }

    // Multi-hit promotion + quarantine handling
    if (hit) {
        // Optionally demote quarantined streams upon touch (scan resistance)
        if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) {
            uint8_t tmp = rrpv[set][way];
            rrpv_set(set, way, (uint8_t)std::min<uint8_t>(maxRRPV, (uint8_t)(tmp + STREAM_DEMOTE_TOUCH)));
        }

        // Prefetch hits do not count toward promotion
        if (is_demand(type)) {
            uint8_t hc = hitcnt[set][way];
            if (hc < 3) hc++;
            hitcnt[set][way] = hc;

            uint8_t need = HITS_PROMOTE_NS;
            if (stream_tag[set][way]) {
                need = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
            }
            if (hc >= need) {
                // escape quarantine and promote to MRU
                stream_tag[set][way] = 0;
                rrpv_set(set, way, 0);
            } else {
                // gentle nudge on first demand hit for non-streams
                if (!stream_tag[set][way] && rrpv[set][way] > 1)
                    rrpv_set(set, way, 1);
            }
        }
        return;
    }

    // Miss fill: compute insertion depth by mode and predictors
    uint8_t ins_depth = INSERT_WARM_DEPTH;
    bool quarantine = false;

    // SHCT prediction for this PC (read-only in followers)
    uint16_t sig = pc_sig12(PC);
    uint32_t si = shct_idx(sig);
    uint8_t shct_val = is_prefetch(type) ? shct_prefetch[si] : shct_demand[si];
    bool hawk_friendly = (shct_val >= SHCT_WARM_THRESH);

    if (use_modeB) {
        if (is_prefetch(type)) {
            ins_depth = STREAM_TAIL_DEPTH; // quarantine all prefetches
            quarantine = true;
        } else if (stream_armed) {
            // LBM-like streaming: hard tail; shallow if PC-hot
            ins_depth = pc_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            quarantine = true;
        } else {
            // Non-stream: gate by SHCT and TinyLFU
            if (hawk_friendly && pc_hot) ins_depth = INSERT_WARM_DEPTH;
            else                         ins_depth = INSERT_COLD_DEPTH;
        }
    } else { // Mode A (Hawkeye-like)
        if (is_prefetch(type)) {
            ins_depth = STREAM_TAIL_DEPTH; // prefetch quarantine near tail
            quarantine = true;
        } else {
            ins_depth = hawk_friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    rrpv_set(set, way, ins_depth);
    hitcnt[set][way]   = 0;
    stream_tag[set][way] = quarantine ? 1 : 0;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}