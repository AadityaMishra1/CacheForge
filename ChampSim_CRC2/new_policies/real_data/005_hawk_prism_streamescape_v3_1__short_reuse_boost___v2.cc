#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "../inc/champsim_crc2.h"

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

// ---------------- Tunables (v3.1 retune) ----------------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  INSERT_TAIL         = 7;   // hard tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // 2 forward steps (+1/+2)
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 9;   // TinyLFU hot threshold (↑)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// decay period for TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 2048;// selector epoch (faster)
static constexpr uint8_t  GSEL_MAX            = 31;  // global selector range
static constexpr uint8_t  GSEL_THRES          = 12;  // lower global threshold (followers)
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set bandit threshold

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Per-line metadata (pack logically: 3b rrpv, 2b hitcnt, 1b stream_lock) ----------------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set and global selectors ----------------
static int8_t  bandit_score[LLC_SETS]; // conceptual 4b [-8..7]
static uint8_t GSEL = 0;
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

// ---------------- Mode A: Hawkeye-like SHiP (safe fallback) ----------------
// Tiny SHCTs (1024 entries, 5-bit counters) for demand & prefetch
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Training state (only for 64 leader sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // 1b
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1b

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B: Prism (stream-safe DBP path) -------------------
// Compact per-PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 10b line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // last line (10b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // PC coldness (0..3)

// ---------------- Helpers ----------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void gsel_inc() { if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec() { if (GSEL > 0) GSEL--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;   // force Mode B on B-leaders
    if (LEADER_A(set)) return false;  // force Mode A on A-leaders
    // Followers: enable B only if both global and local gates allow
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 1, sizeof(shct_demand));
    std::memset(shct_prefetch, 1, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Victim selection (RRIP, invalid-first) ----------------
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return first invalid if available
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP: pick a maxRRPV; if none, age until exists
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age (saturating) all ways
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // unreachable
}

// ---------------- State update ----------------
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
    // Global periodic maintenance
    op_count++;
    if ((op_count % LFU_DECAY_PERIOD) == 0) {
        // Decay TinyLFU and gently cool PCs
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
            sat_dec_u2(pc_cold2[i]);
        }
        // Update global selector using leader hits
        if (leaderB_hits_epoch >= leaderA_hits_epoch) gsel_inc();
        else gsel_dec();
        leaderA_hits_epoch = leaderB_hits_epoch = 0;
    }

    // Per-access PC accounting (before computing insertion)
    const uint32_t pidx = pc_index(PC);
    const uint16_t cur_line = line10(paddr);
    const uint16_t last_line = pc_last_line10[pidx];
    int32_t delta = (int32_t)cur_line - (int32_t)last_line;

    // Forward-run (+1 or +2), 8-line "window" approximated by requiring small positive stride
    bool fwd_small = (delta == 1) || (delta == 2);

    // Update stream confidence on demand accesses (history first used for insertion below)
    uint8_t prev_stream_conf = pc_stream_conf[pidx];
    if (is_demand(type)) {
        if (fwd_small) {
            if (pc_stream_conf[pidx] < 3) pc_stream_conf[pidx]++;
        } else {
            if (pc_stream_conf[pidx] > 0) pc_stream_conf[pidx]--;
        }
    }
    // TinyLFU: count all accesses (demand+prefetch hits/misses)
    if (pc_use4[pidx] < 15) pc_use4[pidx]++;

    // Mode selection (leaders are forced; followers gated)
    const bool useB = modeB_enabled(set);

    if (hit) {
        // Leader hit accounting for selector
        if (LEADER_A(set)) leaderA_hits_epoch++;
        if (LEADER_B(set)) leaderB_hits_epoch++;

        // Per-set bandit update on followers
        if (!SEL_SAMPLED(set)) {
            if (useB) sat_inc_i8(bandit_score[set]);
            else      sat_dec_i8(bandit_score[set]);
        }

        // Train Hawkeye reuse on A-leaders
        if (LEADER_A(set)) {
            uint32_t ls = LEADER_SLOT(set);
            hawk_used[ls][way] = 1; // mark reuse observed
        }

        // Promotion/demotion policy
        if (stream_lock[set][way]) {
            // Stream-locked: demote-on-touch; escape only on 2nd demand hit
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    // short-reuse escape (zeusmp): unlock and promote to RRPV=1
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, 1);
                } else {
                    rrpv_demote(set, way); // keep it cold while locked
                }
                // PC got reuse: reduce coldness
                sat_dec_u2(pc_cold2[pidx]);
            } else {
                // prefetch/writeback hit while locked: demote softly
                rrpv_demote(set, way);
            }
        } else {
            // Non-stream: multi-hit promotion
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU on 2nd demand hit
                } else {
                    // gentle nudge
                    rrpv_promote(set, way);
                }
                sat_dec_u2(pc_cold2[pidx]);
            } else {
                // prefetch hit: never promote
            }
        }
        return;
    }

    // ---------------- Miss: determine insertion ----------------
    // Train Hawkeye on A-leaders using evicted line's reuse outcome
    if (LEADER_A(set)) {
        uint32_t ls = LEADER_SLOT(set);
        // Train old resident in this way
        uint16_t old_sig = hawk_sig[ls][way];
        uint32_t sidx = shct_idx(old_sig);
        if (hawk_is_pref[ls][way]) {
            if (hawk_used[ls][way]) shct_inc(shct_prefetch[sidx]);
            else                    shct_dec(shct_prefetch[sidx]);
        } else {
            if (hawk_used[ls][way]) shct_inc(shct_demand[sidx]);
            else                    shct_dec(shct_demand[sidx]);
        }
        // Install new signature
        hawk_sig[ls][way]      = pc_sig12(PC);
        hawk_is_pref[ls][way]  = (uint8_t)is_prefetch(type);
        hawk_used[ls][way]     = 0;
    }

    // Reset per-line metadata for the new fill
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    // Decide insertion policy by mode
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    if (!useB) {
        // ---------------- Mode A (Hawkeye-like SHiP) ----------------
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_writeback(type)) {
            ins_rrpv = INSERT_WARM_DEPTH; // never bypass on WB
        } else if (is_prefetch(type)) {
            ins_rrpv = INSERT_TAIL;       // quarantine prefills
            stream_lock[set][way] = 1;
        } else {
            uint8_t conf = shct_demand[idx];
            if (conf >= 16) ins_rrpv = INSERT_WARM_DEPTH;
            else            ins_rrpv = INSERT_COLD_DEPTH;
        }
    } else {
        // ---------------- Mode B (Stream-safe DBP) ----------------
        bool stream_candidate = fwd_small && (prev_stream_conf >= STREAM_CONF_THRESH);
        if (is_writeback(type)) {
            ins_rrpv = INSERT_WARM_DEPTH; // keep WB safe
        } else if (is_prefetch(type)) {
            // quarantine prefills deeply
            ins_rrpv = INSERT_TAIL;
            stream_lock[set][way] = 1;
        } else if (stream_candidate) {
            // strong stream bypass via hard-tail insert + lock
            ins_rrpv = INSERT_TAIL;
            stream_lock[set][way] = 1;
        } else {
            // PC gates for irregular/object-heavy
            bool pc_hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            bool pc_cold = (pc_cold2[pidx] >= 2);
            if (pc_hot && !pc_cold) ins_rrpv = INSERT_WARM_DEPTH;
            else                    ins_rrpv = INSERT_COLD_DEPTH;
            // Coldness learns from fills (bias to cold until reuse appears)
            sat_inc_u2(pc_cold2[pidx]);
        }
    }

    rrpv_set(set, way, ins_rrpv);

    // Update per-set bandit pessimistically on misses for followers (favor A by default)
    if (!SEL_SAMPLED(set)) {
        if (useB) sat_dec_i8(bandit_score[set]); // if we missed under B, nudge back to A
        else      sat_inc_i8(bandit_score[set]); // if we missed under A, small positive bias (stay)
    }

    // Update last-line after using history for insertion decision
    pc_last_line10[pidx] = cur_line;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}