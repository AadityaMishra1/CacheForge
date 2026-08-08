#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// Leader-set sampling (64 leaders total: 32 A, 32 B)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Tunables
static constexpr uint8_t  maxRRPV             = 7;   // 3b RRIP window
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot PCs
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// periodic decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set enable

// Per-line metadata (conceptual packing: rrpv:3, hitcnt:2, stream_lock:1)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// Per-set and global selectors
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B
static uint8_t GSEL = 0;               // global gate (0..31)

// Mode A (Hawkeye-like SHiP) components
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
// Per-line training only in leader-A sets (64 x 16)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Mode B PC predictors (stride runs + TinyLFU + coldness)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b id
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b deadness

// Helpers
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

// Forward-run detector (+1/+2 stride)
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;             // force Mode B on B leaders
    if (LEADER_A(set)) return false;            // force Mode A on A leaders
    // Followers: enable Mode B only if both global and local scores suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Periodic decay and selector update
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;
    // Decay TinyLFU and PC coldness (lazy, periodic)
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;  // halve
            if (pc_cold2[i] > 0) pc_cold2[i]--;    // approach neutral
        }
    }
    // Bandit epoch end
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) gsel_inc();
        else if (leaderB_hits_epoch < leaderA_hits_epoch) gsel_dec();
        leaderA_hits_epoch = leaderB_hits_epoch = 0;
    }
}

// Initialize replacement state
void InitReplacementState() {
    std::memset(rrpv, 0, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    std::memset(bandit_score, 0, sizeof(bandit_score));

    for (uint32_t s = 0; s < LLC_SETS; s++)
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            rrpv[s][w] = maxRRPV; // start as old

    // SHiP init (neutral)
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 16;
        shct_prefetch[i] = 16;
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    // 1 = neutral, gives room to go colder (2..3) and hotter (0)
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_cold2[i] = 1;

    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection with aging to maxRRPV
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// Update replacement state
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
    periodic_housekeeping();

    const bool ld_or_rfo = is_demand(type);

    // Per-set bandit slow drift toward global decision (followers only)
    if (!SEL_SAMPLED(set)) {
        if (GSEL >= GSEL_THRES) sat_inc_i8(bandit_score[set]);
        else                     sat_dec_i8(bandit_score[set]);
    }

    // Update PC popularity/coldness on demand references
    if (ld_or_rfo) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);        // TinyLFU
        if (hit) {
            // strengthen hotness on confirmed reuse
            if (hitcnt[set][way] + 1 >= HITS_PROMOTE_NS) pc_cold2[pidx] = 0;
            else if (pc_cold2[pidx] > 0) pc_cold2[pidx]--;
        } else {
            // miss: nudge toward cold
            sat_inc_u2(pc_cold2[pidx]);
        }
    }

    // Leader hit accounting for selector
    if (hit) {
        if (LEADER_A(set)) leaderA_hits_epoch++;
        else if (LEADER_B(set)) leaderB_hits_epoch++;
    }

    // Hits: gated promotion and RunLock handling
    if (hit) {
        if (ld_or_rfo) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                // Demote on touch unless it escapes by 2nd demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0; // escape quarantine
                    rrpv_set(set, way, 0);     // MRU
                } else {
                    rrpv_demote(set, way);     // push out scans
                }
            } else {
                // Non-stream: promote only on 2nd+ demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                }
            }
        } else {
            // Prefetch/writeback touches: never promote stream; soft demote to avoid pollution
            if (stream_lock[set][way]) rrpv_demote(set, way);
        }

        // SHiP reuse mark in leader-A sets
        if (LEADER_A(set) && ld_or_rfo) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // Miss path: prepare to insert; perform SHiP training on A leaders
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t psig = hawk_sig[slot][way];
        if (psig != 0) {
            uint32_t idx = shct_idx(psig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                else                      shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                else                      shct_dec(shct_demand[idx]);
            }
        }
    }

    // Stream detection (use current access to update PC stride state)
    bool is_stream = !is_writeback(type) && detect_and_update_stream(PC, paddr);
    bool useModeB  = modeB_enabled(set);

    uint8_t ins_rrpv = INSERT_WARM_DEPTH;
    uint8_t runlock  = 0;

    if (is_writeback(type)) {
        // never bypass writebacks; insert conservatively
        ins_rrpv = INSERT_COLD_DEPTH;
        runlock  = 0;
    } else if (useModeB) {
        uint32_t pidx = pc_index(PC);
        if (is_prefetch(type)) {
            // Prefetch: quarantine at tail; RunLock
            ins_rrpv = maxRRPV;
            runlock  = is_stream ? 1 : 0;
        } else {
            if (is_stream) {
                // Strong scan resistance, with zeusmp carve-out for hot PCs
                ins_rrpv = (pc_use4[pidx] >= PC_USE_HOT_THRESH) ? 5 : maxRRPV;
                runlock  = 1;
            } else {
                // PC coldness drives insertion
                ins_rrpv = (pc_cold2[pidx] >= 2) ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
                runlock  = 0;
            }
        }
    } else {
        // Mode A: Hawkeye-like SHiP
        uint16_t sig  = pc_sig12(PC);
        uint32_t idx  = shct_idx(sig);
        uint8_t  pred = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];

        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV; // quarantine prefetches
            runlock  = is_stream ? 1 : 0;
        } else {
            ins_rrpv = (pred <= 1) ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            if (is_stream) {
                // Respect clear stream scans even under Mode A
                ins_rrpv = maxRRPV;
                runlock  = 1;
            } else {
                runlock = 0;
            }
        }
        // Record new line training state for leader-A
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way]     = sig;
            hawk_used[slot][way]    = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    }

    // Apply insertion
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way]    = 0;
    stream_lock[set][way] = runlock;

    // For leader-B, nothing to train; for followers, training happens implicitly via bandit.
}

// Print end-of-simulation statistics
void PrintStats() {
    // KEEP BLANK
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // KEEP BLANK
}