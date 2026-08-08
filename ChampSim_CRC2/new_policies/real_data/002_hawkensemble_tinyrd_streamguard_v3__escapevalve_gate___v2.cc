#include <cstdint>
#include <cstring>
#include <algorithm>
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

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (v3 EscapeValve+Gate) ------------------
static constexpr uint8_t  maxRRPV             = 7;     // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;     // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;     // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;     // +1/+2 forward steps needed
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;     // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 3;     // stream escape threshold (default)
static constexpr uint8_t  PC_USE_HOT_THRESH   = 5;     // TinyLFU hot PCs (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;  // periodic decay of PC TinyLFU
static constexpr uint32_t BANDIT_EPOCH        = 4096;  // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;    // global gate range
static constexpr uint8_t  GSEL_THRES          = 20;    // followers choose Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 3;     // per-slot enable threshold

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b (stream/quarantine mark)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Selector: per-slot bandit + global gate --------
static int8_t   bandit_score_slot[64];     // [-8..7]
static uint16_t leader_hits_A_slot[64];    // epoch counters
static uint16_t leader_hits_B_slot[64];
static uint8_t  GSEL = 0;                  // global preference 0..31
static uint64_t access_ctr = 0;            // global access counter

// ---------------- Mode A (Hawkeye-like SHiP) ---------------------
// Tiny SHCT (1024 x 5b)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS)
#define SHCT_MAX 31
static uint8_t shct[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // compact 12b signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature + training flags per A-leader slot
static uint16_t hawk_sig_slot[64][LLC_WAYS];   // 12b effective
static uint8_t  hawk_used_slot[64][LLC_WAYS];  // reuse seen
static uint8_t  hawk_pref_slot[64][LLC_WAYS];  // filled by prefetch

// ---------------- Mode B (TinyRD-StreamGuard v3) -----------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
// 64B lines -> 10b line-id per PC
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); }

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_dead2[PC_TBL_SIZE];       // 2b deadness (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    // +1/+2 forward run detector with hysteresis (decay on break)
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (fwd) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        if (pc_stream_conf[idx] > 0) pc_stream_conf[idx]--; // hysteresis
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline uint32_t pseudo_hash10(uint64_t PC, uint64_t paddr) {
    uint64_t x = PC ^ (PC >> 17) ^ (paddr >> 6) ^ 0x9E3779B97F4A7C15ull;
    return (uint32_t)(x % 10u);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;       // force Mode B on B leaders
    if (LEADER_A(set)) return false;      // force Mode A on A leaders
    uint32_t slot = LEADER_SLOT(set);
    return (GSEL >= GSEL_THRES) && (bandit_score_slot[slot] >= MODEB_ENABLE_THRESH);
}

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_inc_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    rrpv[set][way] = 0;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

// ---------------- Core interface ---------------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv)); // default to far RRPV
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));

    std::memset(shct, 0, sizeof(shct));
    std::memset(hawk_sig_slot, 0, sizeof(hawk_sig_slot));
    std::memset(hawk_used_slot, 0, sizeof(hawk_used_slot));
    std::memset(hawk_pref_slot, 0, sizeof(hawk_pref_slot));

    std::memset(bandit_score_slot, 0, sizeof(bandit_score_slot));
    std::memset(leader_hits_A_slot, 0, sizeof(leader_hits_A_slot));
    std::memset(leader_hits_B_slot, 0, sizeof(leader_hits_B_slot));
    GSEL = 0;
    access_ctr = 0;

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_dead2, 0, sizeof(pc_dead2));
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP victim: search for maxRRPV; age if not found
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] >= maxRRPV) return w;
        }
        rrpv_inc_all(set);
    }
    // unreachable
    // return 0;
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
    access_ctr++;

    // Periodic TinyLFU decay
    if ((access_ctr % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
        }
    }
    // Periodic bandit epoch update
    if ((access_ctr % BANDIT_EPOCH) == 0) {
        int betterB = 0, betterA = 0;
        for (uint32_t s = 0; s < 64; s++) {
            if (leader_hits_B_slot[s] > leader_hits_A_slot[s]) { sat_inc_i8(bandit_score_slot[s]); betterB++; }
            else if (leader_hits_B_slot[s] < leader_hits_A_slot[s]) { sat_dec_i8(bandit_score_slot[s]); betterA++; }
            leader_hits_A_slot[s] = leader_hits_B_slot[s] = 0;
        }
        if (betterB > betterA) { if (GSEL < GSEL_MAX) GSEL++; }
        else if (betterA > betterB) { if (GSEL > 0) GSEL--; }
    }

    // Update TinyLFU on any demand access (hit or miss)
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    // Evaluate stream detector (used by Mode B)
    bool stream_conf = detect_and_update_stream(PC, paddr);
    uint32_t pidx = pc_index(PC);

    // Decide active mode for this set
    bool forceA = LEADER_A(set);
    bool forceB = LEADER_B(set);
    bool useModeB = modeB_enabled(set);

    if (hit) {
        // Demand hit handling
        if (is_demand(type)) {
            if (forceA || (!forceB && !useModeB)) {
                // Mode A hit: SHiP-style promote, but keep multi-hit gate (irregular protection)
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                // Stream quarantine honored
                if (stream_lock[set][way]) {
                    // demote-on-touch until escape
                    rrpv_demote(set, way);
                    uint8_t thr = (pc_use4[pidx] >= PC_USE_HOT_THRESH) ? HITS_PROMOTE_NS : HITS_PROMOTE_STR;
                    if (hitcnt[set][way] >= thr) {
                        stream_lock[set][way] = 0;
                        rrpv_promote(set, way);
                    }
                } else {
                    // Non-stream: promote only on 2nd+ demand hit
                    if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                        rrpv_promote(set, way);
                    } else {
                        // gentle nudge
                        if (rrpv[set][way] > 0) rrpv[set][way]--;
                    }
                }
                // A-leader reuse training
                if (forceA) {
                    uint32_t slot = LEADER_SLOT(set);
                    hawk_used_slot[slot][way] = 1;
                    leader_hits_A_slot[slot]++; // count successful hits for A
                }
            } else {
                // Mode B hit: multi-hit gate + escape valve
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (stream_lock[set][way]) {
                    rrpv_demote(set, way); // quarantine demotion
                    uint8_t thr = (pc_use4[pidx] >= PC_USE_HOT_THRESH) ? HITS_PROMOTE_NS : HITS_PROMOTE_STR;
                    if (hitcnt[set][way] >= thr) {
                        stream_lock[set][way] = 0;
                        rrpv_promote(set, way); // escape to MRU
                    }
                } else {
                    if (hitcnt[set][way] >= HITS_PROMOTE_NS) rrpv_promote(set, way);
                    else if (rrpv[set][way] > 0) rrpv[set][way]--;
                }
                // Mode B hit credit
                if (forceB) {
                    uint32_t slot = LEADER_SLOT(set);
                    leader_hits_B_slot[slot]++;
                }
                // Deadness learns from reuse
                sat_dec_u2(pc_dead2[pidx]);
            }
        } else {
            // Non-demand hit (e.g., prefetch hit): never promote, but demote quarantined streams
            if (stream_lock[set][way]) rrpv_demote(set, way);
        }
        return;
    }

    // Miss / fill path
    // If WRITEBACK, never bypass; insert moderately cold
    if (is_writeback(type)) {
        rrpv_set(set, way, INSERT_COLD_DEPTH);
        hitcnt[set][way] = 0;
        stream_lock[set][way] = 0;
        return;
    }

    // Before overwriting, do Hawkeye training on A-leader evictions
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t old_sig = hawk_sig_slot[slot][way];
        uint32_t idx = shct_idx(old_sig);
        if (hawk_sig_slot[slot][way] != 0 || hawk_pref_slot[slot][way] || hawk_used_slot[slot][way]) {
            if (hawk_used_slot[slot][way]) shct_inc(shct[idx]);
            else                           shct_dec(shct[idx]);
        }
    }

    // Decide insertion policy for this miss
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t lock = 0;

    if (forceA || (!forceB && !useModeB)) {
        // Mode A: SHiP-guided insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        bool pred_hot = (shct[idx] >= (SHCT_MAX / 2));
        ins_rrpv = pred_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        lock = 0;

        // Record new line’s sig/flags in A-leaders
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig_slot[slot][way]  = sig;
            hawk_used_slot[slot][way] = 0;
            hawk_pref_slot[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    } else {
        // Mode B: StreamGuard + PC gates
        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV; // quarantine
            lock = 1;
        } else {
            if (stream_conf) {
                // EscapeValve: approximate 90% bypass via hard-tail quarantine
                bool bypass_like = (pseudo_hash10(PC, paddr) < 9);
                (void)bypass_like; // approximated by always hard-tail inserting
                ins_rrpv = maxRRPV;
                lock = 1;
            } else {
                // PC deadness + TinyLFU gating
                bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
                bool pc_deadish = (pc_dead2[pidx] >= 2);
                if (pc_deadish && !pc_hot) {
                    ins_rrpv = INSERT_COLD_DEPTH;
                    lock = 0;
                } else {
                    ins_rrpv = INSERT_WARM_DEPTH;
                    lock = 0;
                }
            }
        }
        // B-leaders don't need extra bookkeeping here
        // Deadness learns from fills that are likely dead (non-stream demand)
        if (is_demand(type) && !stream_conf) sat_inc_u2(pc_dead2[pidx]);
    }

    // Commit insertion state
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock;
}

void PrintStats() {
    // keep blank
}

void PrintStats_Heartbeat() {
    // keep blank
}