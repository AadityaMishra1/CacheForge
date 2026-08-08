#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types (fixed for CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (exactly 64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Hash: 64 leaders across the 2048 sets
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (retuned) -------------------------------------
static constexpr uint8_t  maxRRPV             = 7;    // 3b RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 forward steps before quarantine
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream/prefetch escape on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote stream-locked line on any touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 7;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 1024; // faster decay
static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch (accesses)
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers enable Mode B only if >=

// ---------------- Per-line metadata (packed conceptually) -----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b => 6 bits/line (stored in uint8_t arrays)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3 demand-hit counter
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];// 0/1 quarantine flag

// ---------------- Selector state -----------------------------------------
static uint8_t  GSEL = GSEL_THRES - 2;  // start slightly favoring Hawkeye
static uint32_t epoch_ctr = 0;

// Leader hit accounting (demand only)
static uint32_t leaderA_hits = 0, leaderB_hits = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) ---------------------
// Two 2K-entry x 5-bit SHCTs: demand and prefetch (global, trained in leader-A)
#define SHCT_BITS 11
#define SHCT_SIZE (1u << SHCT_BITS) // 2048
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Per-line signature/flags stored only for 64 leader-A sets (64x16 entries)
static uint16_t hawk_sig[64][LLC_WAYS];      // store 12-bit effective
static uint8_t  hawk_used[64][LLC_WAYS];     // 0/1 seen reuse
static uint8_t  hawk_is_pref[64][LLC_WAYS];  // 0/1 filled by prefetch
static uint8_t  hawk_valid[64][LLC_WAYS];    // 0/1 occupancy for training

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B (Stream-SMILE PC tables) -------------------------
// 512-entry PC tables (TinyLFU + coldness + last lineID + stream conf)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b (stored in uint16_t)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3)
static uint32_t lfu_decay_ctr = 0;

// ---------------- Helpers -------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) { if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++; }
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline void tiny_lfu_on_access(uint64_t PC, bool demand) {
    uint32_t idx = pc_index(PC);
    if (demand) {
        sat_inc_u4(pc_use4[idx]);
        // strengthen coldness on reuse
        sat_inc_u2(pc_cold2[idx]);
    }
}

static inline void tiny_lfu_decay() {
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_use4[i] >>= 1;           // halve
        if (pc_cold2[i] > 0) pc_cold2[i]--; // mild decay
    }
}

static inline bool use_modeB(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    return (GSEL >= GSEL_THRES);
}

// ---------------- API: Initialization ------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
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
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    GSEL = GSEL_THRES - 2;
    epoch_ctr = 0;
    leaderA_hits = leaderB_hits = 0;
    lfu_decay_ctr = 0;
}

// ---------------- API: Victim selection ----------------------------------
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

    // RRIP: search for a line with rrpv == maxRRPV; if none, age all and retry (bounded)
    for (uint32_t round = 0; round <= maxRRPV; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }

    // Fallback (should not happen)
    return 0;
}

// ---------------- API: Update replacement state --------------------------
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
    epoch_ctr++;
    lfu_decay_ctr++;
    if (lfu_decay_ctr >= LFU_DECAY_PERIOD) {
        tiny_lfu_decay();
        lfu_decay_ctr = 0;
    }

    // Selector epoch maintenance (compare leader hits, then adjust GSEL)
    if (epoch_ctr % BANDIT_EPOCH == 0) {
        if (leaderB_hits + 2 < leaderA_hits) { // A leads with margin
            if (GSEL > 0) GSEL--;
        } else if (leaderA_hits + 2 < leaderB_hits) { // B leads with margin
            if (GSEL < GSEL_MAX) GSEL++;
        }
        leaderA_hits = leaderB_hits = 0;
    }

    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    bool modeB   = use_modeB(set);

    // Demand hit accounting for leaders
    if (hit && is_demand(type)) {
        if (leaderA) leaderA_hits++;
        else if (leaderB) leaderB_hits++;
    }

    // Hawkeye per-line training (only leader-A sets)
    if (leaderA) {
        uint32_t slot = LEADER_SLOT(set);
        if (hit) {
            // mark reuse
            hawk_used[slot][way] = 1;
        } else {
            // On a fill, first train the evicted occupant in this way
            if (hawk_valid[slot][way]) {
                uint16_t sig = hawk_sig[slot][way];
                uint32_t idx = shct_idx(sig);
                if (hawk_is_pref[slot][way]) {
                    if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                    else                      shct_dec(shct_prefetch[idx]);
                } else {
                    if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                    else                      shct_dec(shct_demand[idx]);
                }
            }
            // Then install new metadata for incoming line
            hawk_sig[slot][way]     = pc_sig12(PC);
            hawk_used[slot][way]    = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
            hawk_valid[slot][way]   = 1;
        }
    }

    // Stream-touch demotion while quarantined
    if (hit && stream_lock[set][way]) {
        for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);
    }

    // Multi-hit promotion/escape logic on hits
    if (hit) {
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (stream_lock[set][way]) {
                // Escape from quarantine on 2nd demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, INSERT_WARM_DEPTH); // shallow depth
                }
            } else {
                // Non-stream: promote to MRU only on 2nd+ demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                } else {
                    // gentle nudge
                    rrpv_promote(set, way);
                }
            }
        } else {
            // Prefetch hit: never early-promote
            rrpv_promote(set, way);
        }
        return;
    }

    // Miss/Fill path
    // Update TinyLFU on demand access
    if (!is_writeback(type))
        tiny_lfu_on_access(PC, is_demand(type));

    bool stream_now = false;
    if (!is_writeback(type))
        stream_now = detect_and_update_stream(PC, paddr);

    // Choose insertion strategy by mode and access type
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t lock_flag = 0;

    if (is_writeback(type)) {
        // Never bypass/delay writebacks; keep modest priority
        ins_depth = INSERT_WARM_DEPTH;
        lock_flag = 0;
    } else if (modeB) {
        if (is_prefetch(type)) {
            // Prefetches always quarantined at tail
            ins_depth = maxRRPV;
            lock_flag = 1;
        } else if (stream_now) {
            // Stream quarantine: hard tail, demote-on-touch; (soft-bypass via tail)
            ins_depth = maxRRPV;
            lock_flag = 1;
        } else {
            // Non-stream: TinyLFU-biased insertion
            uint32_t pidx = pc_index(PC);
            bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH) || (pc_cold2[pidx] >= 2);
            ins_depth = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock_flag = 0;
        }
    } else {
        // Mode A (Hawkeye-like): SHCT-driven insertion; prefetches quarantined
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_prefetch(type)) {
            ins_depth = (shct_prefetch[idx] > (SHCT_MAX >> 1)) ? INSERT_WARM_DEPTH : maxRRPV;
            lock_flag = 1; // quarantine prefetches
        } else {
            ins_depth = (shct_demand[idx] > (SHCT_MAX >> 1)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock_flag = 0;
        }
    }

    // Install new state
    rrpv_set(set, way, ins_depth);
    stream_lock[set][way] = lock_flag;
    hitcnt[set][way] = 0;
}

// ---------------- Stats (silent) -----------------------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}