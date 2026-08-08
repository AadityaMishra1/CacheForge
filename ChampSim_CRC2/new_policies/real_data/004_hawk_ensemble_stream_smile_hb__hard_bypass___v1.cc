#include <cstdint>
#include <cstring>
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

// ---------------- Leader-set sampling (64 leaders total) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // hash to 64 leaders uniformly
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (retuned for lbm/zeusmp + irregulars) -------
static constexpr uint8_t  maxRRPV             = 7;    // 3b RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 forward steps before quarantine
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream: escape quarantine on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote stream-locked line on any touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 7;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 1024; // faster decay
static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers enable Mode B only if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set bias to enable Mode B

// ---------------- Per-line metadata (packed conceptually) -----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b => 6 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];// 0/1

// ---------------- Selector state -----------------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] local bias toward Mode B for followers
static uint8_t GSEL = GSEL_THRES - 2;  // start slightly favoring Hawkeye
static uint32_t epoch_ctr = 0;
static uint32_t lfu_decay_ctr = 0;

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

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3)

// ---------------- Helpers -------------------------------------------------
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

static inline bool modeB_allowed_for_followers(uint32_t set) {
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}
static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return modeB_allowed_for_followers(set);
}

// ---------------- ChampSim interface -------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        // mildly neutral initialization
        shct_demand[i] = 15;
        shct_prefetch[i] = 15;
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
            hawk_valid[ls][w] = 0;
        }
    }

    GSEL = GSEL_THRES - 2;
    epoch_ctr = 0;
    lfu_decay_ctr = 0;
    leaderA_hits = leaderB_hits = 0;
}

// Victim selection: SRRIP with insertion at max, age until a victim appears
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return any invalid way first
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP victim search
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age everyone by +1 (saturate)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
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
    // Global periodic maintenance
    epoch_ctr++;
    lfu_decay_ctr++;
    if (lfu_decay_ctr >= LFU_DECAY_PERIOD) {
        lfu_decay_ctr = 0;
        // TinyLFU decay + coldness relaxation
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i]--;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
            if (pc_stream_conf[i] > 0) pc_stream_conf[i]--; // decay stream confidence slowly
        }
        // Bandit gate hysteresis every epoch by leader hit comparison
        if (epoch_ctr >= BANDIT_EPOCH) {
            // require margin to switch
            if (leaderB_hits + 8 < leaderA_hits) {
                if (GSEL > 0) GSEL--;
            } else if (leaderB_hits > leaderA_hits + 8) {
                if (GSEL < GSEL_MAX) GSEL++;
            }
            leaderA_hits = leaderB_hits = 0;
            epoch_ctr = 0;
        }
    }

    // Update TinyLFU / cold guard on demand events
    uint32_t pc_idx = pc_index(PC);
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_idx]);
    }

    // Stream detection runs on every ref (load/RFO/pf), ignored for WB
    bool stream_pc = (!is_writeback(type)) ? detect_and_update_stream(PC, paddr) : false;

    // Mode selection for this set
    bool modeB = use_modeB(set);

    // Hits: apply promotion/demotion rules
    if (hit) {
        if (is_demand(type)) {
            // leader hit accounting for bandit (demand hits only)
            if (LEADER_A(set)) leaderA_hits++;
            if (LEADER_B(set)) leaderB_hits++;
        }

        // Update per-set bias if follower (simple win/loss on demand)
        if (!LEADER_A(set) && !LEADER_B(set) && is_demand(type)) {
            if (modeB) sat_inc_i8(bandit_score[set]); // reward on demand hit when using Mode B
            else {
                // gentle decay toward enabling decision stability
                if (bandit_score[set] > 0) bandit_score[set]--;
                else if (bandit_score[set] < 0) bandit_score[set]++;
            }
        }

        // Stream-locked lines: demote on touch; escape on 2nd demand hit
        if (stream_lock[set][way]) {
            // demote on any touch to keep streams cold
            for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);

            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    // escape quarantine to shallow depth; still no MRU
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, INSERT_WARM_DEPTH);
                    hitcnt[set][way] = 1;
                }
            }
            return;
        }

        // Non-stream-locked: multi-hit gating w/ cold PC guard
        if (is_demand(type)) {
            bool cold_pc = (pc_use4[pc_idx] < PC_USE_HOT_THRESH) || (pc_cold2[pc_idx] >= 2);
            uint8_t need = cold_pc ? (uint8_t)3 : HITS_PROMOTE_NS;

            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= need) {
                rrpv_set(set, way, 0); // MRU
                hitcnt[set][way] = need; // cap
            } else {
                // gentle promotion
                rrpv_promote(set, way);
            }
        } else {
            // prefetch touch: no early promotion
            rrpv_promote(set, way);
        }

        // Train Hawkeye reuse in leader-A sets
        if (LEADER_A(set)) {
            uint32_t ls = LEADER_SLOT(set);
            hawk_used[ls][way] = 1;
        }

        // Coldness improves on any hit
        if (is_demand(type)) {
            sat_dec_u2(pc_cold2[pc_idx]);
        }
        return;
    }

    // Miss / insertion path
    if (is_demand(type)) {
        // Demand miss -> coldness increases
        sat_inc_u2(pc_cold2[pc_idx]);
    }

    // Train Hawkeye on eviction of previous line in leader-A set
    if (LEADER_A(set)) {
        uint32_t ls = LEADER_SLOT(set);
        if (hawk_valid[ls][way]) {
            uint16_t psig = hawk_sig[ls][way];
            uint32_t idx = shct_idx(psig);
            if (hawk_is_pref[ls][way]) {
                if (hawk_used[ls][way]) shct_inc(shct_prefetch[idx]);
                else                    shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[ls][way]) shct_inc(shct_demand[idx]);
                else                    shct_dec(shct_demand[idx]);
            }
        }
    }

    // Insertion decision
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t set_stream_lock = 0;

    if (is_writeback(type)) {
        // never bypass writebacks; keep them reasonably warm
        ins_rrpv = INSERT_WARM_DEPTH;
        set_stream_lock = 0;
    } else if (modeB) {
        // Mode B: Hard quarantine for streams and prefetch; multi-hit escape only
        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV;
            set_stream_lock = 1;
        } else if (stream_pc) {
            ins_rrpv = maxRRPV; // hard-tail to approximate bypass
            set_stream_lock = 1;
        } else {
            // non-stream: use PC hotness/coldness
            bool hot_pc  = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH) && (pc_cold2[pc_idx] == 0);
            ins_rrpv = hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            set_stream_lock = 0;
        }
    } else {
        // Mode A (Hawkeye-like): SHCT-guided insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_prefetch(type)) {
            ins_rrpv = (shct_prefetch[idx] >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        } else {
            ins_rrpv = (shct_demand[idx]  >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
        set_stream_lock = 0;
    }

    rrpv_set(set, way, ins_rrpv);
    stream_lock[set][way] = set_stream_lock;
    hitcnt[set][way] = 0;

    // Record new Hawkeye line metadata only in leader-A sets
    if (LEADER_A(set)) {
        uint32_t ls = LEADER_SLOT(set);
        hawk_sig[ls][way]      = pc_sig12(PC);
        hawk_is_pref[ls][way]  = is_prefetch(type) ? 1 : 0;
        hawk_used[ls][way]     = 0;
        hawk_valid[ls][way]    = 1;
    }

    // Update follower per-set bandit on demand misses (penalize Mode B)
    if (!LEADER_A(set) && !LEADER_B(set) && is_demand(type)) {
        if (modeB) sat_dec_i8(bandit_score[set]);
        else {
            // gentle decay toward neutral to allow switching if B later wins
            if (bandit_score[set] > 0) bandit_score[set]--;
            else if (bandit_score[set] < 0) bandit_score[set]++;
        }
    }
}

void PrintStats() { }
void PrintStats_Heartbeat() { }