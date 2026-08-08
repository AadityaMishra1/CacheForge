#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

// ChampSim CRC2 constants
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

// Leader-set sampling (64 leaders total: 32 A, 32 B)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Tunables (streaming balance / gating)
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot PCs
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// periodic LFU/coldness decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;// global selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;  // followers prefer B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set enable threshold

// Per-line metadata (conceptual packing: rrpv:3, hitcnt:2, stream_lock:1)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// Per-set local selector (bandit)
static int8_t  bandit_score[LLC_SETS]; // [-8..7]

// Global selector
static uint8_t GSEL = 0;

// ------------- Mode A: Hawkeye-like SHiP fallback (safe) -------------
// Tiny SHCTs (1024 entries, 5-bit counters) for demand & prefetch
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Training state only for 64 leader sets (A side only)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse observed
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // was prefetched at fill

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ------------- Mode B: Prism (stream-safe DBP path) -------------------
// Compact per-PC tables (512 entries)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line id (10b)

static uint16_t pc_last_line10[PC_TBL_SIZE]; // last 10b line
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // PC deadness (0..3)

// Small helpers
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
    if (LEADER_B(set)) return true;            // always test Mode B on B-leaders
    if (LEADER_A(set)) return false;           // always test Mode A on A-leaders
    // Followers: enable Mode B only if both global and local gates say so
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
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
        }
    }
    // Update global selector
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) gsel_inc();
        else gsel_dec();
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// ---------------- Initialization ---------------------------------
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
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
}

// ---------------- RRIP victim selection --------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes to ensure termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback (should not happen)
    return 0;
}

// ---------------- Victim selection (with A-leader training) ------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Always return a way in [0, LLC_WAYS-1]; never bypass here
    // Prefer invalid if available
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Find victim via RRIP; perform bounded aging if needed
    uint32_t victim = rrip_victim_and_age(set, current_set);

    // Mode A (Hawkeye-like) negative training on A-leader eviction if no reuse
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig = hawk_sig[slot][victim];
        uint32_t idx = shct_idx(sig);
        if (hawk_is_pref[slot][victim]) {
            if (hawk_used[slot][victim] == 0) shct_dec(shct_prefetch[idx]);
        } else {
            if (hawk_used[slot][victim] == 0) shct_dec(shct_demand[idx]);
        }
        // Reset reuse marker for the new line that will fill here
        hawk_used[slot][victim] = 0;
    }
    return victim;
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    periodic_housekeeping();

    // Ignore writebacks for replacement state (never bypass on WB)
    if (is_writeback(type)) return;

    uint32_t pc_idx = pc_index(PC);
    bool follower_modeB = modeB_enabled(set);

    if (hit) {
        // Track leader hit statistics for selector (A/B)
        if (LEADER_A(set)) leaderA_hits_epoch++;
        if (LEADER_B(set)) leaderB_hits_epoch++;

        // PC usefulness and coldness updates on demand hits
        if (is_demand(type)) {
            sat_inc_u4(pc_use4[pc_idx]);
            if (pc_cold2[pc_idx] > 0) pc_cold2[pc_idx]--;
        }

        // Mark reuse for A-leaders for positive SHCT reinforcement
        if (LEADER_A(set) && is_demand(type)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
            // Optional positive reinforcement for the PC's table
            uint16_t sig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
            else                         shct_inc(shct_demand[idx]);
        }

        // Multi-hit gating + stream demote-on-touch
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++; // 2b cap at 3
        }
        if (stream_lock[set][way]) {
            // Stream-locked lines: escape on 2nd demand hit, else demote-on-touch
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_STR) {
                rrpv_set(set, way, 0);
                stream_lock[set][way] = 0;
            } else {
                rrpv_demote(set, way);
            }
        } else {
            // Non-stream lines: promote only on 2nd+ demand hit
            if (is_demand(type) && hitcnt[set][way] >= HITS_PROMOTE_NS) {
                rrpv_set(set, way, 0);
            } else {
                // gentle promotion for recency
                rrpv_promote(set, way);
            }
        }
        return;
    }

    // Miss/Fill path (choose insertion policy)
    // Stream detection (always tracked, used only if Mode B is active)
    bool is_stream = detect_and_update_stream(PC, paddr);

    // Update bandit: sets trending to streams favor Mode B; others do not
    if (is_stream) sat_inc_i8(bandit_score[set]);
    else           sat_dec_i8(bandit_score[set]);

    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t ins_stream_lock = 0;

    if (follower_modeB) {
        // Mode B: Prism stream-safe + PC-guided usefulness
        bool hot_pc  = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);
        bool pc_cold = (pc_cold2[pc_idx] >= 2);

        if (is_prefetch(type)) {
            // quarantine all prefetch at tail
            ins_rrpv = maxRRPV;
            ins_stream_lock = 1;
        } else if (is_stream) {
            // streaming: hard tail; allow shallow tail for hot PCs to preserve short reuse
            ins_rrpv = hot_pc ? (uint8_t)5 : maxRRPV;
            ins_stream_lock = 1;
        } else {
            // non-stream: cold-insert noisy PCs, warm-insert hot ones
            ins_rrpv = (hot_pc && !pc_cold) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            ins_stream_lock = 0;
        }

        // Update PC deadness on miss
        if (is_demand(type)) sat_inc_u2(pc_cold2[pc_idx]);
    } else {
        // Mode A: Hawkeye-like SHiP fallback
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);

        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way]      = sig;
            hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
            hawk_used[slot][way]     = 0;
        }

        if (is_prefetch(type)) {
            ins_rrpv = maxRRPV; // prefetches always at tail (low priority)
        } else {
            // Friendly vs averse by SHCT
            bool friendly = (shct_demand[idx] > (SHCT_MAX >> 1));
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
        ins_stream_lock = 0;
    }

    // Apply insertion
    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = ins_stream_lock;
}

// ---------------- Stats (silent) ---------------------------------
void PrintStats_Heartbeat() {}
void PrintStats() {}