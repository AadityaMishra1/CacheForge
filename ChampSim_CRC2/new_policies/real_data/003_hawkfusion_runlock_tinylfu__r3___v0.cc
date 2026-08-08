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

static inline bool is_demand(uint32_t t)   { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t) { return (t == ACCESS_PREFETCH); }

// ---------------- Leader-set sampling (64 leaders total) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (balanced for lbm, mcf, gcc/omnetpp, astar) ----
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail insertion
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 stride run confidence
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;    // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream: escape quarantine on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;    // demote stream-locked lines on each touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;    // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // periodic decay of TinyLFU/coldness
static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch (ops)
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set bandit enable threshold

// ---------------- Per-line metadata (conceptually bit-packed) -------------
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Selectors (global + per-set bandit) ---------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] bias for Mode B
static uint8_t GSEL = 0;               // global gate (0..31)

// ---------------- Mode A: Hawkeye-like SHiP fallback ----------------------
// Tiny SHCTs (demand & prefetch), trained on leader-A sets
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // low-cost mix for 12-bit signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Per-line training only for leader sets (64 x 16)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B: RunLock Stream Shield + TinyLFU ------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b line id (stored in 16b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// ---------------- Helpers --------------------------------------------------
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
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;   // Mode B on B-leaders
    if (LEADER_A(set)) return false;  // Mode A on A-leaders
    // Followers: enable Mode B only if both global and local suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Housekeeping: LFU decay and selector updates ------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;
    // Periodic TinyLFU and coldness decay
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
        }
    }
    // Selector epoch update
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) {
            if (GSEL < GSEL_MAX) GSEL++;
        } else if (leaderA_hits_epoch > leaderB_hits_epoch) {
            if (GSEL > 0) GSEL--;
        }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// ---------------- Init -----------------------------------------------------
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
    op_count = leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Victim selection: RRIP with bounded aging ---------------
static inline uint32_t rrip_find_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Age until some reach maxRRPV (bounded passes guarantee termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: choose the line with largest RRPV
    uint8_t best = 0;
    uint32_t victim = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    return rrip_find_victim(set, current_set);
}

// ---------------- Update replacement state --------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    periodic_housekeeping();

    // Ignore writebacks for predictor/selector updates; do not bypass WB
    if (type == ACCESS_WRITEBACK) {
        if (!hit) {
            // Insert conservative for writebacks
            rrpv_set(set, way, INSERT_COLD_DEPTH);
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 0;
        }
        return;
    }

    // Determine which mode is active for this set
    bool useModeB = modeB_enabled(set);

    // Leader hit accounting for selector (safe: leaders are forced modes)
    if (hit) {
        if (LEADER_A(set)) leaderA_hits_epoch++;
        if (LEADER_B(set)) leaderB_hits_epoch++;
    }

    // Bandit scoring for followers (reward B hits, penalize B misses)
    if (!SEL_SAMPLED(set)) {
        if (useModeB) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else     sat_dec_i8(bandit_score[set]);
        }
    }

    // PC tables: update LFU and stream detector for all non-WB accesses
    uint32_t pcidx = pc_index(PC);
    if (is_demand(type)) sat_inc_u4(pc_use4[pcidx]);
    bool is_stream = detect_and_update_stream(PC, paddr);

    // Handle hits: promotion with multi-hit gating; RunLock demotion
    if (hit) {
        // Multi-hit gating
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        // Mark reuse for leader-A training
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
            // Positive reinforcement on demand hits
            if (is_demand(type)) {
                uint16_t sig = hawk_sig[slot][way];
                uint32_t idx = shct_idx(sig);
                shct_inc(shct_demand[idx]);
            }
        }

        // Demote quarantined stream lines on touch
        if (STREAM_DEMOTE_TOUCH && stream_lock[set][way]) rrpv_demote(set, way);

        // Promote only after sufficient demand hits
        if (is_demand(type)) {
            uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                rrpv_set(set, way, 0); // MRU
                if (stream_lock[set][way]) stream_lock[set][way] = 0; // escape quarantine
            }
        }
        return;
    }

    // Miss: perform negative training (before overwrite) for leader-A sets
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t old_sig = hawk_sig[slot][way];
        uint8_t  old_pref = hawk_is_pref[slot][way];
        uint8_t  old_used = hawk_used[slot][way];

        if (old_sig != 0) {
            uint32_t idx = shct_idx(old_sig);
            if (old_pref) shct_dec(shct_prefetch[idx]);
            else          shct_dec(shct_demand[idx]);

            // Train PC coldness: dead (no reuse) => increment coldness; reused => decay
            if (!old_used) {
                if (pc_cold2[pc_index((uint64_t)old_sig)] < 3) pc_cold2[pc_index((uint64_t)old_sig)]++;
            } else {
                if (pc_cold2[pc_index((uint64_t)old_sig)] > 0) pc_cold2[pc_index((uint64_t)old_sig)]--;
            }
        }
        // Record new line metadata for future training
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]    = 0;
    }

    // Decide insertion for the new line
    uint8_t depth = INSERT_COLD_DEPTH;
    uint8_t lock  = 0;

    if (useModeB) {
        // Mode B: Stream shield + TinyLFU + coldness
        if (is_prefetch(type)) {
            depth = maxRRPV; // quarantine all prefetches
            lock  = 1;
        } else if (is_stream) {
            // Aggressive quarantine; allow hot PCs slightly shallower tail for short reuse
            bool hot = (pc_use4[pcidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pcidx] <= 1);
            depth = hot ? (uint8_t)5 : maxRRPV;
            lock  = 1;
        } else {
            bool hot = (pc_use4[pcidx] >= PC_USE_HOT_THRESH);
            bool cold = (pc_cold2[pcidx] >= 2);
            depth = (hot && !cold) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock  = 0;
        }
    } else {
        // Mode A: Hawkeye-like SHiP insertion (prefetch always quarantined)
        if (is_prefetch(type)) {
            depth = maxRRPV;
            lock  = 1;
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            // Friendly if counter is warm (midpoint threshold)
            bool friendly = (shct_demand[idx] >= (SHCT_MAX >> 1));
            depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock  = 0;
        }
    }

    rrpv_set(set, way, depth);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock;
}

// ---------------- Stats (keep minimal) ------------------------------------
void PrintStats_Heartbeat() {}
void PrintStats() {}