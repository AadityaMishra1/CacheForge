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

// -------- Leader-set sampling (64 leaders) --------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// -------- Tunables (carefully tuned for lbm/mcf/gcc/omnetpp/astar balance) --------
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near tail
static constexpr uint8_t  PREFETCH_TAIL       = 7;    // quarantine
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;    // +1/+2 strides
static constexpr uint8_t  HITS_PROMOTE_MRU    = 2;    // MRU only on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_ON_TOUCH = 1; // demote quarantined lines on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;    // TinyLFU hot if >=
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048; // periodic decay
static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;   // followers prefer B if GSEL >= THRES
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set enable threshold

// -------- Per-line metadata (conceptual packing: rrpv:3b, hitcnt:2b, stream_lock:1b) --------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// -------- Per-set and global selectors --------
static int8_t  bandit_score[LLC_SETS]; // followers bias toward Mode B if >= MODEB_ENABLE_THRESH
static uint8_t GSEL = 0;               // global gate (0..31)
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

// -------- Mode A (Hawkeye-like SHiP) --------
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5b counters
static uint8_t shct_prefetch[SHCT_SIZE]; // 5b counters

// signatures + training flags for 64 leader-A sets only (64*16 entries)
static uint16_t hawk_sig[64][LLC_WAYS];  // store 12b effective (in 16b)
static uint8_t  hawk_used[64][LLC_WAYS]; // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // line filled by prefetch (0/1)

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// -------- Mode B (StreamGuardian++) --------
// 512-entry per-PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B-lined

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b id in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3) deadness/coldness

// -------- Helpers --------
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
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if both global and per-set suggest it
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// -------- Init --------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand,   0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig,      0, sizeof(hawk_sig));
    std::memset(hawk_used,     0, sizeof(hawk_used));
    std::memset(hawk_is_pref,  0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4,        0, sizeof(pc_use4));
    std::memset(pc_cold2,       0, sizeof(pc_cold2));
    GSEL = 0;
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
}

// -------- RRIP victim selection --------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes to avoid infinite loops
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: pick the way with max RRPV (deterministic)
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// -------- Periodic maintenance --------
static uint32_t op_count = 0;
static inline void periodic_housekeeping() {
    op_count++;
    // Decay TinyLFU and coldness
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1;
            if (pc_cold2[i] > 0) pc_cold2[i]--;
        }
    }
    // End of epoch: update global gate and per-set bandits
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) {
            if (GSEL < GSEL_MAX) GSEL++;
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_inc_i8(bandit_score[s]);
        } else if (leaderB_hits_epoch < leaderA_hits_epoch) {
            if (GSEL > 0) GSEL--;
            for (uint32_t s = 0; s < LLC_SETS; s++) sat_dec_i8(bandit_score[s]);
        }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// -------- ChampSim hooks --------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    return rrip_victim_and_age(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Ignore writebacks for replacement state
    if (is_writeback(type)) return;

    const bool demand   = is_demand(type);
    const bool prefetch = is_prefetch(type);

    // Update per-PC signals
    const uint32_t pcidx = pc_index(PC);
    if (demand) {
        sat_inc_u4(pc_use4[pcidx]);   // TinyLFU credit on any demand reference
        // coldness: credit on miss (handled at fill below), relief on demand hit
        if (hit) sat_dec_u2(pc_cold2[pcidx]);
    }

    // Stream detection (forward +1/+2) for this access
    const bool stream_detected = detect_and_update_stream(PC, paddr);

    // Update on hit
    if (hit) {
        // Count leader hits per mode (demand hits only)
        if (demand) {
            if (LEADER_A(set)) leaderA_hits_epoch++;
            else if (LEADER_B(set)) leaderB_hits_epoch++;
        }

        // Multi-hit promotion gating and stream quarantine handling
        if (stream_lock[set][way] && (STREAM_DEMOTE_ON_TOUCH != 0) && (hitcnt[set][way] < (HITS_PROMOTE_MRU - 0))) {
            // keep quarantined lines near tail until they prove reuse
            rrpv_demote(set, way);
        }

        if (demand) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++; // 2b saturating but guard
            if (LEADER_A(set)) {
                // Mode A training: mark reuse on hit for leader-A lines
                if (SEL_SAMPLED(set)) {
                    uint32_t slot = LEADER_SLOT(set);
                    hawk_used[slot][way] = 1;
                    // Positive reinforcement for the stored signature
                    uint16_t sig = hawk_sig[slot][way];
                    uint32_t idx = shct_idx(sig);
                    shct_inc(shct_demand[idx]);
                }
            }
        }

        // Only on/after 2nd demand hit promote to MRU and clear quarantine
        if (demand && hitcnt[set][way] >= HITS_PROMOTE_MRU) {
            stream_lock[set][way] = 0;    // escape quarantine
            rrpv_set(set, way, 0);        // MRU
        } else {
            // Gentle promotion for non-quarantined lines
            if (!stream_lock[set][way]) rrpv_promote(set, way);
        }

        periodic_housekeeping();
        return;
    }

    // Miss path (about to fill): train Mode A on eviction for leader-A set
    if (LEADER_A(set) && SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Evicted line's outcome
        if (hawk_is_pref[slot][way]) {
            // prefetch line: reward/penalize prefetch SHCT
            uint16_t sig_ev = hawk_sig[slot][way];
            uint32_t idx_ev = shct_idx(sig_ev);
            if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx_ev]);
            else                      shct_dec(shct_prefetch[idx_ev]);
        } else {
            uint16_t sig_ev = hawk_sig[slot][way];
            uint32_t idx_ev = shct_idx(sig_ev);
            if (hawk_used[slot][way]) shct_inc(shct_demand[idx_ev]);
            else                      shct_dec(shct_demand[idx_ev]);
        }
        // Initialize new line's training state
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = prefetch ? 1 : 0;
        hawk_sig[slot][way]     = pc_sig12(PC);
    }

    // Demand miss increases PC coldness
    if (demand) sat_inc_u2(pc_cold2[pcidx]);

    // Decide mode for followers (leaders are forced)
    const bool useModeB = modeB_enabled(set);

    // Compute insertion depth and quarantine flag
    uint8_t insert_rrpv = INSERT_COLD_DEPTH;
    uint8_t qlock = 0; // 1 => quarantined stream/prefetch

    if (prefetch) {
        // Prefetches always quarantine at tail
        insert_rrpv = PREFETCH_TAIL;
        qlock = 1;
    } else if (useModeB) {
        // Mode B: Stream + TinyLFU + coldness
        if (stream_detected) {
            // Aggressive tail insertion to protect lbm/milc
            // TinyLFU-hot PCs get a slightly shallower tail to preserve short reuse (e.g., zeusmp)
            if (pc_use4[pcidx] >= PC_USE_HOT_THRESH) insert_rrpv = std::min<uint8_t>(maxRRPV, (uint8_t)5);
            else                                      insert_rrpv = maxRRPV;
            qlock = 1; // quarantine; escape only after 2nd demand hit
        } else {
            bool pc_hot  = (pc_use4[pcidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pcidx] == 0);
            insert_rrpv  = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            qlock = 0;
        }
    } else {
        // Mode A: Hawkeye-like SHiP
        uint16_t sig  = pc_sig12(PC);
        uint32_t idx  = shct_idx(sig);
        if (demand) {
            // Friendly if SHCT > mid (15)
            insert_rrpv = (shct_demand[idx] > 15) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        } else {
            insert_rrpv = INSERT_COLD_DEPTH;
        }
        qlock = 0; // quarantine only in Mode B
    }

    // Apply insertion
    rrpv_set(set, way, insert_rrpv);
    stream_lock[set][way] = qlock;
    hitcnt[set][way] = 0;

    periodic_housekeeping();
}

void PrintStats_Heartbeat() {}
void PrintStats() {}