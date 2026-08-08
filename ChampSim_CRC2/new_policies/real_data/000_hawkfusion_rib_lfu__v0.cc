#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 sets) -------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (balanced for lbm + irregulars) -------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;   // hard tail
static constexpr uint8_t  STREAM_SHALLOW_TAIL = 5;   // for TinyLFU-hot streams
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 3;   // stream promote on 3rd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;   // demote stream lines on touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 8;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH   = 2;   // 2b coldness (>=2 => cold)
static constexpr uint32_t LFU_DECAY_PERIOD    = 1024;// fast decay
static constexpr uint32_t BANDIT_EPOCH        = 8192;// selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 20;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set bias threshold

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set/global selectors -----------------------
static int8_t   bandit_score[LLC_SETS]; // [-8..7]
static uint8_t  GSEL = 0;               // global gate
static uint64_t access_count = 0;
static int32_t  leaderA_score = 0;      // hits - misses
static int32_t  leaderB_score = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b PC signature hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader A/B per-line buffers for training (only 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

// ---------------- Mode B (Run-Intention Bypass + TinyLFU) --------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B -> 10b

// Per-PC state: 10b last_line, 2b stream_conf, 4b use, 2b coldness
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE];
static uint8_t  pc_use4[PC_TBL_SIZE];
static uint8_t  pc_cold2[PC_TBL_SIZE];

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_u5(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// +1/+2 forward-run detector with 2-step confidence; reset on non-forward
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else         pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    // Followers: conservative enable
    if (GSEL < GSEL_THRES) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig,     0, sizeof(hawk_sig));
    std::memset(hawk_used,    0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4,        0, sizeof(pc_use4));
    std::memset(pc_cold2,       0, sizeof(pc_cold2));

    GSEL = 0;
    access_count = 0;
    leaderA_score = leaderB_score = 0;
}

// ---------------- Victim selection (RRIP + Mode A training) -------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging to guarantee termination
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

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    uint32_t v = rrip_victim_and_age(set, current_set);

    // Train Mode A predictor on sampled sets upon eviction
    if (SEL_SAMPLED(set) && current_set[v].valid) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t sig  = hawk_sig[slot][v];
        uint8_t  used = hawk_used[slot][v];
        uint8_t  was_pref = hawk_is_pref[slot][v];

        // Split training for demand vs prefetch PCs
        if (sig) {
            uint32_t idx = shct_idx(sig);
            if (was_pref) {
                if (used) shct_inc(shct_prefetch[idx]);
                else      shct_dec(shct_prefetch[idx]);
            } else {
                if (used) shct_inc(shct_demand[idx]);
                else      shct_dec(shct_demand[idx]);
            }
        }
        // Clear old metadata
        hawk_sig[slot][v] = 0;
        hawk_used[slot][v] = 0;
        hawk_is_pref[slot][v] = 0;
    }

    return v;
}

// ---------------- Update replacement state ------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Ignore writebacks entirely
    if (is_writeback(type)) return;

    // Update global counters and TinyLFU decay
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
    }

    // Bandit: collect leader outcomes (hits - misses)
    if (LEADER_A(set)) {
        leaderA_score += hit ? 1 : -1;
    } else if (LEADER_B(set)) {
        leaderB_score += hit ? 1 : -1;
    } else {
        // Followers: gently bias per-set bandit toward the current global gate
        if (GSEL >= GSEL_THRES) sat_inc_i8(bandit_score[set]);
        else                    sat_dec_i8(bandit_score[set]);
    }

    // Periodic selector gate update
    if ((access_count % BANDIT_EPOCH) == 0) {
        if (leaderB_score > leaderA_score) sat_inc_u5(GSEL, GSEL_MAX);
        else                               sat_dec_u5(GSEL);
        leaderA_score = leaderB_score = 0;
    }

    // Demand accounting for PC state
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);              // TinyLFU tick
        if (hit) sat_dec_u2(pc_cold2[pidx]);    // hot on hit
        else     sat_inc_u2(pc_cold2[pidx]);    // colder on miss
    }

    // Decide which mode this set uses
    bool useB = modeB_enabled(set);

    // STREAM detection (used by Mode B decisions and for prefetch quarantine)
    bool stream_now = detect_and_update_stream(PC, paddr);

    // On hit: promotion policy
    if (hit) {
        if (useB) {
            // Multi-hit gating + scan resistance
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            }
            if (stream_tag[set][way]) {
                if (STREAM_DEMOTE_TOUCH) rrpv_demote(set, way); // quarantine stays cold
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    rrpv_set(set, way, 0);
                    stream_tag[set][way] = 0; // escape quarantine after proven reuse
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                } else {
                    // gentle promotion for early reuse evidence
                    rrpv_promote(set, way);
                }
            }
        } else {
            // Mode A: classic Hawkeye-style MRU on hit
            rrpv_set(set, way, 0);
        }

        // Mark reuse for leader training
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }
        return;
    }

    // Miss path: insertion policy (called after fill)
    // Reset per-line counters
    hitcnt[set][way] = 0;

    if (useB) {
        // Mode B: RIB-LFU insertion
        uint32_t pidx = pc_index(PC);
        bool hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        bool cold = (pc_cold2[pidx] >= PC_COLD_STRICT_TH);

        if (is_prefetch(type)) {
            // Always quarantine prefetches
            rrpv_set(set, way, STREAM_TAIL_DEPTH);
            stream_tag[set][way] = stream_now ? 1u : 0u;
        } else {
            if (stream_now) {
                // Confident stream: hard tail (shallower if PC is very hot)
                rrpv_set(set, way, hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH);
                stream_tag[set][way] = 1;
            } else {
                // Non-stream: adaptive insertion with cold gate
                uint8_t depth = cold ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
                rrpv_set(set, way, depth);
                stream_tag[set][way] = 0;
            }
        }
    } else {
        // Mode A: Hawkeye-like SHCT insertion
        uint16_t sig  = pc_sig12(PC);
        uint32_t idx  = shct_idx(sig);
        uint8_t  depth;

        if (is_prefetch(type)) {
            depth = STREAM_TAIL_DEPTH; // prefetch cold
        } else {
            // friendly if counter high
            depth = (shct_demand[idx] >= (SHCT_MAX / 2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
        rrpv_set(set, way, depth);
        stream_tag[set][way] = 0;

        // Record per-line training buffers for sampled sets
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}