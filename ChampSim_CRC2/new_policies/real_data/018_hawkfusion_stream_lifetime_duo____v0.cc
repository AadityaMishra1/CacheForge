#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types aligned to ChampSim CRC2
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
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // Stream-Lifetime
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (streaming + irregular balance) ---------------
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU for friendly/hot
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail for cold
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for confident stream
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;   // shallower tail for hot streams (short reuse)
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // +1/+2 forward steps needed
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH     = 8;   // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024;// fast decay for phasey apps
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;   // per-set bandit threshold

// ---------------- Per-line metadata (conceptual packing) -----------------
// rrpv:3b, hitcnt:2b, stream_tag:1b (quarantine/stream)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set selector (bandit) ------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers enable Mode B when >= MODEB_ENABLE_THRESH

// ---------------- Global selector (dueling via leaders) ------------------
static int32_t leaderA_good = 0; // hits - misses on A leaders
static int32_t leaderB_good = 0; // hits - misses on B leaders
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHiP) --------------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12-bit PC signature hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A training state (only for 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)
static uint8_t  leader_valid[64][LLC_WAYS]; // line valid (0/1) for eviction training

// ---------------- Mode B (Stream-Lifetime guard) -------------------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

// Per-PC state (information-theoretic: 10b, 2b, 4b)
static uint16_t pc_last_line10[PC_TBL_SIZE]; // last line (10b in 16b storage)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 4b (0..15)

// Tiny region heatmap (demote scans)
static uint8_t region_heat2[256]; // 2b counters (0..3) stored in 8b

// ---------------- Helpers ------------------------------------------------
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
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// +1/+2 forward-run detector with 2-step confidence; reset on non-forward
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
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
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    // Followers: enable Mode B only if leaders show B >= A and local bandit_score is confident
    bool b_wins_globally = (leaderB_good >= leaderA_good);
    return b_wins_globally && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Initialization ----------------------------------------
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
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(leader_valid, 0, sizeof(leader_valid));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(region_heat2, 0, sizeof(region_heat2));
    leaderA_good = leaderB_good = 0;
    access_count = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_pick_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes (ensure termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: evict the way with largest RRPV
    uint32_t victim = 0, best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    return rrip_pick_victim(set, current_set);
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    access_count++;

    // Decay TinyLFU periodically to track phases
    if ((access_count & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1;
        }
        // mild cool-down of regions
        for (uint32_t r = 0; r < 256; r++) {
            if (region_heat2[r] > 0) region_heat2[r]--;
        }
    }

    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    bool useB    = modeB_enabled(set);

    // Selector accounting (dueling): count hits-misses per leader type
    if (leaderA) {
        leaderA_good += (hit ? 1 : -1);
    } else if (leaderB) {
        leaderB_good += (hit ? 1 : -1);
    } else {
        // Follower: adjust per-set bandit if using Mode B
        if (useB) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else     sat_dec_i8(bandit_score[set]);
        } else {
            // decay toward neutral slowly
            if (bandit_score[set] > 0) bandit_score[set]--;
            else if (bandit_score[set] < 0) bandit_score[set]++;
        }
    }

    // Ignore writebacks for policy decisions
    if (is_writeback(type)) return;

    // Region id for heatmap
    uint32_t region_id = (uint32_t)((paddr >> 18) & 0xFFu);

    // Demand PC TinyLFU update
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }

    if (hit) {
        // HIT path: multi-hit promotion and quarantine handling
        if (stream_tag[set][way]) {
            // Quarantined/stream line: demote-on-touch until confirmed reuse
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                bool hot_pc = (pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH);
                if (hitcnt[set][way] >= HITS_PROMOTE_STR || hot_pc) {
                    rrpv_set(set, way, 0); // escape quarantine to MRU
                    stream_tag[set][way] = 0;
                } else {
                    rrpv_demote(set, way); // keep cold
                }
            } else {
                // prefetch touch: keep quarantined
                rrpv_demote(set, way);
            }
        } else {
            // Non-stream lines: promote only on 2nd+ demand hit
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                }
            }
        }

        // Leader-A reuse mark for training
        if (leaderA) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Cool down region slightly on reuse
        if (region_heat2[region_id] > 0) region_heat2[region_id]--;
        return;
    }

    // MISS/FILL path: choose insertion by mode and context
    uint8_t insert_depth = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;

    // Stream detection (update on all demand/prefetch)
    bool stream_seen = detect_and_update_stream(PC, paddr);
    bool region_hot  = (region_heat2[region_id] >= 3);
    bool stream_like = stream_seen || region_hot;

    // Mode A (Hawkeye-like via SHCT) decision
    auto hawk_decide_insert = [&](bool is_pref) -> uint8_t {
        uint16_t sig  = pc_sig12(PC);
        uint32_t idx  = shct_idx(sig);
        uint8_t  pred = is_pref ? shct_prefetch[idx] : shct_demand[idx];
        // Friendly if strong; prefetches always insert cold
        if (is_pref) return maxRRPV;
        return (pred > (SHCT_MAX / 2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    };

    // Mode B (Stream-Lifetime) decision
    auto guard_decide_insert = [&](bool is_pref) -> std::pair<uint8_t,uint8_t> {
        if (is_pref) {
            return {maxRRPV, 1u}; // quarantine prefetches
        }
        uint32_t pidx = pc_index(PC);
        bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        if (stream_like) {
            return { hot_pc ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH, 1u };
        } else {
            return { hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH, 0u };
        }
    };

    if (useB) {
        auto pr = guard_decide_insert(is_prefetch(type));
        insert_depth   = pr.first;
        new_stream_tag = pr.second;
    } else {
        insert_depth   = hawk_decide_insert(is_prefetch(type));
        new_stream_tag = is_prefetch(type) ? 1u : (stream_like ? 1u : 0u); // conservative quarantine if stream-like
    }

    // Perform insertion
    rrpv_set(set, way, insert_depth);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = new_stream_tag;

    // Region heat-up on miss/fill
    sat_inc_u2(region_heat2[region_id]);

    // Leader-A training on eviction and install
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);

        // Train old line if valid
        if (leader_valid[slot][way]) {
            uint16_t osig = hawk_sig[slot][way];
            uint32_t oidx = shct_idx(osig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[oidx]);
                else                      shct_dec(shct_prefetch[oidx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[oidx]);
                else                      shct_dec(shct_demand[oidx]);
            }
        }

        // Install new training record
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]    = 0;
        leader_valid[slot][way] = 1;
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}