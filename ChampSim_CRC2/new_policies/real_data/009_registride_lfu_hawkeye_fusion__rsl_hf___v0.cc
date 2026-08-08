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

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (balanced for lbm/mcf/gcc/omnetpp) -----
static constexpr uint8_t  maxRRPV              = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH    = 2;  // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH    = 6;  // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH    = 7;  // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_ARM_THRESH    = 2;  // arm after 2 fwd steps
static constexpr uint8_t  STREAM_LOCK_THRESH   = 3;  // lock after 3
static constexpr uint8_t  HITS_PROMOTE_NS      = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR     = 2;  // stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT = 1;  // hot-PC stream: MRU on 1st demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH  = 1;  // demote quarantined on touch
static constexpr uint8_t  PC_USE_HOT_THRESH    = 8;  // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD     = 2048; // decay every N accesses

// Selector epoch and gates (bias to Hawkeye)
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward A

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b (quarantine tag)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], followers need >= MODEB_ENABLE_THRESH to use B
static bool    prefer_B = false;       // global gate from leaders
static int32_t leaderA_reward = 0;     // hits - misses on A leaders
static int32_t leaderB_reward = 0;     // hits - misses on B leaders
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-lite via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (RegiStride-LFU PC state) ----------------
// 512-entry PC tables: region (12b hash), last offset (6b), last global line10 (10b),
// stream conf (2b), TinyLFU (4b), last delta bucket (3b), diversity (2b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // low 10b of line #
static inline uint16_t region12(uint64_t paddr) {
    // 4KB region: paddr>>12; xor-fold to 12b
    uint64_t x = (paddr >> 12);
    x ^= (x >> 7) ^ (x >> 13);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint8_t region_off6(uint64_t paddr) { return (uint8_t)((paddr >> 6) & 0x3Fu); }

static uint16_t pc_last_region12[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_last_off6[PC_TBL_SIZE];     // 0xFF=uninit
static uint16_t pc_last_line10[PC_TBL_SIZE];   // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE];   // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];          // 0..15 (TinyLFU)
static uint8_t  pc_last_delta3[PC_TBL_SIZE];   // 0..7 bucketed delta
static uint8_t  pc_diversity2[PC_TBL_SIZE];    // 0..3: higher => more random

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}

static inline uint8_t delta_bucket(int32_t d) {
    // Bucket -2,-1,0,+1,+2 as 0..4, others as 5,6,7 depending on sign/magnitude
    if (d <= -2 && d >= -4) return 0;
    if (d == -1) return 1;
    if (d == 0)  return 2;
    if (d == 1)  return 3;
    if (d >= 2 && d <= 4) return 4;
    if (d < -4)  return 5;
    if (d > 4)   return 6;
    return 7;
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t reg = region12(paddr);
    uint8_t  off = region_off6(paddr);
    uint16_t ln  = line10(paddr);

    bool fwd_local = false, fwd_global = false;

    if (pc_last_region12[idx] != 0xFFFFu && pc_last_off6[idx] != 0xFFu) {
        if (pc_last_region12[idx] == reg) {
            int32_t d = (int32_t)off - (int32_t)pc_last_off6[idx];
            fwd_local = (d == 1) || (d == 2);
            // update diversity on local stride
            uint8_t cur_bucket = delta_bucket(d);
            if (pc_last_off6[idx] != 0xFFu) {
                if (pc_last_delta3[idx] != cur_bucket) sat_inc_u2(pc_diversity2[idx]);
                else sat_dec_u2(pc_diversity2[idx]);
            }
            pc_last_delta3[idx] = cur_bucket;
        }
    }
    if (pc_last_line10[idx] != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(pc_last_line10[idx] + 1);
        uint16_t exp2 = (uint16_t)(pc_last_line10[idx] + 2);
        fwd_global = (ln == exp1) || (ln == exp2);
    }

    bool forward = fwd_local || fwd_global;
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;

    pc_last_region12[idx] = reg;
    pc_last_off6[idx]     = off;
    pc_last_line10[idx]   = ln;

    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

// ---------------- Initialization ----------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_region12[i] = 0xFFFFu;
        pc_last_off6[i]     = 0xFFu;
        pc_last_line10[i]   = 0xFFFFu;
        pc_stream_conf[i]   = 0;
        pc_use4[i]          = 0;
        pc_last_delta3[i]   = 0;
        pc_diversity2[i]    = 0;
    }

    prefer_B = false;
    leaderA_reward = 0;
    leaderB_reward = 0;
    access_count = 0;
}

// ---------------- Victim selection (RRIP) -------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP victim: search for maxRRPV; age if none
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                // Train Hawkeye-lite predictors on eviction for sampled sets
                if (SEL_SAMPLED(set)) {
                    uint32_t slot = LEADER_SLOT(set);
                    uint16_t sig = hawk_sig[slot][w] & 0x0FFFu;
                    bool was_pref = (hawk_is_pref[slot][w] != 0);
                    bool was_used = (hawk_used[slot][w] != 0);
                    if (was_pref) {
                        if (was_used) shct_inc(shct_prefetch[shct_idx(sig)]);
                        else          shct_dec(shct_prefetch[shct_idx(sig)]);
                    } else {
                        if (was_used) shct_inc(shct_demand[shct_idx(sig)]);
                        else          shct_dec(shct_demand[shct_idx(sig)]);
                    }
                    // clear used bit for the evicted slot (new fill will set)
                    hawk_used[slot][w] = 0;
                }
                return w;
            }
        }
        // Age all RRPVs (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // unreachable
    return 0;
}

// ---------------- State update (hits and fills) -------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Epoch management and TinyLFU decay
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
        }
    }
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Global decision: prefer Mode B only if it convincingly wins on leaders
        int32_t diff = (leaderB_reward - leaderA_reward);
        prefer_B = (diff > GATE_MARGIN);
        leaderA_reward = 0;
        leaderB_reward = 0;
        // Mild decay bandit scores toward neutral
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (bandit_score[s] > 0) bandit_score[s]--;
            else if (bandit_score[s] < 0) bandit_score[s]++;
        }
    }

    // Update PC tables (stream, diversity, LFU)
    uint32_t pc_idx = pc_index(PC);
    bool armed_stream = detect_and_update_stream(PC, paddr);
    if (is_demand(type)) sat_inc_u4(pc_use4[pc_idx]);

    // Mode selection
    bool use_modeB = modeB_enabled(set);

    // Leader rewards (hit: +1, miss: -1)
    if (LEADER_A(set)) {
        leaderA_reward += (hit ? 1 : -1);
    } else if (LEADER_B(set)) {
        leaderB_reward += (hit ? 1 : -1);
    }

    // On hit: multi-hit promotion logic
    if (hit) {
        // Mark reuse for Hawkeye-lite training in sampled sets
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Count hits (demand hits only gate promotion)
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            bool is_stream_line = (stream_tag[set][way] != 0);
            bool hot_pc = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);

            if (!is_stream_line) {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU
                }
            } else {
                // Stream quarantine: demote on touch unless promotion condition trips
                if ((hot_pc && hitcnt[set][way] >= HITS_PROMOTE_STR_HOT) ||
                    (hitcnt[set][way] >= HITS_PROMOTE_STR)) {
                    rrpv_set(set, way, 0); // escape quarantine
                    stream_tag[set][way] = 0;
                } else if (STREAM_DEMOTE_TOUCH) {
                    sat_inc_u3(rrpv[set][way]);
                }
            }
        }
        return;
    }

    // Miss/fill path: choose insertion policy per mode
    // Never bypass writebacks: insert them near-tail conservatively
    if (is_writeback(type)) {
        stream_tag[set][way] = 0;
        hitcnt[set][way] = 0;
        rrpv_set(set, way, INSERT_COLD_DEPTH);
        return;
    }

    // Compute PC-guided features for insertion
    bool is_stream_now = armed_stream || (pc_stream_conf[pc_idx] >= STREAM_LOCK_THRESH);
    bool is_pf = is_prefetch(type);
    bool hot_pc = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH);
    bool randomish = (pc_diversity2[pc_idx] >= 2);

    uint8_t ins_depth = INSERT_WARM_DEPTH;
    uint8_t depth_cold = INSERT_COLD_DEPTH;

    if (use_modeB) {
        // RegiStride-LFU
        if (is_pf || is_stream_now) {
            ins_depth = STREAM_TAIL_DEPTH;  // aggressive quarantine
            stream_tag[set][way] = 1;
        } else if (randomish || !hot_pc) {
            ins_depth = depth_cold;         // cold insert for noisy/cold PCs
            stream_tag[set][way] = 0;
        } else {
            ins_depth = INSERT_WARM_DEPTH;  // warm for hot PCs
            stream_tag[set][way] = 0;
        }
        // Update local confidence for followers based on observed outcome (miss -> neutral, hit updates handled above)
        if (!LEADER_A(set) && !LEADER_B(set)) {
            // On fill, nudge confidence toward chosen mode B ever so slightly
            sat_inc_i8(bandit_score[set]);
        }
    } else {
        // Mode A: Hawkeye-lite (tiny SHCT) insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        bool friendly = false;
        if (is_pf) friendly = (shct_prefetch[idx] > (SHCT_MAX / 2));
        else        friendly = (shct_demand[idx]  > (SHCT_MAX / 2));

        ins_depth = friendly ? INSERT_WARM_DEPTH : depth_cold;
        stream_tag[set][way] = is_pf ? 1 : 0; // prefetch quarantine at tail handled by ins_depth below
        if (is_pf && ins_depth < STREAM_TAIL_DEPTH) {
            // Prefetch still low priority: push a bit deeper
            ins_depth = std::max<uint8_t>(ins_depth, (uint8_t)INSERT_COLD_DEPTH);
        }
        // Followers: if we stuck with Mode A during a period where B is preferred, reduce confidence
        if (!LEADER_A(set) && !LEADER_B(set) && prefer_B) {
            sat_dec_i8(bandit_score[set]);
        }
    }

    // Apply insertion
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way] = 0;

    // Record Hawkeye-lite per-line info for sampled sets (training done on eviction)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_pf ? 1 : 0;
        hawk_used[slot][way]    = 0;
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}