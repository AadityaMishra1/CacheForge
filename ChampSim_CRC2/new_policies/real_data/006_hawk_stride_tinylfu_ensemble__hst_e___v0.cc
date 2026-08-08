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

// ---------------- Tunables (retuned for lbm/zeusmp balance) -------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 8;  // TinyLFU hot threshold (0..15)

static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence to enable B
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Mode A (Hawkeye-like)

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode A unless confident
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_score = 0;      // hits - misses on leader A sets
static int32_t leaderB_score = 0;      // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (trained only in leaders)
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b xor-folded hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (Stride-Sieve + TinyLFU) -----------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (ln == exp1) || (ln == exp2);
    }
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
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
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    for (uint32_t s = 0; s < 64; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[s][w] = 0;
            hawk_used[s][w] = 0;
            hawk_is_pref[s][w] = 0;
        }
    }
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV; bounded aging to guarantee termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // 3) Fallback: choose the way with highest RRPV
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
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update replacement on hit/fill ------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Never act on writebacks beyond maintaining RRPV state
    if (is_writeback(type)) return;

    // Selector accounting (leader scores)
    bool useB = modeB_enabled(set);
    if (LEADER_A(set)) useB = false;
    if (LEADER_B(set)) useB = true;

    // Reward: +1 on hit, -1 on miss (demand or prefetch)
    if (SEL_SAMPLED(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    // Global epoch and per-set bandit drift toward global decision
    access_count++;
    if (access_count % BANDIT_EPOCH == 0) {
        // Global gate with bias toward Mode A
        prefer_B = (leaderB_score > (leaderA_score + GATE_MARGIN));
        // Drift followers slowly toward/away from Mode B
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!SEL_SAMPLED(s)) {
                if (prefer_B) sat_inc_i8(bandit_score[s]);
                else sat_dec_i8(bandit_score[s]);
            }
        }
        leaderA_score = 0;
        leaderB_score = 0;
    }

    // TinyLFU: count demand references per PC
    if (is_demand(type)) sat_inc_u4(pc_use4[pc_index(PC)]);

    // Stream detection: only demand accesses influence the run detector
    bool stream_armed = false;
    if (is_demand(type)) stream_armed = detect_and_update_stream(PC, paddr);

    // Multi-hit gating on hits
    if (hit) {
        // Demote quarantined streams on any touch to keep them evictable
        if (stream_tag[set][way] && (rrpv[set][way] < maxRRPV)) {
            uint8_t newv = (uint8_t)std::min<int>(maxRRPV, rrpv[set][way] + STREAM_DEMOTE_TOUCH);
            rrpv_set(set, way, newv);
        }

        // Update leader-use bit for Mode A training
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Demand-hit promotion with multi-hit gating
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            bool is_stream_line = (stream_tag[set][way] != 0);
            uint8_t need_hits = is_stream_line ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;

            if (hitcnt[set][way] >= need_hits) {
                // Escape quarantine if any and promote
                stream_tag[set][way] = 0;
                rrpv_set(set, way, 0);
            } else {
                // Light promotion toward MRU
                if (rrpv[set][way] > 0) rrpv_set(set, way, (uint8_t)(rrpv[set][way] - 1));
            }
        }

        // Mode A counter reinforcement on hits (leaders only)
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            uint16_t sig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
            else                         shct_inc(shct_demand[idx]);
        }

        return;
    }

    // Miss + fill path
    // Train Mode A predictor on eviction at this way (leaders only)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // If previous line didn't see reuse, penalize its PC
        uint16_t old_sig = hawk_sig[slot][way];
        uint32_t idx     = shct_idx(old_sig);
        if (hawk_used[slot][way] == 0) {
            if (hawk_is_pref[slot][way]) shct_dec(shct_prefetch[idx]);
            else                         shct_dec(shct_demand[idx]);
        } else {
            if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[idx]);
            else                         shct_inc(shct_demand[idx]);
        }
        // Record new fill signature and reset reuse mark
        hawk_sig[slot][way]      = pc_sig12(PC);
        hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]     = 0;
    }

    // Decide which mode governs insertion
    if (useB) {
        // Mode B: Stride-Sieve + TinyLFU admission + quarantine
        uint32_t pidx = pc_index(PC);
        bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        bool quarantine = is_prefetch(type) || stream_armed;

        if (quarantine) {
            rrpv_set(set, way, STREAM_TAIL_DEPTH);
            stream_tag[set][way] = 1;
        } else {
            rrpv_set(set, way, hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
            stream_tag[set][way] = 0;
        }
        hitcnt[set][way] = 0;
    } else {
        // Mode A: Hawkeye-like insertion from tiny SHCT
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t pred = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];

        if (is_prefetch(type)) {
            // Quarantine prefetches at tail to avoid pollution
            rrpv_set(set, way, STREAM_TAIL_DEPTH);
            stream_tag[set][way] = 1;
        } else {
            // Friendly PCs warm, averse near-tail
            rrpv_set(set, way, (pred >= (SHCT_MAX / 2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
            stream_tag[set][way] = 0;
        }
        hitcnt[set][way] = 0;
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}