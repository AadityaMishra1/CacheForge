#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types per ChampSim CRC2
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

// ---------------- Tunables (retuned) ------------------------------
static constexpr uint8_t maxRRPV               = 7;  // 3-bit RRIP (0..7)
static constexpr uint8_t INSERT_WARM_DEPTH     = 3;  // near-MRU for likely-reuse
static constexpr uint8_t INSERT_COLD_DEPTH     = 6;  // near-tail for cold PCs
static constexpr uint8_t STREAM_TAIL_DEPTH     = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_ARM_THRESH     = 2;  // +1/+2 forward steps to arm
static constexpr uint8_t HITS_PROMOTE_NS       = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t PC_USE_HOT_THRESH     = 8;  // TinyLFU hot threshold (0..15)
static constexpr uint8_t PC_BYPASS_COLD_THRESH = 6;  // cold PCs (<6) enable strong stream tail
static constexpr uint32_t BANDIT_EPOCH         = 2048;
static constexpr int32_t  GATE_MARGIN          = 2;  // B must beat A by >=2

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b (0..3), stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1 if quarantined stream/prefetch

// ---------------- Selector (leader-driven global enable) ----------
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_score = 0;      // hits - misses (demand only)
static int32_t leaderB_score = 0;
static uint64_t access_count = 0;
static uint8_t consec_B = 0;           // hysteresis counters
static uint8_t consec_A = 0;

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
static uint8_t  hawk_valid[64][LLC_WAYS];   // valid slot (0/1)

// ---------------- Mode B (Stride-Sieve + TinyLFU) -----------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // only 10 bits used
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU usefulness)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    // +1/+2 forward-run detector with 2-bit confidence (0..3)
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];

    uint16_t exp1 = (uint16_t)(last + 1);
    uint16_t exp2 = (uint16_t)(last + 2);
    bool forward = (ln == exp1) || (ln == exp2);

    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;

    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_ARM_THRESH); // arms after 2 forward steps
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    return prefer_B;                   // followers
}

// ---------------- API: Initialize -------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));

    // SHCT: neutral-ish start
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 8;
        shct_prefetch[i] = 4;
    }

    // Hawkeye leader buffers
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(hawk_valid, 0, sizeof(hawk_valid));

    prefer_B = false;
    leaderA_score = leaderB_score = 0;
    access_count = 0;
    consec_A = consec_B = 0;
}

// ---------------- API: Victim selection --------------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP: look for maxRRPV; age until found
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_age_all(set);
    }
    // unreachable
    return 0;
}

// ---------------- API: Update state ------------------------------
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    // Update leader selector (demand only)
    if (is_demand(type)) {
        if (LEADER_A(set)) leaderA_score += (hit ? 1 : -1);
        if (LEADER_B(set)) leaderB_score += (hit ? 1 : -1);

        access_count++;
        if ((access_count % BANDIT_EPOCH) == 0) {
            if (leaderB_score >= leaderA_score + GATE_MARGIN) {
                consec_B++;
                consec_A = 0;
            } else if (leaderA_score >= leaderB_score + GATE_MARGIN) {
                consec_A++;
                consec_B = 0;
            } else {
                // tie/uncertain: decay hysteresis slightly towards A
                consec_A = std::min<uint8_t>(consec_A + 1, 2);
                consec_B = 0;
            }
            // Hysteresis: require two consecutive B wins to enable; one A epoch to disable
            if (!prefer_B && consec_B >= 2) prefer_B = true;
            if (prefer_B && consec_A >= 1)  prefer_B = false;

            leaderA_score = leaderB_score = 0;
        }
    }

    // Periodically decay TinyLFU to keep it responsive
    if ((access_count & 0x7FFu) == 0) { // every 2048 accesses
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i]--;
        }
    }

    // Stream detection (skip writebacks)
    bool armed_stream = false;
    if (!is_writeback(type)) {
        armed_stream = detect_and_update_stream(PC, paddr);
    }

    // On hits: multi-hit gated promotion; prefetch touches quarantined
    if (hit) {
        // Demand usefulness credit for TinyLFU on demand hits
        if (is_demand(type)) {
            sat_inc_u4(pc_use4[pc_index(PC)]);
        }

        // Leader training: mark reuse
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            if (hawk_valid[slot][way]) {
                hawk_used[slot][way] = 1;
            }
        }

        // Demote-on-touch for quarantined streams/prefetch
        if (stream_tag[set][way]) {
            // Demote slightly to keep in quarantine unless it escapes
            if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }

        // Multi-hit gate: only demand hits count
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            uint8_t need = stream_tag[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                // Escape quarantine and promote hard
                stream_tag[set][way] = 0;
                rrpv_set(set, way, 0); // MRU
            }
            // No promotion on first demand hit (keeps scan resistance)
        }

        return;
    }

    // Miss path: a fill will occur in (set, way). If leader, train the evicted line first.
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hawk_valid[slot][way]) {
            uint16_t osig = hawk_sig[slot][way];
            uint32_t idx = shct_idx(osig);
            if (hawk_is_pref[slot][way]) {
                // Prefetch stream was useful?
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                else                      shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                else                      shct_dec(shct_demand[idx]);
            }
        }
        // Overwrite with new fill context
        hawk_sig[slot][way]      = pc_sig12(PC);
        hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]     = 0;
        hawk_valid[slot][way]    = 1;
    }

    // Compute insertion policy
    uint8_t insert_rrpv = INSERT_COLD_DEPTH;
    uint8_t set_stream_tag = 0;
    hitcnt[set][way] = 0; // reset on new fill

    if (is_writeback(type)) {
        // Never bypass writebacks: moderate insertion
        insert_rrpv = 2;
        set_stream_tag = 0;
    } else if (modeB_enabled(set)) {
        uint32_t pidx = pc_index(PC);
        uint8_t use = pc_use4[pidx];

        if (is_prefetch(type)) {
            insert_rrpv = STREAM_TAIL_DEPTH; // quarantine
            set_stream_tag = 1;
        } else {
            bool cold_pc = (use < PC_BYPASS_COLD_THRESH);
            if (armed_stream && cold_pc) {
                // Strong stream bypass-equivalent for lbm: hard tail + quarantine
                insert_rrpv = STREAM_TAIL_DEPTH;
                set_stream_tag = 1;
            } else if (armed_stream && !cold_pc) {
                // Preserve short reuse in zeusmp
                insert_rrpv = INSERT_WARM_DEPTH;
                set_stream_tag = 1; // still quarantine until 2nd demand hit
            } else {
                // Non-stream: TinyLFU-weighted insertion
                insert_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                set_stream_tag = 0;
            }
        }
    } else {
        // Mode A (Hawkeye-like)
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);

        if (is_prefetch(type)) {
            insert_rrpv = STREAM_TAIL_DEPTH; // keep prefetches at tail
            set_stream_tag = 1;
        } else {
            bool hot = (shct_demand[idx] >= 16);
            insert_rrpv = hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            set_stream_tag = 0;
        }
    }

    rrpv_set(set, way, insert_rrpv);
    stream_tag[set][way] = set_stream_tag;
}

// ---------------- Stats (blank by requirement) -------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}