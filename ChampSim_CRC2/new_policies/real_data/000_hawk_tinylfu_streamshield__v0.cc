#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include "../inc/champsim_crc2.h"

// ====================== CRC2 constants ======================
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (aligned to ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ====================== Leader-set sampling =================
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ====================== Tunables ============================
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near MRU for hot PCs
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // low priority non-stream cold
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps to call stream
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 3;  // stream/quarantined: MRU on 3rd demand hit
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // enable B when bandit_score >= 2
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined line on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15)

// ====================== Per-line metadata ===================
// Conceptual bit-pack (we store bytes, account minimal bits in storage section):
// rrpv:3b, hitcnt:2b, stream_lock:1b  => 6 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];// 0/1: quarantined (stream/prefetch)

// ====================== Per-set bandit ======================
static int8_t bandit_score[LLC_SETS]; // followers use B only if >= MODEB_ENABLE_THRESH

// ====================== Mode A: Hawkeye-like =================
// Tiny SHCT and sampled per-line signatures for training on sampled sets only.
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* demand_predictor = nullptr;
static HAWKEYE_PC_PREDICTOR* prefetch_predictor = nullptr;

// 64 sampled sets x 16 ways x 12-bit signatures + 1b prefetched flag
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // store lower 12 bits of PC; only valid for sampled sets
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1; used only on sampled sets

// ====================== Mode B: StreamShield + TinyLFU ======
// 512-entry PC-indexed tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 10-bit line idx

// Per-PC state: last line (10b), forward-run conf (2b), TinyLFU usefulness (4b)
static uint16_t pc_last_line10[PC_TBL_SIZE]; // store 0..1023
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15

// ====================== Helpers =============================
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

// ====================== Initialization =======================
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));

    demand_predictor = new HAWKEYE_PC_PREDICTOR();   // 2K x 5-bit
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR(); // 2K x 5-bit
}

// ====================== Victim selection (RRIP) ==============
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Pick any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Age bounded (<=8 passes) then pick the most aged
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: choose the way with largest RRPV
    uint32_t v = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; v = w; }
    }
    return v;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    uint32_t v = rrip_victim_and_age(set, current_set);

    // Bandit feedback: attribute only quarantined (Mode B hallmark) lines
    if (stream_lock[set][v]) {
        if (hitcnt[set][v] >= HITS_PROMOTE_NS) sat_inc_i8(bandit_score[set]);
        else                                   sat_dec_i8(bandit_score[set]);
    }

    return v;
}

// ====================== Mode B: StreamShield logic ===========
static inline bool pc_is_stream(uint32_t idx, uint16_t curr_line10) {
    uint16_t last = pc_last_line10[idx];
    uint16_t delta = (curr_line10 - last) & 0x03FFu; // modulo 1024
    if (delta == 1 || delta == 2) sat_inc_u2(pc_stream_conf[idx]);
    else                          sat_dec_u2(pc_stream_conf[idx]);
    pc_last_line10[idx] = curr_line10;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline uint8_t warm_depth_from_use(uint8_t use4) {
    return (use4 >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
}

// ====================== Mode A: Hawkeye-like logic ===========
static inline void hawk_train_on_hit(uint32_t set, uint32_t way, uint32_t type) {
    if (!SEL_SAMPLED(set)) return;
    uint64_t sig = hawk_signatures[set][way];
    if (!sig) return;
    if (type == ACCESS_PREFETCH) {
        prefetch_predictor->increment(sig);
    } else if (is_demand(type)) {
        if (hawk_prefetched[set][way]) prefetch_predictor->increment(sig);
        else                           demand_predictor->increment(sig);
    }
}

static inline void hawk_train_on_eviction(uint32_t set, uint32_t way) {
    if (!SEL_SAMPLED(set)) return;
    uint64_t sig = hawk_signatures[set][way];
    if (!sig) return;
    if (hawk_prefetched[set][way]) prefetch_predictor->decrement(sig);
    else                           demand_predictor->decrement(sig);
    hawk_signatures[set][way] = 0;
    hawk_prefetched[set][way] = 0;
}

static inline uint8_t hawk_insert_depth(uint64_t PC, uint32_t type) {
    if (type == ACCESS_PREFETCH) return maxRRPV; // tail
    bool friendly = demand_predictor->get_prediction(PC);
    return friendly ? INSERT_WARM_DEPTH : maxRRPV;
}

// ====================== Update state =========================
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Train Hawkeye on eviction of the victim (before overwrite)
    if (!hit && type != ACCESS_WRITEBACK) {
        hawk_train_on_eviction(set, way);
    }

    // Writebacks: do nothing else
    if (type == ACCESS_WRITEBACK) return;

    // Decide policy mode for this set
    bool forceA = LEADER_A(set);
    bool forceB = LEADER_B(set);
    bool use_mode_b = false;
    if (forceA) use_mode_b = false;
    else if (forceB) use_mode_b = true;
    else use_mode_b = (bandit_score[set] >= MODEB_ENABLE_THRESH);

    // Common: track hit counts and RRIP touches
    if (hit) {
        if (is_demand(type)) {
            // Demand hit: multi-hit promotion with stream-aware gating
            uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] + 1 >= need) {
                rrpv[set][way] = 0; // promote to MRU
            } else {
                // gentle aging: slight promotion for progress; optional demote for streams
                if (stream_lock[set][way] && STREAM_DEMOTE_TOUCH) {
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; // discourage scans
                } else {
                    if (rrpv[set][way] > 0) rrpv[set][way]--;
                }
            }
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            // Train Mode A on demand hits (sampled sets)
            hawk_train_on_hit(set, way, type);
            // TinyLFU usefulness update
            uint32_t pi = pc_index(PC);
            sat_inc_u4(pc_use4[pi]);
        } else if (type == ACCESS_PREFETCH) {
            // Prefetch hit: keep low priority; do not promote on first touch
            if (stream_lock[set][way] && STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            // Train Mode A on prefetch hits (sampled sets)
            hawk_train_on_hit(set, way, type);
        }
        return;
    }

    // Miss: we are inserting a new line at [set][way]
    hitcnt[set][way] = 0;

    if (use_mode_b) {
        // ---------------- Mode B: StreamShield + TinyLFU ----------------
        uint32_t pi = pc_index(PC);
        bool streamy = pc_is_stream(pi, line10(paddr));

        // Prefetches and stream-marked lines are quarantined at tail
        if (type == ACCESS_PREFETCH || streamy) {
            rrpv[set][way] = maxRRPV;
            stream_lock[set][way] = 1;
        } else {
            // Non-stream demand: TinyLFU warm insertion for hot PCs
            uint8_t depth = warm_depth_from_use(pc_use4[pi]);
            rrpv[set][way] = depth;
            stream_lock[set][way] = 0;
        }

        // Record Hawkeye signature for sampled sets (for A's training overlap)
        if (SEL_SAMPLED(set)) {
            hawk_signatures[set][way] = (uint16_t)(PC & 0x0FFFu);
            hawk_prefetched[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        }

    } else {
        // ---------------- Mode A: Hawkeye-like --------------------------
        // Signature only for sampled sets
        if (SEL_SAMPLED(set)) {
            hawk_signatures[set][way] = (uint16_t)(PC & 0x0FFFu);
            hawk_prefetched[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        }

        uint8_t depth = hawk_insert_depth(PC, type);
        rrpv[set][way] = depth;

        // Mode A never sets stream_lock; prefetch still acts quarantined via insertion depth
        stream_lock[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}