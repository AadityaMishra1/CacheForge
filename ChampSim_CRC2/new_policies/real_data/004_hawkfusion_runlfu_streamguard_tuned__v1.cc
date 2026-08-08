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

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Leader-set sampling (64 leaders; 32 A + 32 B) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables (tuned for lbm/mcf/gcc/omnetpp/astar balance) ----
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP (0..7)
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU insert
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail insert
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps to tag as stream
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream promote at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined stream on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU hot threshold (0..15), lowered
static constexpr int8_t  MODEB_ENABLE_THRESH = 3;  // followers enable B only if >=3

// ---------------- Per-line packed metadata (conceptual bit-packing) ----------
// rrpv:3b, hit_once:1b, stream_tag:1b, from_modeB:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hit_once[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];
static uint8_t from_modeB[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector -----------------------------------
static int8_t bandit_score[LLC_SETS]; // followers use Mode B only if >= MODEB_ENABLE_THRESH

// ---------------- Mode A (Hawkeye-like) -------------------------------------
// Tiny SHCTs (demand & prefetch), trained on leader-A sets only
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature (lower 12 bits of PC) + prefetched flag per line for training (leader-A sets only)
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // 12b effective in 16b storage
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Mode B (RunGuard + TinyLFU) --------------------------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)

// Simple decay for TinyLFU
static uint32_t decay_ptr = 0;
static uint32_t decay_epoch = 0;

// ---------------- Helpers ----------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x, int8_t d=1)  { int16_t t = (int16_t)x + d; if (t > 7) t = 7; x = (int8_t)t; }
static inline void sat_dec_i8(int8_t& x, int8_t d=1)  { int16_t t = (int16_t)x - d; if (t < -8) t = -8; x = (int8_t)t; }

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Initialization ---------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hit_once[s][w] = 0;
            stream_tag[s][w] = 0;
            from_modeB[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    decay_ptr = 0;
    decay_epoch = 0;
}

// ---------------- Victim selection (RRIP) ------------------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) If any at maxRRPV, evict it
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes (guarantee termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // 4) Fallback: return first maxRRPV (should exist)
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    return 0; // safety
}

// ---------------- ChampSim hooks ---------------------------------------------
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    return rrip_victim_and_age(set, current_set);
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
    bool demand = is_demand(type);
    bool prefetch = (type == ACCESS_PREFETCH);

    // Update TinyLFU and stream detector on demand references
    if (demand) {
        uint32_t pi = pc_index(PC);
        sat_inc_u4(pc_use4[pi]);
        (void)detect_and_update_stream(PC, paddr);
    }

    if (hit) {
        // stream demote-on-touch (keeps scans cold)
        if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) {
            if (rrpv[set][way] < maxRRPV) rrpv[set][way] += STREAM_DEMOTE_TOUCH;
            if (rrpv[set][way] > maxRRPV) rrpv[set][way] = maxRRPV;
        }
        // multi-hit promotion gate (demand only)
        if (demand) {
            if (hit_once[set][way] == 0) {
                hit_once[set][way] = 1; // first demand hit; no promotion
            } else {
                // second+ demand hit: promote to MRU and allow streams to escape
                rrpv[set][way] = 0;
                stream_tag[set][way] = 0;
            }
        }
        return;
    }

    // Miss/Fill path: train and insert

    // 1) Bandit training using victim's outcome
    {
        uint8_t old_hits = hit_once[set][way];
        uint8_t was_modeB = from_modeB[set][way];

        // Reward reused lines mildly; penalize dead lines harshly
        if (old_hits) {
            if (was_modeB) sat_inc_i8(bandit_score[set], 1);
            else            sat_dec_i8(bandit_score[set], 1);
        } else {
            if (was_modeB) sat_dec_i8(bandit_score[set], 3);
            else            sat_inc_i8(bandit_score[set], 3);
        }

        // Mode A SHCT training (leader-A sets only)
        if (LEADER_A(set)) {
            uint16_t sig = hawk_signatures[set][way] & 0x0FFFu;
            uint32_t idx = sig & (SHCT_SIZE - 1);
            if (hawk_prefetched[set][way]) {
                if (old_hits) shct_inc(shct_prefetch[idx]);
                else          shct_dec(shct_prefetch[idx]);
            } else {
                if (old_hits) shct_inc(shct_demand[idx]);
                else          shct_dec(shct_demand[idx]);
            }
        }
    }

    // 2) Choose mode for insertion
    bool use_modeB = modeB_enabled(set);

    // 3) Decide insertion RRPV and tags
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t ins_stream = 0;

    if (prefetch) {
        // Prefetch quarantine at tail
        ins_rrpv = maxRRPV;
        ins_stream = 0;
    } else if (use_modeB) {
        // Mode B: stream guard + TinyLFU
        bool is_stream = detect_and_update_stream(PC, paddr); // demand path already updated, harmless if repeated
        if (is_stream) {
            ins_rrpv = maxRRPV;  // hard-tail
            ins_stream = 1;
        } else {
            uint32_t pi = pc_index(PC);
            if (pc_use4[pi] >= PC_USE_HOT_THRESH) ins_rrpv = INSERT_WARM_DEPTH;
            else                                  ins_rrpv = INSERT_COLD_DEPTH;
            ins_stream = 0;
        }
    } else {
        // Mode A: Hawkeye-like SHCT-based insertion
        uint32_t idx = shct_idx(PC);
        uint8_t score = demand ? shct_demand[idx] : shct_prefetch[idx];
        if (demand) {
            ins_rrpv = (score >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        } else {
            ins_rrpv = maxRRPV; // prefetch tail
        }
        ins_stream = 0;
    }

    // 4) Install new line state
    rrpv[set][way]        = (ins_rrpv > maxRRPV) ? maxRRPV : ins_rrpv;
    hit_once[set][way]    = 0;
    stream_tag[set][way]  = ins_stream ? 1 : 0;
    from_modeB[set][way]  = use_modeB ? 1 : 0;

    // Store signature/prefetch bit for Mode A training in leader-A sets
    if (LEADER_A(set)) {
        hawk_signatures[set][way] = (uint16_t)(PC & 0x0FFFu);
        hawk_prefetched[set][way] = prefetch ? 1 : 0;
    }

    // 5) Light decay of TinyLFU to avoid stale hotness
    if ((++decay_epoch & 63u) == 0u) {
        sat_dec_u4(pc_use4[decay_ptr & (PC_TBL_SIZE - 1)]);
        decay_ptr++;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}