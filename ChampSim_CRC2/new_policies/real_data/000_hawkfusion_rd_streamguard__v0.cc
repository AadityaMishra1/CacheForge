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

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables ---------------------------------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
// Mode A (Hawkeye-like via SHCT)
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t SHCT_HOT_THRESH     = 16; // 5b counter threshold (0..31)
// Mode B (RD-StreamGuard)
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t RD_DEAD_THRESH      = 14; // per-PC RD sketch (0..255)
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote stream-tagged PCs on touch
// Global selector
static constexpr uint8_t GSEL_MAX            = 31;
static constexpr uint8_t GSEL_ENABLE_THRESH  = 2;  // followers use Mode B if >= threshold

// ---------------- Per-line metadata (minimal) ---------------------
// Only RRIP is per-line globally; leader-only lines carry tiny bits (below).
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// ---------------- Leader-only per-line metadata ------------------
// Used for bandit credit and local training without global overhead.
static uint8_t leader_ins_mode[LLC_SETS][LLC_WAYS];   // 0=A, 1=B (leaders only)
static uint8_t leader_seen_once[LLC_SETS][LLC_WAYS];  // first demand hit seen (leaders only)
static uint8_t leader_two_hit[LLC_SETS][LLC_WAYS];    // 2+ hits (leaders only)
static uint16_t leader_pc_sig_shct[LLC_SETS][LLC_WAYS]; // 11-bit PC sig for SHCT training (leaders only)

// ---------------- Mode A (Hawkeye-like via SHCT) -----------------
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 2K x 5b
static uint8_t shct_prefetch[SHCT_SIZE]; // 2K x 5b
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// ---------------- Mode B (RD-StreamGuard) ------------------------
// 512-entry PC table: last line (10b), stream conf (2b), RD score (8b), prior_hit (1b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b effective
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_rd8[PC_TBL_SIZE];         // 0..255 (reuse-distance sketch)
static uint8_t  pc_prior_hit[PC_TBL_SIZE];   // 0/1

// Tiny Evicted Address Filter (direct-mapped)
static constexpr uint32_t EAF_SIZE = 256;
static uint16_t eaf_tag[EAF_SIZE]; // 12-bit tags stored in 16b
static inline uint32_t eaf_idx(uint64_t line_addr) { return (uint32_t)(line_addr ^ (line_addr >> 10)) & (EAF_SIZE - 1); }
static inline uint16_t eaf_make_tag(uint64_t line_addr) { return (uint16_t)(line_addr & 0x0FFFu); } // 12b

// ---------------- Global selector (set dueling) -------------------
static uint8_t GSEL = 0; // saturating 0..31; followers use Mode B if >= GSEL_ENABLE_THRESH

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u8(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t& x) { if (x > 0) x--; }
static inline void gsel_inc() { if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec() { if (GSEL > 0) GSEL--; }

static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_ENABLE_THRESH);
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

static inline bool eaf_predict_dead(uint64_t paddr) {
    uint64_t line_addr = paddr >> 6;
    uint32_t i = eaf_idx(line_addr);
    uint16_t t = eaf_make_tag(line_addr);
    return (eaf_tag[i] == t);
}

static inline void eaf_insert(uint64_t victim_addr) {
    uint64_t line_addr = victim_addr >> 6;
    uint32_t i = eaf_idx(line_addr);
    eaf_tag[i] = eaf_make_tag(line_addr);
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            if (SEL_SAMPLED(s)) {
                leader_ins_mode[s][w] = 0;
                leader_seen_once[s][w] = 0;
                leader_two_hit[s][w] = 0;
                leader_pc_sig_shct[s][w] = 0;
            }
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_rd8, 0, sizeof(pc_rd8));
    std::memset(pc_prior_hit, 0, sizeof(pc_prior_hit));
    std::memset(eaf_tag, 0xFF, sizeof(eaf_tag)); // invalid tag
    GSEL = 0;
}

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging to avoid infinite loops
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
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update state -----------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Ignore writebacks for policy decisions
    if (type == ACCESS_WRITEBACK) return;

    uint32_t pcidx = pc_index(PC);
    bool demand = is_demand(type);

    // Leader accounting for hits (for bandit reward)
    if (hit && SEL_SAMPLED(set)) {
        if (demand) {
            if (leader_seen_once[set][way] == 0) leader_seen_once[set][way] = 1;
            else leader_two_hit[set][way] = 1;
        }
    }

    // Demand hit handling: multi-hit gating + stream demote-on-touch
    if (hit && demand) {
        // Stream demote on touch (PC-level stream tag)
        if (STREAM_DEMOTE_TOUCH && pc_stream_conf[pcidx] >= STREAM_CONF_THRESH) {
            if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }
        // Multi-hit gating: promote on 2nd demand hit only (per-PC prior_hit)
        if (pc_prior_hit[pcidx]) {
            // If PC currently flagged as stream, still allow promotion only at 2nd+ hit
            rrpv[set][way] = 0; // MRU
        } else {
            pc_prior_hit[pcidx] = 1;
        }
        return;
    }

    // On fills (miss or prefetch)
    if (!hit) {
        // Bandit credit: attribute evicted line outcome for leaders before overwriting metadata
        if (SEL_SAMPLED(set)) {
            uint8_t old_mode = leader_ins_mode[set][way]; // 0=A, 1=B
            bool two_plus_hits = (leader_two_hit[set][way] != 0);
            if (LEADER_B(set)) {
                if (two_plus_hits) gsel_inc(); else gsel_dec();
            } else if (LEADER_A(set)) {
                if (two_plus_hits) gsel_dec(); else gsel_inc();
            }
            // Train per-PC RD sketch from the evicted line's insertion PC (leaders only)
            uint32_t old_shct_idx = leader_pc_sig_shct[set][way];
            if (two_plus_hits) sat_dec_u8(pc_rd8[old_shct_idx]); else sat_inc_u8(pc_rd8[old_shct_idx], 255);
            // Reset leader outcome bits for the new line
            leader_seen_once[set][way] = 0;
            leader_two_hit[set][way] = 0;
        }

        // Insert victim into EAF (never for writebacks; we already returned above for WB)
        eaf_insert(victim_addr);

        // Decide ensemble mode
        bool modeB = use_modeB(set);

        // Compute insertion depth
        uint8_t ins_rrpv = INSERT_COLD_DEPTH; // default conservative

        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // prefetch quarantine at tail
        } else if (modeB) {
            bool is_stream = detect_and_update_stream(PC, paddr);
            bool pred_dead = (pc_rd8[pcidx] >= RD_DEAD_THRESH) || eaf_predict_dead(paddr);
            if (is_stream || pred_dead) {
                ins_rrpv = maxRRPV; // hard-tail quarantine
            } else {
                ins_rrpv = INSERT_WARM_DEPTH; // warm insert for likely reusers
            }
        } else {
            // Mode A: Hawkeye-like (SHCT-guided) insertion
            uint32_t idx = shct_idx(PC);
            bool friendly = (shct_demand[idx] >= SHCT_HOT_THRESH);
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }

        // Apply insertion
        rrpv[set][way] = std::min<uint8_t>(ins_rrpv, maxRRPV);

        // Leader metadata for the new line
        if (SEL_SAMPLED(set)) {
            leader_ins_mode[set][way] = modeB ? 1 : 0;
            leader_pc_sig_shct[set][way] = (uint16_t)shct_idx(PC);
        }

        // SHCT training (lightweight Hawkeye-like reuser detector)
        // Positive on demand fill, negative on dead evict (leaders decide credit; followers rely on stream/RD).
        if (demand) {
            shct_inc(shct_demand[shct_idx(PC)]);
        } else if (type == ACCESS_PREFETCH) {
            shct_inc(shct_prefetch[shct_idx(PC)]);
        }

        // Reset per-PC prior-hit on a miss to avoid false promotion bursts
        pc_prior_hit[pcidx] = 0;

        return;
    }

    // Prefetch hit: do not promote on first touch (gated by pc_prior_hit)
    if (hit && type == ACCESS_PREFETCH) {
        // do nothing (kept quarantined)
        return;
    }
}

// ---------------- Stats (quiet) ----------------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}