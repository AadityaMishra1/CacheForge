#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// -------------------- Tunables --------------------
static constexpr uint32_t SAMPLE_STRIDE = 32;         // 2048/32 = 64 sampled sets
// PC friendliness table (2-bit saturated), hashed by PC
static constexpr uint32_t FRIEND_BITS = 11;           // 2048 entries
static constexpr uint32_t FRIEND_SIZE = (1u << FRIEND_BITS);
// Per-PC ±1-line stream detector, hashed by PC
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
// Stream thresholds: conf >=2 to stream; if PC strongly friendly, require >=3
static constexpr uint8_t STREAM_CONF_NORM  = 2;
static constexpr uint8_t STREAM_CONF_HOT   = 3;
// Insertion depths by friendliness
static constexpr uint8_t INS_FRIENDLY = 1;            // RRPV=1
static constexpr uint8_t INS_NEUTRAL  = 2;            // RRPV=2
static constexpr uint8_t INS_COLD     = 3;            // RRPV=3
// Decay cadence for PC friendliness (phase stability)
static constexpr uint32_t DECAY_PERIOD_MISSES = 64;   // one entry decays every 64 demand misses

// -------------------- Per-line state --------------------
// SRRIP re-reference prediction value (logical 2-bit 0..3 stored in uint8_t)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];
// Per-line reuse state: 0 = no reuse yet, 1 = one demand hit, 2 = 2+ demand hits
static uint8_t LINE_STATE[LLC_SETS][LLC_WAYS];
// Per-line "no-train/no-promote" tag for stream/writeback lines
static uint8_t LINE_NT[LLC_SETS][LLC_WAYS]; // 0/1

// -------------------- Sampler state (sampled sets only) --------------------
// For sampled sets, keep PC-index and a "saw demand hit" bit to train friendliness on eviction
static uint16_t SAMPLER_PCIDX[LLC_SETS][LLC_WAYS]; // 11-bit stored in 16
static uint8_t  SAMPLER_HIT[LLC_SETS][LLC_WAYS];   // 0/1

// -------------------- Global predictors --------------------
// PC friendliness table: 2-bit saturating counters (0=cold .. 3=hot)
static uint8_t FRIEND[FRIEND_SIZE];

// Per-PC ±1-line stream detector
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit confidence 0..3
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // last step was |delta|==1

// Decay machinery
static uint32_t decay_ptr = 0;
static uint32_t demand_miss_ctr = 0;

// -------------------- Helpers --------------------
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x)               { if (x > 0) x--; }

static inline bool is_sampled_set(uint32_t set) { return (set % SAMPLE_STRIDE) == 0; }
static inline bool is_demand(uint32_t type) { return (type == ACCESS_LOAD) || (type == ACCESS_RFO); }

static inline uint32_t friend_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (FRIEND_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu);
}

// Update per-PC stream detector on demand access, return confidence after this access
static inline uint8_t update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(curr - prev);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc(ST_CONF[idx], 3); // consecutive ±1
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; // first indication
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        sat_dec(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

static inline uint8_t get_friend(uint64_t pc) {
    return FRIEND[friend_index(pc)]; // 0..3
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]       = 3;
            LINE_STATE[s][w] = 0;
            LINE_NT[s][w]    = 0;
            SAMPLER_PCIDX[s][w] = 0;
            SAMPLER_HIT[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < FRIEND_SIZE; i++) {
        FRIEND[i] = 1; // conservative neutral-cold start
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS1[i]  = 0;
    }
    decay_ptr = 0;
    demand_miss_ctr = 0;
}

// Find victim in the set (SRRIP with safe aging + eviction-time training)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: search for RRPV==3; if none, age and retry (bounded)
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) {
                // Train PC friendliness on sampled sets at eviction; skip stream/writeback-tagged lines
                if (is_sampled_set(set)) {
                    if (LINE_NT[set][w] == 0) {
                        uint32_t fidx = SAMPLER_PCIDX[set][w] & (FRIEND_SIZE - 1);
                        if (SAMPLER_HIT[set][w]) sat_inc(FRIEND[fidx], 3);
                        else                      sat_dec(FRIEND[fidx]);
                    }
                }
                return w;
            }
        }
        // Age all ways safely
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    // Fallback (should be unreachable)
    return 0;
}

// Update replacement state
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
    bool demand = is_demand(type);

    // Demand access: update stream detector and decide "hard stream"
    uint8_t conf = 0;
    uint8_t fval = 1;
    bool is_stream_now = false;
    if (demand) {
        conf = update_pc_stream(PC, paddr);
        fval = get_friend(PC);
        uint8_t thr = (fval == 3) ? STREAM_CONF_HOT : STREAM_CONF_NORM;
        is_stream_now = (conf >= thr);
    }

    if (hit) {
        // On hit: no promotion for prefetches or stream-tagged accesses
        if (type == ACCESS_PREFETCH) {
            // no-op
        } else {
            if (is_stream_now) {
                // Guard: mark as no-train/no-promote for streams
                LINE_NT[set][way] = 1;
            } else if (LINE_NT[set][way] == 0) {
                // Gentle multi-hit promotion on demand hits
                if (LINE_STATE[set][way] == 0) {
                    LINE_STATE[set][way] = 1;
                    if (RRPV[set][way] > 1) RRPV[set][way] = 1; // first hit -> RRPV=1
                } else {
                    LINE_STATE[set][way] = 2;
                    RRPV[set][way] = 0; // 2+ hits -> MRU
                }
                // Sampler: record a demand hit only for trainable lines
                if (is_sampled_set(set)) {
                    SAMPLER_HIT[set][way] = 1;
                }
            }
        }
        return;
    }

    // Miss: choose insertion policy
    if (type == ACCESS_WRITEBACK) {
        // Never bypass writebacks: insert at tail with no-train/no-promote
        RRPV[set][way]       = 3;
        LINE_STATE[set][way] = 0;
        LINE_NT[set][way]    = 1;
        // Do not sample writeback lines
        if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = 0;
            SAMPLER_HIT[set][way]   = 0;
        }
        return;
    }

    // Soft-bypass for streams (LOAD/RFO/PREFETCH)
    if (is_stream_now) {
        RRPV[set][way]       = 3;
        LINE_STATE[set][way] = 0;
        LINE_NT[set][way]    = 1; // exclude from training and promotion
        // Do not sample stream-tagged lines
        if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = 0;
            SAMPLER_HIT[set][way]   = 0;
        }
    } else {
        // Non-stream insertion based on friendliness
        uint8_t ins = INS_COLD;
        if (fval == 3)      ins = INS_FRIENDLY;
        else if (fval == 2) ins = INS_NEUTRAL;
        else                ins = INS_COLD;

        // Prefetches always insert at tail, regardless of friendliness
        if (type == ACCESS_PREFETCH) ins = 3;

        RRPV[set][way]       = ins;
        LINE_STATE[set][way] = 0;
        LINE_NT[set][way]    = 0;

        // Sample only demand inserts (avoid prefetch noise) and only for trainable lines
        if (demand && is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(friend_index(PC));
            SAMPLER_HIT[set][way]   = 0;
        } else if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = 0;
            SAMPLER_HIT[set][way]   = 0;
        }
    }

    // Periodic decay: one FRIEND entry every N demand misses
    if (demand) {
        demand_miss_ctr++;
        if (demand_miss_ctr >= DECAY_PERIOD_MISSES) {
            demand_miss_ctr = 0;
            // Decay one entry (saturating down)
            sat_dec(FRIEND[decay_ptr & (FRIEND_SIZE - 1)]);
            decay_ptr = (decay_ptr + 1) & (FRIEND_SIZE - 1);
        }
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}