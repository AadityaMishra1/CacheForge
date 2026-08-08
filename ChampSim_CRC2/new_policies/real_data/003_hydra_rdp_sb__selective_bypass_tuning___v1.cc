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
// Sampled sets (every Nth set) for PC-friendliness training
static constexpr uint32_t SAMPLE_STRIDE = 64;            // 2048/64 = 32 sampled sets
// PC-friendliness table (2-bit saturated)
static constexpr uint32_t FRIEND_BITS   = 11;            // 2048 entries
static constexpr uint32_t FRIEND_SIZE   = (1u << FRIEND_BITS);
// Per-PC stream detector (±1-line stride)
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;      // hashed by PC
// Stream thresholds
static constexpr uint8_t  STREAM_TAIL_THRESH  = 2;       // conf==2 -> tail insert sentinel
static constexpr uint8_t  STREAM_BYP_THRESH   = 3;       // conf>=3 -> soft-bypass (tail+sentinel)
// Insertion depths (SRRIP)
static constexpr uint8_t INS_RRPV_FRIEND3 = 1;           // FRIEND=3 (hot)
static constexpr uint8_t INS_RRPV_FRIEND2 = 2;           // FRIEND=2
static constexpr uint8_t INS_RRPV_TAIL    = 3;           // tail insert
// Decay cadence for PC-friendliness (phase adaptation)
static constexpr uint32_t DECAY_PERIOD_DEMAND_MISSES = 256; // one entry decays every 256 demand misses

// -------------------- Per-line state --------------------
// SRRIP re-reference prediction value (0..3)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line reuse state:
// 0 = no reuse yet, 1 = one demand hit, 2 = 2+ hits (eligible for MRU), 3 = sentinel (no-train/no-promote)
static uint8_t LINE_STATE[LLC_SETS][LLC_WAYS];

// -------------------- Sampler state (sampled sets only) --------------------
static uint16_t SAMPLER_PCIDX[LLC_SETS][LLC_WAYS]; // 11-bit stored in 16
static uint8_t  SAMPLER_HIT[LLC_SETS][LLC_WAYS];   // 0/1, demand hit observed
static uint8_t  SAMPLER_VALID[LLC_SETS][LLC_WAYS]; // 0/1, entry contains a trackable line
static uint8_t  SAMPLER_TRAINABLE[LLC_SETS][LLC_WAYS]; // 0/1, eligible for training on eviction

// -------------------- Global predictors --------------------
// PC-friendliness table: 2-bit saturating counters (0=cold .. 3=hot)
static uint8_t FRIEND[FRIEND_SIZE];

// Per-PC ±1-line stream detector
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit confidence 0..3
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // last step was |delta|==1

// Decay machinery
static uint32_t decay_ptr = 0;
static uint32_t demand_miss_ctr = 0;

// -------------------- Helpers --------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }

static inline bool is_sampled_set(uint32_t set) { return (set % SAMPLE_STRIDE) == 0; }

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

// Update per-PC stream detector, return confidence after this demand access
static inline uint8_t update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(curr - prev);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc_u2(ST_CONF[idx]); // consecutive ±1 steps
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; // first indication
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        sat_dec_u2(ST_CONF[idx]); // decay on non-±1 step
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]       = 3;
            LINE_STATE[s][w] = 0;
            SAMPLER_PCIDX[s][w] = 0;
            SAMPLER_HIT[s][w]   = 0;
            SAMPLER_VALID[s][w] = 0;
            SAMPLER_TRAINABLE[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < FRIEND_SIZE; i++) {
        FRIEND[i] = 1; // neutral-cold start to avoid early over-bypass
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS1[i]  = 0;
    }
    decay_ptr = 0;
    demand_miss_ctr = 0;
}

// Find victim in the set (SRRIP with safe aging)
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
    // SRRIP search/age
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    return 0; // fallback (should not occur)
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
    const bool is_demand = (type == ACCESS_LOAD) || (type == ACCESS_RFO);

    // Update stream detector for demand accesses
    uint8_t stream_conf = 0;
    if (is_demand) {
        stream_conf = update_pc_stream(PC, paddr);
    }

    // HIT path
    if (hit) {
        // Sentinel lines (streams/prefetch/writeback/strict-bypass) never promote or train
        if (LINE_STATE[set][way] == 3) {
            // For sampled sets, we still keep valid but trainable stays as set on insert
            if (is_sampled_set(set)) {
                SAMPLER_VALID[set][way] = 1; // track presence
                // Do not flip TRAINABLE here
                if (is_demand) {
                    // Do not mark hit for sentinel to avoid noise (leave as-is)
                }
            }
            return;
        }

        // Multi-hit gating: only promote on 2nd+ demand hit
        if (is_demand) {
            if (LINE_STATE[set][way] == 0) {
                LINE_STATE[set][way] = 1; // first demand hit: no promotion
            } else {
                LINE_STATE[set][way] = 2;
                RRPV[set][way] = 0; // MRU on 2nd+ hit
            }
            // Sampler: record a demand hit
            if (is_sampled_set(set)) {
                SAMPLER_VALID[set][way] = 1;
                SAMPLER_HIT[set][way] = 1;
            }
        }
        // Prefetch/Writeback hits: no promotion
        return;
    }

    // MISS / FILL path
    // Train FRIEND on eviction for sampled sets (only if previous line was trainable)
    if (is_sampled_set(set)) {
        if (SAMPLER_VALID[set][way] && SAMPLER_TRAINABLE[set][way]) {
            uint16_t idx = SAMPLER_PCIDX[set][way];
            if (SAMPLER_HIT[set][way]) sat_inc_u2(FRIEND[idx]);
            else                       sat_dec_u2(FRIEND[idx]);
        }
        // Prepare new sampler entry
        SAMPLER_VALID[set][way]     = 1;
        SAMPLER_PCIDX[set][way]     = static_cast<uint16_t>(friend_index(PC));
        SAMPLER_HIT[set][way]       = 0;
        // TRAINABLE is decided below after we compute sentinel
    }

    // Determine insertion policy
    uint8_t new_rrpv = INS_RRPV_TAIL;
    uint8_t new_state = 0; // default non-sentinel

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback: insert at tail, sentinel (no promote/train)
        new_rrpv = INS_RRPV_TAIL;
        new_state = 3;
    } else if (type == ACCESS_PREFETCH) {
        new_rrpv = INS_RRPV_TAIL;
        new_state = 3; // never promote/train prefetch lines
    } else {
        // Demand insertion uses stream guard first
        if (stream_conf >= STREAM_BYP_THRESH) {
            new_rrpv = INS_RRPV_TAIL;
            new_state = 3; // strong stream: soft-bypass sentinel
        } else if (stream_conf == STREAM_TAIL_THRESH) {
            new_rrpv = INS_RRPV_TAIL;
            new_state = 3; // medium confidence stream: tail sentinel
        } else {
            // Use PC-friendliness predictor
            uint8_t fr = FRIEND[friend_index(PC)];
            if (fr == 0) {
                new_rrpv = INS_RRPV_TAIL;
                new_state = 3; // cold PC: bypass-like sentinel
            } else if (fr == 1) {
                new_rrpv = INS_RRPV_TAIL;
                new_state = 0; // allow potential reuse tracking
            } else if (fr == 2) {
                new_rrpv = INS_RRPV_FRIEND2;
                new_state = 0;
            } else { // fr == 3
                new_rrpv = INS_RRPV_FRIEND3;
                new_state = 0;
            }
        }
    }

    RRPV[set][way]       = new_rrpv;
    LINE_STATE[set][way] = new_state;

    // Set sampler trainability for the new line
    if (is_sampled_set(set)) {
        // Train only non-sentinel demand lines
        SAMPLER_TRAINABLE[set][way] = (is_demand && (new_state != 3)) ? 1 : 0;
    }

    // Global FRIEND decay on demand misses only
    if (is_demand) {
        demand_miss_ctr++;
        if (demand_miss_ctr >= DECAY_PERIOD_DEMAND_MISSES) {
            demand_miss_ctr = 0;
            sat_dec_u2(FRIEND[decay_ptr]);
            decay_ptr++;
            if (decay_ptr >= FRIEND_SIZE) decay_ptr = 0;
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