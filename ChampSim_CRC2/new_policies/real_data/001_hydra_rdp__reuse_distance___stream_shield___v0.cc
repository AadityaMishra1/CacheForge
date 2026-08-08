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
// Sampled sets (every Nth set) for PC friendliness training
static constexpr uint32_t SAMPLE_STRIDE = 32;           // 2048/32 = 64 sampled sets
// PC friendliness table (2-bit saturated)
static constexpr uint32_t FRIEND_BITS   = 11;           // 2048 entries
static constexpr uint32_t FRIEND_SIZE   = (1u << FRIEND_BITS);
// Per-PC stream detector (±1-line stride)
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;     // small, hashed by PC
static constexpr uint8_t  STREAM_THRESH = 2;            // require 2 consecutive steps
// Insertion depths
static constexpr uint8_t INS_FRIENDLY = 1;              // predicted-friendly PCs
static constexpr uint8_t INS_NEUTRAL  = 2;              // weak PCs
static constexpr uint8_t INS_COLD     = 3;              // cold PCs and sentinels
// Decay cadence for PC friendliness (phase adaptation)
static constexpr uint32_t DECAY_PERIOD_MISSES = 64;     // one entry decays every 64 demand misses

// -------------------- Per-line state --------------------
// SRRIP re-reference prediction value (logical 2-bit 0..3 stored in 8-bit)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line reuse state:
// 0 = no reuse yet, 1 = one hit, 2 = 2+ hits (eligible for MRU promotion), 3 = sentinel (no-train/no-promote)
static uint8_t LINE_STATE[LLC_SETS][LLC_WAYS];

// -------------------- Sampler state (sampled sets only) --------------------
// For sampled sets, keep PC-index and a "saw hit" bit to train friendliness on eviction
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

// Update per-PC stream detector, return confidence after this access (demand only)
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

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]       = 3;
            LINE_STATE[s][w] = 0;
            SAMPLER_PCIDX[s][w] = 0;
            SAMPLER_HIT[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < FRIEND_SIZE; i++) {
        FRIEND[i] = 1; // neutral-cold start (avoid overly aggressive early bypass)
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
    // First, prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: look for RRPV==3; if none, age and retry
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    return 0; // safe fallback (should not happen)
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

    // Update stream detector (demand only)
    uint8_t stream_conf = 0;
    if (is_demand) {
        stream_conf = update_pc_stream(PC, paddr);
    }

    // HIT path
    if (hit) {
        // Never promote/train for sentinel lines (streams/prefetches/writebacks/strict-bypass)
        if (LINE_STATE[set][way] == 3) return;

        // For sampled sets, mark that we observed a reuse
        if (is_sampled_set(set)) {
            SAMPLER_HIT[set][way] = 1;
        }

        // Multi-hit gating: first hit = no-op; second+ hits => MRU
        if (LINE_STATE[set][way] == 0) {
            LINE_STATE[set][way] = 1; // first hit recorded
            return;
        }
        // second or later hit
        LINE_STATE[set][way] = 2;
        RRPV[set][way] = 0; // promote to MRU
        return;
    }

    // MISS path
    if (is_demand) {
        // age friendliness table gradually (phase adaptation)
        demand_miss_ctr++;
        if ((demand_miss_ctr % DECAY_PERIOD_MISSES) == 0) {
            if (FRIEND[decay_ptr] > 0) FRIEND[decay_ptr]--;
            decay_ptr = (decay_ptr + 1) & (FRIEND_SIZE - 1);
        }
    }

    // Train friendliness on the evicted occupant for sampled sets only
    if (victim_addr != 0 && is_sampled_set(set)) {
        uint32_t prev_idx = SAMPLER_PCIDX[set][way] & (FRIEND_SIZE - 1);
        if (SAMPLER_HIT[set][way] == 0) {
            sat_dec(FRIEND[prev_idx]);
        } else {
            sat_inc(FRIEND[prev_idx], 3);
        }
    }

    // Decide insertion behavior for incoming line
    // WRITEBACK: never bypass; insert at tail with sentinel
    if (type == ACCESS_WRITEBACK) {
        RRPV[set][way]       = INS_COLD;
        LINE_STATE[set][way] = 3; // sentinel: no-train/no-promote
        if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = 0;
            SAMPLER_HIT[set][way]   = 0;
        }
        return;
    }

    // PREFETCH: insert at tail; never promote or train
    if (type == ACCESS_PREFETCH) {
        RRPV[set][way]       = INS_COLD;
        LINE_STATE[set][way] = 3; // sentinel
        if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(friend_index(PC));
            SAMPLER_HIT[set][way]   = 0;
        }
        return;
    }

    // Demand insertion
    // Stream shield: two consecutive ±1-line steps => approximate bypass (tail + sentinel)
    if (stream_conf >= STREAM_THRESH) {
        RRPV[set][way]       = INS_COLD;
        LINE_STATE[set][way] = 3; // sentinel, never promote/train
        if (is_sampled_set(set)) {
            SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(friend_index(PC));
            SAMPLER_HIT[set][way]   = 0;
        }
        return;
    }

    // Non-stream demand: consult PC friendliness
    uint32_t idx = friend_index(PC);
    uint8_t score = FRIEND[idx]; // 0..3

    if (is_sampled_set(set)) {
        SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(idx);
        SAMPLER_HIT[set][way]   = 0;
    }

    if (score == 0) {
        // Strongly cold => strict bypass approximation
        RRPV[set][way]       = INS_COLD;
        LINE_STATE[set][way] = 3; // sentinel
        return;
    } else if (score == 1) {
        // Weak/uncertain => conservative insertion; allow multi-hit gating
        RRPV[set][way]       = INS_NEUTRAL;
        LINE_STATE[set][way] = 0;
        return;
    } else {
        // Friendly => slightly optimistic insertion; still multi-hit gated
        RRPV[set][way]       = INS_FRIENDLY;
        LINE_STATE[set][way] = 0;
        return;
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