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

// ------- Per-line state -------
static uint8_t  RRPV[LLC_SETS][LLC_WAYS];       // logical 2-bit (0..3), stored in 8-bit
static uint16_t LINE_PCIDX[LLC_SETS][LLC_WAYS]; // hashed PC index (we use 10 bits of this)
static uint8_t  LINE_HITS[LLC_SETS][LLC_WAYS];  // 0=no hit yet, 1=one hit, 2=2+ hits
static uint8_t  LINE_SENT[LLC_SETS][LLC_WAYS];  // 0=normal(trainable), 1=sentinel(no-train/no-promote)

// ------- Global predictors -------
// Friendly-PC predictor: 1024-entry 3-bit counters (0..7)
static constexpr uint32_t FPC_BITS = 10;
static constexpr uint32_t FPC_SIZE = (1u << FPC_BITS);
static uint8_t FPC[FPC_SIZE];

// Per-PC ±1-line stream detector (1024 entries)
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit confidence (0..3)
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // whether previous step was |delta|==1

// DRRIP-style set-dueling PSEL
static uint16_t PSEL = 512; // 10-bit effective range [0..1023]

// ---------- Helpers ----------
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }
static inline void psel_inc() { if (PSEL < 1023) PSEL++; }
static inline void psel_dec() { if (PSEL > 0) PSEL--; }

static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 10) ^ (pc >> 20);
    return static_cast<uint32_t>(x) & (FPC_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu); // 64B line number low16
}

// Update per-PC stream detector on demand access and return confidence (0..3)
static inline uint8_t update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(curr - prev);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc_u8(ST_CONF[idx], 3); // consecutive ±1 steps
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        sat_dec_u8(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

static inline bool is_leaderA(uint32_t set) { return (set % 64u) == 0u; }   // conservative: RRPV=3 for neutral
static inline bool is_leaderB(uint32_t set) { return (set % 64u) == 32u; } // aggressive:   RRPV=2 for neutral

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]      = 3;
            LINE_PCIDX[s][w]= 0;
            LINE_HITS[s][w] = 0;
            LINE_SENT[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < FPC_SIZE; i++) FPC[i] = 3; // neutral start
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    PSEL = 512;
}

// Find victim in the set (SRRIP)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: look for RRPV==3; if none, age and retry (bounded rounds)
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    return 0; // safe fallback
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

    // HIT path
    if (hit) {
        // Sentinels (streams/prefetches/writebacks) never promote nor train
        if (LINE_SENT[set][way]) return;

        // Multi-hit promotion: only 2nd+ demand hit promotes to MRU
        if (LINE_HITS[set][way] == 0) {
            LINE_HITS[set][way] = 1; // first reuse seen; no promotion
        } else {
            LINE_HITS[set][way] = 2; // 2+ hits
            RRPV[set][way] = 0;      // MRU
        }
        return;
    }

    // MISS path: train on the evicted occupant (dead-on-evict feedback)
    if (victim_addr != 0) {
        if (!LINE_SENT[set][way]) {
            uint32_t prev_idx = LINE_PCIDX[set][way] & (FPC_SIZE - 1);
            if (LINE_HITS[set][way] == 0) {
                sat_dec_u8(FPC[prev_idx]); // dead-on-evict => less friendly
            } else {
                sat_inc_u8(FPC[prev_idx], 7); // saw reuse while resident
            }
        }
    }

    // Decide insertion policy for the incoming line
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback
        RRPV[set][way]       = 3;
        LINE_PCIDX[set][way] = 0;
        LINE_HITS[set][way]  = 0;
        LINE_SENT[set][way]  = 1; // sentinel: no-train/no-promote
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetches: insert at tail, never promote, never train
        RRPV[set][way]       = 3;
        LINE_PCIDX[set][way] = 0;
        LINE_HITS[set][way]  = 0;
        LINE_SENT[set][way]  = 1; // sentinel
        return;
    }

    // Demand: optionally update stream detector and apply stream shielding
    uint8_t stream_conf = update_pc_stream(PC, paddr);
    if (stream_conf >= 2) {
        // Stream-like: approximate bypass (tail insert, no-train/no-promote)
        RRPV[set][way]       = 3;
        LINE_PCIDX[set][way] = 0;
        LINE_HITS[set][way]  = 0;
        LINE_SENT[set][way]  = 1; // sentinel
        // DRRIP set-dueling update on demand miss
        if (is_leaderA(set)) psel_inc();
        if (is_leaderB(set)) psel_dec();
        return;
    }

    // Friendly-PC guided insertion for normal demand lines
    uint32_t idx = pc_index(PC);
    uint8_t pred = FPC[idx];

    uint8_t insert_rrpv;
    if (pred <= 1) {              // strong dead/cold PCs
        insert_rrpv = 3;
    } else if (pred >= 5) {       // friendly PCs
        insert_rrpv = 1;
    } else {
        // Neutral PCs: DRRIP-style dueling for phase adaptivity
        bool aggressive = (PSEL < 512);
        insert_rrpv = aggressive ? 2 : 3;
    }

    RRPV[set][way]       = insert_rrpv;
    LINE_PCIDX[set][way] = static_cast<uint16_t>(idx);
    LINE_HITS[set][way]  = 0;
    LINE_SENT[set][way]  = 0; // trainable

    // Update PSEL on demand misses in leader sets
    if (is_leaderA(set)) psel_inc();
    if (is_leaderB(set)) psel_dec();
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}