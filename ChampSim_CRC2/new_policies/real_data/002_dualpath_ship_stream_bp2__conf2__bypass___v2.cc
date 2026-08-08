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

// SRRIP per-line state (logical 2-bit 0..3, stored in 8-bit)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line SHiP signature (11-bit index into SHCT)
static uint16_t LINE_SIG[LLC_SETS][LLC_WAYS];

// Per-line hit state:
// 0 = no reuse yet, 1 = one hit seen, 2 = 2+ hits,
// 3 = sentinel (no-train/no-promote): streams, prefetches, writebacks
static uint8_t LINE_HITS[LLC_SETS][LLC_WAYS];

// SHiP: 2048-entry, 3-bit saturating counter table (0..7)
static constexpr uint32_t SHIP_SIG_BITS = 11;
static constexpr uint32_t SHCT_SIZE = (1u << SHIP_SIG_BITS);
static uint8_t SHCT[SHCT_SIZE];

// Lightweight per-PC ±1-line stream detector
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit confidence (0..3)
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // last step was |delta|==1

// Helpers
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline uint32_t ship_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (SHCT_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu); // 64B line number low16
}

// Update stream detector and return confidence after this access (demand only)
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
            // first indication of ±1 step
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
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
            RRPV[s][w]      = 3;
            LINE_SIG[s][w]  = 0;
            LINE_HITS[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        SHCT[i] = 2; // slightly cold start
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i] = 0;
        ST_LAST_ABS1[i] = 0;
    }
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
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: search for RRPV==3; if none, age and retry
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
    uint8_t stream_conf = 0;
    if (is_demand) {
        // update detector on all demand accesses (hits+misses)
        stream_conf = update_pc_stream(PC, paddr);
    }
    const uint32_t sig = ship_index(PC);

    // HIT path
    if (hit) {
        // sentinel lines (streams/prefetches/wbs): never promote nor train
        if (LINE_HITS[set][way] == 3) return;

        // First hit: train SHCT; promotion only if currently strong (proxy via SHCT threshold)
        if (LINE_HITS[set][way] == 0) {
            sat_inc(SHCT[LINE_SIG[set][way]], 7);
            if (SHCT[LINE_SIG[set][way]] >= 5) {
                if (RRPV[set][way] > 0) RRPV[set][way]--; // mild promotion for strong PCs
            }
            LINE_HITS[set][way] = 1;
            return;
        }

        // Second or later hit: promote to MRU
        LINE_HITS[set][way] = 2;
        RRPV[set][way] = 0;
        return;
    }

    // MISS path: train on the evicted occupant (dead-on-evict), excluding sentinels
    if (victim_addr != 0) {
        if (LINE_HITS[set][way] == 0) {
            uint32_t prev_sig = LINE_SIG[set][way] & (SHCT_SIZE - 1);
            // Do not train on sentinels (3)
            if (LINE_HITS[set][way] != 3) {
                sat_dec(SHCT[prev_sig]);
            }
        }
    }

    // Insertion policy for the incoming line
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert far and never promote/train
        RRPV[set][way]      = 3;
        LINE_HITS[set][way] = 3; // sentinel
        LINE_SIG[set][way]  = 0;
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetches: low priority, no promotion, no training
        RRPV[set][way]      = 3;
        LINE_HITS[set][way] = 3; // sentinel
        LINE_SIG[set][way]  = 0;
        return;
    }

    // Demand insertions: apply stream bypass intent first
    if (stream_conf >= 2) {
        // True-bypass intent: occupy at lowest priority, no training/promotion
        RRPV[set][way]      = 3;
        LINE_HITS[set][way] = 3; // sentinel
        LINE_SIG[set][way]  = sig; // keep for potential debugging; won't train
        return;
    }
    if (stream_conf == 1) {
        // Low-confidence stream: identical to bypass treatment
        RRPV[set][way]      = 3;
        LINE_HITS[set][way] = 3; // sentinel
        LINE_SIG[set][way]  = sig;
        return;
    }

    // Non-stream demand: SHiP-guided adaptive insertion
    uint8_t c = SHCT[sig];
    uint8_t ins_rrpv = (c <= 1) ? 3 : ((c >= 5) ? 1 : 2);
    RRPV[set][way]      = ins_rrpv;
    LINE_HITS[set][way] = 0;     // eligible for first-hit training
    LINE_SIG[set][way]  = sig;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}