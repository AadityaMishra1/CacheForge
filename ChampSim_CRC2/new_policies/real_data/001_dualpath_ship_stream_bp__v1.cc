#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (best-effort for CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// SRRIP state: we store in 8-bit but logically use 2 bits (0..3)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line signature (10-bit index into SHCT; stored in 16-bit)
static uint16_t LINE_SIG[LLC_SETS][LLC_WAYS];

// Per-line capped hit counter:
// 0 = not yet re-referenced, 1 = one hit, 2 = 2+ hits (eligible for MRU promotion)
// 3 = sentinel: "no-train/no-promote" (used for prefetches and writebacks)
static uint8_t LINE_HITS[LLC_SETS][LLC_WAYS];

// Per-line stream-mark: lines from streaming PCs never promote
static uint8_t LINE_STREAM[LLC_SETS][LLC_WAYS]; // 0/1

// SHiP: signature history counter table (3-bit saturating, 0..7)
static constexpr uint32_t SHIP_SIG_BITS = 10;
static constexpr uint32_t SHCT_SIZE = (1u << SHIP_SIG_BITS);
static uint8_t SHCT[SHCT_SIZE];

// Lightweight per-PC stream detector for ±1-line strides
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit (0..3)
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // whether previous step had |delta|==1

// Helpers
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline uint32_t ship_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 10) ^ (pc >> 20);
    return static_cast<uint32_t>(x) & (SHCT_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu); // 64B lines
}

// Update stream detector and return confidence after this access
static inline uint8_t update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(curr - prev);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc(ST_CONF[idx], 3); // consecutive abs1
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
            RRPV[s][w]       = 3; // distant re-ref
            LINE_SIG[s][w]   = 0;
            LINE_HITS[s][w]  = 0;
            LINE_STREAM[s][w]= 0;
        }
    }
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        SHCT[i] = 3; // neutral start
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS1[i]  = 0;
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
    // SRRIP selection: look for RRPV==3; if none, age all and retry
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
    // Update stream detector with this access and snapshot confidence
    uint8_t stream_conf = update_pc_stream(PC, paddr);
    const uint32_t sig = ship_index(PC);

    if (hit) {
        // No-op for prefetch/writeback sentinel lines
        if (LINE_HITS[set][way] == 3) return;

        // Streaming lines: never promote, but allow adaptation in SHCT on first hit
        if (LINE_STREAM[set][way]) {
            if (LINE_HITS[set][way] == 0) sat_inc(SHCT[LINE_SIG[set][way]], 7);
            if (LINE_HITS[set][way] < 2) LINE_HITS[set][way]++; // cap at 2 for normal lines
            return;
        }

        // Normal multi-hit promotion and SHiP training
        if (LINE_HITS[set][way] == 0) {
            sat_inc(SHCT[LINE_SIG[set][way]], 7); // first reuse => live
            if (RRPV[set][way] > 0) RRPV[set][way]--; // mild promotion
            LINE_HITS[set][way] = 1;
        } else {
            LINE_HITS[set][way] = 2; // 2+ hits
            RRPV[set][way] = 0;      // strong promotion
        }
        return;
    }

    // Miss path: train on evicted occupant (dead-on-evict)
    if (victim_addr != 0) {
        // Decrement only if the previous line had no hits and was trainable
        if (LINE_HITS[set][way] == 0) {
            uint32_t prev_sig = LINE_SIG[set][way] & (SHCT_SIZE - 1);
            sat_dec(SHCT[prev_sig]);
        }
    }

    // Decide insertion policy for the incoming line
    // WRITEBACK: never bypass; insert at lowest priority; no training/promotion
    if (type == ACCESS_WRITEBACK) {
        RRPV[set][way]       = 3;
        LINE_SIG[set][way]   = 0;    // don't care; won't train (sentinel)
        LINE_HITS[set][way]  = 3;    // sentinel: no-train/no-promote
        LINE_STREAM[set][way]= 0;
        return;
    }

    // PREFETCH: always lowest priority; never train or promote
    if (type == ACCESS_PREFETCH) {
        RRPV[set][way]       = 3;
        LINE_SIG[set][way]   = sig;  // not used for training (sentinel)
        LINE_HITS[set][way]  = 3;    // sentinel: no-train/no-promote
        LINE_STREAM[set][way]= (stream_conf >= 1) ? 1 : 0;
        return;
    }

    // Demand / RFO: combine stream detection with SHiP
    bool streamy = (stream_conf >= 1); // conf=1 => treat as streaming candidate
    if (streamy) {
        // Aggressive stream protection: approximate bypass by lowest insertion and no promotion
        RRPV[set][way]       = 3;
        LINE_SIG[set][way]   = sig;
        LINE_HITS[set][way]  = 0;    // allow dead-on-evict to train SHCT as dead
        LINE_STREAM[set][way]= 1;
        return;
    }

    // Non-streaming: SHiP-guided adaptive insertion
    uint8_t c = SHCT[sig];
    bool strong_dead = (c <= 1);
    bool predicted_live = (c >= 5);

    LINE_SIG[set][way]   = sig;
    LINE_HITS[set][way]  = 0;
    LINE_STREAM[set][way]= 0;

    if (strong_dead) {
        // Predicted cold: lowest insertion to limit pollution
        RRPV[set][way] = 3;
    } else if (predicted_live) {
        // Predicted hot: optimistic but cautious
        RRPV[set][way] = 1;
    } else {
        // Neutral
        RRPV[set][way] = 2;
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