#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Design parameters
#define SAMPLE_RATE 32               // 3.125% sampled sets
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define SHCT_MAX 7                   // 3-bit SHiP counter
#define LIFE_MAX 3                   // 2-bit lifetime counters
#define STREAM_MAX 3                 // 2-bit stream score
#define FRIEND_THR 3                 // SHCT >= 3 => friendly
#define SHORT_THR 2                  // short >= 2 => short lifetime
#define LONG_THR 2                   // long >= 2 => long lifetime

// Access types (from champsim)
#ifndef LOAD
#define LOAD 0
#endif
#ifndef RFO
#define RFO 1
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif
#ifndef WRITEBACK
#define WRITEBACK 3
#endif

// Per-line metadata (minimized)
struct LineMeta {
    uint8_t rrpv;   // 0..3
    uint8_t hits;   // 2-bit saturating: 0,1,2+(cap at 3)
    uint8_t pf;     // prefetch bit
};

static LineMeta meta[LLC_SETS][LLC_WAYS];

// Sampled-set signature table (stores PC index for eviction training)
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // 11-bit valid indices, 0xFFFF invalid

// Global per-PC predictors
static uint8_t SHCT[PC_IDX_SIZE];          // 3-bit friendliness predictor
static uint8_t LIFE_SHORT[PC_IDX_SIZE];    // 2-bit short-life score
static uint8_t LIFE_LONG[PC_IDX_SIZE];     // 2-bit long-life score
static uint8_t PC_STREAM[PC_IDX_SIZE];     // 2-bit streaming/scan score
static uint32_t PC_LAST_LINE[PC_IDX_SIZE]; // last line address seen by PC
static int16_t PC_LAST_STRIDE[PC_IDX_SIZE];// last stride seen by PC

// Epoch counter for periodic decay (simple phase adaptation)
static uint64_t epoch_ctr = 0;

// Helpers
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix then mask to 11 bits
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}

static inline bool is_sampled(uint32_t set) {
    return (set % SAMPLE_RATE) == 0;
}

static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3; // cold
            meta[s][w].hits = 0;
            meta[s][w].pf = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[i] = FRIEND_THR;       // neutral
        LIFE_SHORT[i] = 1;          // slight bias to short
        LIFE_LONG[i] = 1;           // slight bias to long
        PC_STREAM[i] = 1;           // mild stream suspicion
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
    }
    for (uint32_t s = 0; s < (LLC_SETS / SAMPLE_RATE); s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_sig[s][w] = 0xFFFF;
        }
    }
    epoch_ctr = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP: look for RRPV==3, else age
    for (int iter = 0; iter < 4; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        // age all ways (saturating at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // Fallback (should not happen)
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
    LineMeta &m = meta[set][way];

    // Periodic decay to adapt to phase changes (rare to avoid overhead)
    if ((++epoch_ctr & ((1u << 13) - 1)) == 0) { // every 8192 events
        for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
            SHCT[i] >>= 1;
            LIFE_SHORT[i] >>= 1;
            LIFE_LONG[i] >>= 1;
            PC_STREAM[i] >>= 1;
        }
    }

    if (hit) {
        // Multi-hit promotion with first-hit hesitation
        // Only count/progress on demand hits; prefetch accesses should not promote on first hit
        bool is_demand = (type == LOAD) || (type == RFO);
        if (is_demand) {
            if (m.pf && m.hits == 0) {
                // first demand hit on a prefetched line: clear pf, do not promote
                m.pf = 0;
            }
            if (m.hits < 3) m.hits++;
            if (m.hits >= 2) {
                m.rrpv = 0; // MRU only on second hit (multi-hit promotion)
            } else {
                // first hit: mild promotion at most to RRPV=1 (scan resistant)
                if (!m.pf) {
                    if (m.rrpv > 1) m.rrpv = 1;
                }
            }
        }
        return;
    }

    // Miss path: train on evicted line if set is sampled
    if (is_sampled(set)) {
        uint16_t old_idx = sample_sig[set / SAMPLE_RATE][way];
        if (old_idx != 0xFFFF) {
            uint8_t old_hits = m.hits;
            if (old_hits == 0) {
                sat_dec(SHCT[old_idx]);               // dead -> less friendly
                sat_inc(LIFE_SHORT[old_idx], LIFE_MAX);
                if (LIFE_LONG[old_idx] > 0) LIFE_LONG[old_idx]--;
                sat_inc(PC_STREAM[old_idx], STREAM_MAX); // more likely streaming
            } else if (old_hits == 1) {
                sat_inc(SHCT[old_idx], SHCT_MAX);     // saw reuse
                sat_inc(LIFE_SHORT[old_idx], LIFE_MAX);
                if (LIFE_LONG[old_idx] > 0) LIFE_LONG[old_idx]--;
            } else { // 2+ hits: long-lived object-like
                sat_inc(SHCT[old_idx], SHCT_MAX);
                if (LIFE_SHORT[old_idx] > 0) LIFE_SHORT[old_idx]--;
                sat_inc(LIFE_LONG[old_idx], LIFE_MAX);
                if (PC_STREAM[old_idx] > 0) PC_STREAM[old_idx]--; // not a scan
            }
        }
    }

    // Compute PC-based predictions for the incoming fill
    uint32_t idx = pc_index(PC);

    // Lightweight per-PC stride/scan detector; ignore writebacks
    if (type != WRITEBACK) {
        uint64_t line = paddr >> 6;
        if (PC_LAST_LINE[idx] != 0) {
            int64_t stride = static_cast<int64_t>(line) - static_cast<int64_t>(PC_LAST_LINE[idx]);
            // stable stride or unit stride => streaming
            if (stride == 1 || stride == static_cast<int64_t>(PC_LAST_STRIDE[idx])) {
                sat_inc(PC_STREAM[idx], STREAM_MAX);
            } else {
                if (PC_STREAM[idx] > 0) PC_STREAM[idx]--;
            }
            // clamp stride to int16_t range
            if (stride > 32767) stride = 32767;
            else if (stride < -32768) stride = -32768;
            PC_LAST_STRIDE[idx] = static_cast<int16_t>(stride);
        }
        PC_LAST_LINE[idx] = static_cast<uint32_t>(line);
    }

    bool friendly = (SHCT[idx] >= FRIEND_THR);
    bool short_life = (LIFE_SHORT[idx] >= SHORT_THR);
    bool long_life = (LIFE_LONG[idx] >= LONG_THR);
    bool streaming = (PC_STREAM[idx] >= 2);

    // Adaptive insertion (soft-bypass for dead/short/scan)
    uint8_t new_rrpv = 2;
    if ((streaming && !long_life) || (short_life && !long_life) || !friendly) {
        new_rrpv = 3; // lowest priority, emulate bypass
    } else if (friendly && long_life) {
        new_rrpv = 1; // higher priority for reusable objects
    } else {
        new_rrpv = 2; // moderate
    }
    if (type == PREFETCH) {
        // De-prioritize prefetch insertions
        if (new_rrpv < 3) new_rrpv++;
    }

    // Install new line metadata
    m.rrpv = new_rrpv;
    m.hits = 0;
    m.pf = (type == PREFETCH) ? 1 : 0;

    // Record signature for training if sampled
    if (is_sampled(set)) {
        sample_sig[set / SAMPLE_RATE][way] = static_cast<uint16_t>(idx);
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