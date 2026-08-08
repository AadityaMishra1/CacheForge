#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// SRRIP parameters
static const uint8_t RRPV_MAX = 3;

// SHiP signature table size (PC-indexed)
static const uint32_t SIG_BITS = 10;
static const uint32_t SIG_SIZE = (1u << SIG_BITS);

// SHCT: 3-bit saturating counters (0..7); higher => more dead
static uint8_t shct[SIG_SIZE];

// Per-set, per-way metadata
static uint8_t rrpv[LLC_SETS][LLC_WAYS];         // 2-bit logical (stored in uint8_t)
static uint16_t line_sig[LLC_SETS][LLC_WAYS];    // 10-bit signature
static uint8_t line_reuse[LLC_SETS][LLC_WAYS];   // 1 if had a demand reuse
static uint8_t line_stream[LLC_SETS][LLC_WAYS];  // 1 if stream-locked (never promote)
static uint8_t meta_valid[LLC_SETS][LLC_WAYS];   // local valid for SHCT training

// Lightweight per-PC (+64B) stream detector
static uint16_t pc_last_line16[SIG_SIZE];     // last line number (lower 16 bits of paddr>>6)
static uint8_t  pc_run_count[SIG_SIZE];       // 2-bit run length for consecutive +1 steps
static uint8_t  pc_last_dir_plus[SIG_SIZE];   // last step direction was +1

// Utility helpers
static inline uint32_t pc_sig(uint64_t PC) {
    // Simple hash of PC bits -> 10-bit signature
    return (uint32_t)((PC ^ (PC >> 2) ^ (PC >> 13)) & (SIG_SIZE - 1));
}

static inline bool is_prefetch(uint32_t type) { return type == 2; }
static inline bool is_demand(uint32_t type) { return (type == 0) || (type == 1); }
static inline bool is_writeback(uint32_t type) { return !is_demand(type) && !is_prefetch(type); }

static inline uint8_t sat_inc(uint8_t v, uint8_t maxv) { return (v < maxv) ? (uint8_t)(v + 1) : v; }
static inline uint8_t sat_dec(uint8_t v) { return (v > 0) ? (uint8_t)(v - 1) : v; }

// Update per-PC stream detector; returns true if streaming (+1) with confidence >= 2
static inline bool update_and_is_stream(uint32_t sig, uint64_t paddr) {
    uint16_t cur_line16 = (uint16_t)((paddr >> 6) & 0xFFFF);
    bool plus1 = (((uint16_t)(pc_last_line16[sig] + 1)) == cur_line16);

    if (plus1) {
        if (pc_last_dir_plus[sig]) pc_run_count[sig] = sat_inc(pc_run_count[sig], 3);
        else pc_run_count[sig] = 1; // first +1 step
        pc_last_dir_plus[sig] = 1;
    } else {
        pc_run_count[sig] = 0;
        pc_last_dir_plus[sig] = 0;
    }
    pc_last_line16[sig] = cur_line16;

    // Require at least two consecutive +1 steps to consider it a stream
    return (pc_run_count[sig] >= 2);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;     // distant by default
            line_sig[s][w] = 0;
            line_reuse[s][w] = 0;
            line_stream[s][w] = 0;
            meta_valid[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < SIG_SIZE; i++) {
        shct[i] = 3; // neutral initial confidence (0..7)
        pc_last_line16[i] = 0;
        pc_run_count[i] = 0;
        pc_last_dir_plus[i] = 0;
    }
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP search for RRPV == MAX; age if necessary
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == RRPV_MAX) return w;
        }
        // Increment all RRPVs (saturate)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < RRPV_MAX) rrpv[set][w]++;
        }
    }
    // Unreachable
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
    uint32_t sig = pc_sig(PC);

    // Update stream detector on every access
    bool pc_stream = update_and_is_stream(sig, paddr);

    if (hit) {
        // On hits: prefetch hits never promote; demand hits use multi-hit gating
        if (is_demand(type)) {
            if (!line_stream[set][way]) {
                if (line_reuse[set][way]) {
                    // Second (or later) demand hit: promote to MRU (RRPV=0)
                    rrpv[set][way] = 0;
                } else {
                    // First demand hit: mark reuse, do not promote
                    line_reuse[set][way] = 1;
                }
            }
            // stream-locked lines never promote, but keep their reuse bit as 0 to train deadness
        }
        // Prefetch hit: do nothing (no promotion, no reuse credit)
        return;
    }

    // Miss: train SHCT on the evicted line (if valid in our metadata)
    if (meta_valid[set][way]) {
        uint16_t ev_sig = line_sig[set][way] & (SIG_SIZE - 1);
        if (line_reuse[set][way] == 0) {
            // Dead: increment deadness
            shct[ev_sig] = sat_inc(shct[ev_sig], 7);
        } else {
            // Reused: decrement deadness
            shct[ev_sig] = sat_dec(shct[ev_sig]);
        }
    }

    // Decide insertion policy for the incoming line
    uint8_t new_rrpv = RRPV_MAX; // default distant
    uint8_t new_stream_lock = 0;

    if (is_prefetch(type)) {
        // Prefetches: always lowest priority and never promote
        new_rrpv = RRPV_MAX;
        new_stream_lock = 1;
    } else if (is_demand(type)) {
        if (pc_stream) {
            // Aggressive streaming bypass: lowest priority and locked
            new_rrpv = RRPV_MAX;
            new_stream_lock = 1;
        } else {
            // SHiP-guided insertion for demand
            uint8_t conf = shct[sig];
            if (conf <= 1)      new_rrpv = RRPV_MAX; // predicted cold -> distant
            else if (conf <= 3) new_rrpv = RRPV_MAX - 1; // medium
            else                new_rrpv = 1;       // hot -> near-MRU
            new_stream_lock = 0;
        }
    } else {
        // Writeback or others: do not bypass; insert moderately
        new_rrpv = RRPV_MAX - 1;
        new_stream_lock = 0;
    }

    // Install metadata for the new line
    rrpv[set][way] = new_rrpv;
    line_sig[set][way] = (uint16_t)sig;
    line_reuse[set][way] = 0;         // must earn reuse on demand hit
    line_stream[set][way] = new_stream_lock;
    meta_valid[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}