#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// RRIP parameters
static constexpr uint8_t RRPV_BITS = 2;          // 0..3
static constexpr uint8_t RRPV_MAX  = (1u << RRPV_BITS) - 1; // 3

// SHiP-lite parameters
static constexpr uint32_t SHCT_ENTRIES = 256;    // 8-bit signature
static constexpr uint8_t SHCT_BITS = 3;          // 0..7 counter
static constexpr uint8_t SHCT_MAX  = (1u << SHCT_BITS) - 1; // 7
static constexpr uint8_t SHCT_INIT = 3;          // neutral start
static constexpr uint8_t HOT_THRESHOLD = 4;      // >=4 => hot

// Special signature meaning "do not train"
static constexpr uint8_t SIG_NO_TRAIN = 0xFF;

// Replacement state
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t reuse_bit[LLC_SETS][LLC_WAYS];  // 1-bit logical
static uint8_t local_valid[LLC_SETS][LLC_WAYS]; // track validity locally
static uint8_t sig8[LLC_SETS][LLC_WAYS];       // 8-bit signature per line
static uint8_t shct[SHCT_ENTRIES];             // 3-bit saturating counters

// Small helpers
static inline uint8_t hash_sig(uint64_t PC) {
    // Lightweight 8-bit hash from PC
    uint64_t x = PC ^ (PC >> 2) ^ (PC >> 5) ^ (PC >> 13);
    return static_cast<uint8_t>(x & 0xFFu);
}
static inline void shct_inc(uint8_t s) {
    if (s == SIG_NO_TRAIN) return;
    if (shct[s] < SHCT_MAX) shct[s]++;
}
static inline void shct_dec(uint8_t s) {
    if (s == SIG_NO_TRAIN) return;
    if (shct[s] > 0) shct[s]--;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
            reuse_bit[s][w] = 0;
            local_valid[s][w] = 0;
            sig8[s][w] = SIG_NO_TRAIN;
        }
    }
    for (uint32_t i = 0; i < SHCT_ENTRIES; i++) {
        shct[i] = SHCT_INIT;
    }
}

// Find victim in the set (SRRIP victim selection)
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

    // Search for a line with RRPV == RRPV_MAX; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == RRPV_MAX) {
                return w;
            }
        }
        // Age all lines (increment RRPV up to MAX)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < RRPV_MAX) rrpv[set][w]++;
        }
    }
    // Unreachable, but keep compiler happy
    // return 0;
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
    if (hit) {
        // On hit: promote and mark reused; reinforce signature
        rrpv[set][way] = 0;
        reuse_bit[set][way] = 1;
        // Increment SHCT for this line's signature (if trainable)
        shct_inc(sig8[set][way]);
        return;
    }

    // Miss/Fill path: train using the evicted line (if previously valid)
    if (local_valid[set][way]) {
        // Decrement SHCT if the evicted line was not reused
        if (reuse_bit[set][way] == 0) {
            shct_dec(sig8[set][way]);
        }
        // If it was reused, no change (we already reinforced on hit)
    }

    // Compute new signature and set insertion policy
    uint8_t new_sig = SIG_NO_TRAIN;
    uint8_t new_rrpv = RRPV_MAX;

    if (type == PREFETCH) {
        // Always deprioritize prefetch fills
        new_sig = SIG_NO_TRAIN; // do not train prefetch PCs
        new_rrpv = RRPV_MAX;
    } else if (type == WRITEBACK) {
        // Never bypass on WRITEBACK; insert neutrally
        new_sig = SIG_NO_TRAIN; // avoid training from writebacks
        new_rrpv = (RRPV_MAX > 1) ? (RRPV_MAX - 1) : RRPV_MAX; // 2 when max=3
    } else {
        // LOAD/RFO: PC-guided insertion
        new_sig = hash_sig(PC);
        uint8_t c = shct[new_sig];
        if (c >= HOT_THRESHOLD) {
            new_rrpv = 1; // protect likely-reusable lines, but not MRU
        } else {
            new_rrpv = RRPV_MAX; // long re-reference, effectively cold
        }
    }

    // Install new line
    sig8[set][way] = new_sig;
    reuse_bit[set][way] = 0;     // not reused yet
    local_valid[set][way] = 1;   // now valid in our local view
    rrpv[set][way] = new_rrpv;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}