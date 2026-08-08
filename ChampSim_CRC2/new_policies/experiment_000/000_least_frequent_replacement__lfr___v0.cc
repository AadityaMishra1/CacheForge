#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// 3‑bit counter per cache line
static uint8_t freq[LLC_SETS][LLC_WAYS];

/* --------------------------------------------------------------------- */
/* Initialize replacement state                                           */
/* --------------------------------------------------------------------- */
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; ++s) {
        for (uint32_t w = 0; w < LLC_WAYS; ++w) {
            freq[s][w] = 0;      // start with zero accesses
        }
    }
}

/* --------------------------------------------------------------------- */
/* Find victim in the set                                                */
/* --------------------------------------------------------------------- */
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // 1. Return first invalid way if available
    for (uint32_t w = 0; w < LLC_WAYS; ++w) {
        if (!current_set[w].valid) return w;
    }

    // 2. All ways are valid – pick the least‑frequent line
    uint8_t min_freq = 8;           // larger than any counter
    uint32_t victim = 0;
    for (uint32_t w = 0; w < LLC_WAYS; ++w) {
        uint8_t f = freq[set][w];
        if (f < min_freq) {
            min_freq = f;
            victim   = w;
        }
    }
    return victim;
}

/* --------------------------------------------------------------------- */
/* Update replacement state                                             */
/* --------------------------------------------------------------------- */
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
    // Ignore writebacks – replacement logic must not act on them
    if (type == WRITEBACK) return;

    if (hit) {                      // hit: increment counter (capped at 7)
        if (freq[set][way] < 7) ++freq[set][way];
    } else {                        // miss/fill: start with 1
        freq[set][way] = 1;
    }
}

/* --------------------------------------------------------------------- */
/* Print end‑of‑simulation statistics                                   */
/* --------------------------------------------------------------------- */
void PrintStats() {
    // This policy does not report any special statistics
}

/* --------------------------------------------------------------------- */
/* Print periodic (heartbeat) statistics                               */
/* --------------------------------------------------------------------- */
void PrintStats_Heartbeat() {
    // No heartbeat statistics required
}