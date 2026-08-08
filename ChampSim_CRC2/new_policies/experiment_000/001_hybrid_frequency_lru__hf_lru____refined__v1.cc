#include "../inc/champsim_crc2.h"
#include <cstdint>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Per-line 3‑bit frequency counter (0‑7)
static uint8_t freq[LLC_SETS][LLC_WAYS];

// Per-line 4‑bit LRU stack position (0 = MRU, 15 = LRU)
static uint8_t lru[LLC_SETS][LLC_WAYS];

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; ++s) {
        for (uint32_t w = 0; w < LLC_WAYS; ++w) {
            freq[s][w] = 0;
            lru[s][w] = w;   // initial ordering: way 0 MRU, way 15 LRU
        }
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Return an invalid way immediately if found
    for (uint32_t w = 0; w < LLC_WAYS; ++w) {
        if (!current_set[w].valid) return w;
    }

    uint8_t min_count = 8;   // larger than any possible counter
    uint8_t min_lru   = 0;
    uint32_t victim   = 0;

    for (uint32_t w = 0; w < LLC_WAYS; ++w) {
        uint8_t c = freq[set][w];
        if (c < min_count) {
            min_count = c;
            min_lru   = lru[set][w];
            victim    = w;
        } else if (c == min_count && lru[set][w] > min_lru) {
            // Prefer the older (larger LRU value) line on tie
            min_lru = lru[set][w];
            victim  = w;
        }
    }
    return victim;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // No update on writebacks to avoid extra overhead
    if (type == WRITEBACK) return;

    // Update frequency counter
    if (hit) {
        if (freq[set][way] < 7) freq[set][way] += 1;
    } else {
        freq[set][way] = 2;   // start with a modest frequency
    }

    // Promote the accessed line to MRU (0) and shift newer lines
    uint8_t old_lru = lru[set][way];
    for (uint32_t w = 0; w < LLC_WAYS; ++w) {
        if (w == way) continue;
        if (lru[set][w] < old_lru) {  // lines newer than the accessed one
            lru[set][w] += 1;
        }
    }
    lru[set][way] = 0;
}

void PrintStats() {
    // End‑of‑simulation statistics are intentionally omitted
}

void PrintStats_Heartbeat() {
    // Periodic statistics are intentionally omitted
}