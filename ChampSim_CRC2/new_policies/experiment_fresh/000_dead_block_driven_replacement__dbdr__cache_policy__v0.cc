#include <algorithm>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SHCT_ENTRIES 8192
#define RDT_ENTRIES 2048
#define MAX_RRPV 3

using std::min;
using std::max;

// Global metadata structures
uint8_t SHCT[SHCT_ENTRIES];
uint8_t RDT[RDT_ENTRIES];
uint8_t rrpv[LLC_SETS][LLC_WAYS];
bool dead_bit[LLC_SETS][LLC_WAYS];
bool reused_bit[LLC_SETS][LLC_WAYS];
bool prefetch_bit[LLC_SETS][LLC_WAYS];
uint16_t pc_sig[LLC_SETS][LLC_WAYS];

// Initialize replacement state
void InitReplacementState() {
    std::fill_n(&SHCT[0], SHCT_ENTRIES, 0);
    std::fill_n(&RDT[0], RDT_ENTRIES, 0);
    for (unsigned i = 0; i < LLC_SETS; i++) {
        for (unsigned j = 0; j < LLC_WAYS; j++) {
            rrpv[i][j] = MAX_RRPV;
            dead_bit[i][j] = false;
            reused_bit[i][j] = false;
            prefetch_bit[i][j] = false;
            pc_sig[i][j] = 0;
        }
    }
}

// Hash functions (mocked for illustration; replace with suitable functions)
inline uint16_t hash12(uint64_t addr) { return (addr ^ (addr >> 12)) & 0xFFF; }
inline uint16_t hash11(uint64_t addr) { return (addr ^ (addr >> 11)) & 0x7FF; }

// Find a victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // First, look for an invalid (empty) way
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (current_set[way].valid == false) {
            return way;
        }
    }

    // Victim selection logic - choose dead if present, else RRIP
    while (true) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (dead_bit[set][way] && rrpv[set][way] == MAX_RRPV) {
                return way;
            }
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (dead_bit[set][way]) {
                rrpv[set][way]++;
            }
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] == MAX_RRPV) {
                return way;
            }
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            rrpv[set][way]++;
        }
    }
    return 0; // Fallback, should never hit due to loop
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
    uint16_t pc = hash12(PC);
    uint16_t region = hash11(paddr >> 12);

    if (hit) {
        dead_bit[set][way] = false;
        rrpv[set][way] = 0;
        if ((set & 0x3FF) < 512) { // Sampling: first 512 sets
            reused_bit[set][way] = true;
        }
        if (dead_bit[set][way]) {
            SHCT[pc] = max(SHCT[pc] - 1, 0);
            RDT[region] = max(RDT[region] - 1, 0);
        }
    } else {
        bool is_prefetch = (type == PREFETCH);
        bool pred_dead = (SHCT[pc] == 3 || RDT[region] == 3 ||
                          (is_prefetch && (SHCT[pc] >= 2 || RDT[region] >= 2)));

        if (pred_dead) {
            dead_bit[set][way] = true;
            rrpv[set][way] = MAX_RRPV;
        } else {
            dead_bit[set][way] = false;
            rrpv[set][way] = 2; // Averse insert
        }

        if ((set & 0x3FF) < 512) { // Sampling: first 512 sets
            pc_sig[set][way] = pc;
            reused_bit[set][way] = false;
        }
    }
}

// Print functions (leave empty as per requirements)
void PrintStats() {}
void PrintStats_Heartbeat() {}