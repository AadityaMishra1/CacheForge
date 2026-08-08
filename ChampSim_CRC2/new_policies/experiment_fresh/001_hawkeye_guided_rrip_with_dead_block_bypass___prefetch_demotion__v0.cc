#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SHCT_ENTRIES 1024
#define DBPT_ENTRIES 1024
#define EPOCH_INTERVAL 16384
#define DEAD_STRONG 6

using std::min;
using std::max;

uint32_t rrpv[LLC_SETS][LLC_WAYS];
uint16_t line_sig[LLC_SETS][LLC_WAYS];
uint8_t reused[LLC_SETS][LLC_WAYS];
uint8_t is_prefetch[LLC_SETS][LLC_WAYS];
uint8_t SHCT[SHCT_ENTRIES];
uint8_t DBPT[DBPT_ENTRIES];
uint16_t epoch_counter = 0;

uint16_t PC_signature(uint64_t pc, uint32_t set) {
    return (pc >> 2 ^ set << 3) & 0x3FF;
}

uint8_t predicted_usefulness(uint16_t sig) {
    return SHCT[sig];
}

bool predicted_dead(uint16_t sig) {
    return DBPT[sig] >= DEAD_STRONG;
}

void decay_DBPT_epoch() {
    if (++epoch_counter >= EPOCH_INTERVAL) {
        for (int i = 0; i < DBPT_ENTRIES; i++) {
            DBPT[i] >>= 1;
        }
        epoch_counter = 0;
    }
}

uint8_t insertion_rrpv(uint8_t u, uint8_t is_prefetch) {
    uint8_t base = u >= 5 ? 1 : (u >= 3 ? 2 : 3);
    if (is_prefetch) base = min(3, base + 1);
    return base;
}

void InitReplacementState() {
    std::fill_n(&rrpv[0][0], LLC_SETS * LLC_WAYS, 3);
    std::memset(line_sig, 0, sizeof(line_sig));
    std::memset(reused, 0, sizeof(reused));
    std::memset(is_prefetch, 0, sizeof(is_prefetch));
    std::memset(SHCT, 0, sizeof(SHCT));
    std::memset(DBPT, 0, sizeof(DBPT));
    epoch_counter = 0;
}

uint32_t GetVictimInSet(
    uint32_t cpu, uint32_t set, const BLOCK *current_set,
    uint64_t PC, uint64_t paddr, uint32_t type
) {
    int candidate = -1;
    for (int way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) return way;
    }
    
    int best_tuple[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
    for (int way = 0; way < LLC_WAYS; way++) {
        if (rrpv[set][way] == 3) {
            uint16_t sig = line_sig[set][way];
            uint8_t u = SHCT[sig];
            int tuple[4] = {
                reused[set][way] == 0 ? 0 : 1,
                is_prefetch[set][way] ? 0 : 1,
                u,
                way
            };
            if (std::lexicographical_compare(tuple, tuple + 4, best_tuple, best_tuple + 4)) {
                candidate = way;
                std::copy(tuple, tuple + 4, best_tuple);
            }
        }
    }
    
    if (candidate != -1) return candidate;

    for (int way = 0; way < LLC_WAYS; way++) {
        rrpv[set][way] = min(3, rrpv[set][way] + 1);
    }
    return way; // nested victim removed
}

void UpdateReplacementState(
    uint32_t cpu, uint32_t set, uint32_t way,
    uint64_t paddr, uint64_t PC, uint64_t victim_addr,
    uint32_t type, uint8_t hit
) {
    decay_DBPT_epoch();

    uint16_t sig = line_sig[set][way];
    if (hit) {
        if (SHCT[sig] < 7) SHCT[sig]++;
        if (DBPT[sig] > 0) DBPT[sig]--;
        reused[set][way] = 1;
        if (type == PREFETCH) {
            rrpv[set][way] = 1;
        } else {
            rrpv[set][way] = 0;
        }
    } else {
        sig = PC_signature(PC, set);
        bool dead = predicted_dead(sig);
        uint8_t u = predicted_usefulness(sig);
        
        if (type == PREFETCH && (u <= 2 || dead)) return;
        if (dead) return;

        line_sig[set][way] = sig;
        is_prefetch[set][way] = (type == PREFETCH ? 1 : 0);
        reused[set][way] = 0;
        rrpv[set][way] = insertion_rrpv(u, type == PREFETCH);

        if (!reused[set][way]) {
            if (SHCT[sig] > 0) SHCT[sig]--;
            if (DBPT[sig] < 7) DBPT[sig]++;
        } else {
            if (SHCT[sig] < 7) SHCT[sig]++;
            if (DBPT[sig] > 0) DBPT[sig]--;
        }
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}
