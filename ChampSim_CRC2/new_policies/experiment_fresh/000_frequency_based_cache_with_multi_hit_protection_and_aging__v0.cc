#include <algorithm>
#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define AGE_INTERVAL 32 // Define the aging interval for demonstration
#define TRAIN_WINDOW 1000 // Arbitrarily chosen for demonstration purposes

using std::min;

uint8_t F[LLC_SETS][LLC_WAYS]; // 3-bit frequency counter per line
uint8_t P[LLC_SETS][LLC_WAYS]; // 1-bit prefetch flag per line
uint16_t Misses_A, Misses_B; // Track misses in leader sets
uint16_t RRptr[LLC_SETS]; // Round-robin tie-breaker pointer per set
uint16_t AgeIntervalCounter; // Main aging interval counter
uint16_t AgingSweepPtr; // Age one set at a time
int8_t PolicyCounter; // Phase control counter
uint8_t CurrentThreshold; // Current hit threshold for protection

// Initialize replacement state
void InitReplacementState() {
    std::fill(&F[0][0], &F[0][0] + sizeof(F), 0);
    std::fill(&P[0][0], &P[0][0] + sizeof(P), 0);
    std::fill(RRptr, RRptr + LLC_SETS, 0);
    Misses_A = 0;
    Misses_B = 0;
    AgeIntervalCounter = 0;
    AgingSweepPtr = 0;
    PolicyCounter = 0;
    CurrentThreshold = 2;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    uint32_t minF = 8; // Greater than max F value of 7
    uint32_t candidate[LLC_WAYS] = {0};
    uint32_t count = 0;

    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid)
            return way;
        if (F[set][way] < minF) {
            minF = F[set][way];
            candidate[0] = way;
            count = 1;
        } else if (F[set][way] == minF) {
            candidate[count++] = way;
        }
    }

    // Tie-break using round-robin pointer
    uint32_t start = RRptr[set] % count;
    uint32_t victim = candidate[(start % count)];
    RRptr[set] = (victim + 1) % LLC_WAYS;
    return victim;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    if (hit) {
        // Increment frequency on demand hit
        F[set][way] = min(F[set][way] + 1, 7);
    } else {
        // On miss, new line insertion
        F[set][way] = 0;
        P[set][way] = (type == 2);
        // Policy state adjustment
        if (true /* leader condition for A */) Misses_A++;
        if (true /* leader condition for B */) Misses_B++;
    }

    // Increment and check age interval counter
    AgeIntervalCounter += 1;
    if (AgeIntervalCounter == AGE_INTERVAL) {
        AgeIntervalCounter = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (F[AgingSweepPtr][w] > 0)
                F[AgingSweepPtr][w]--;
        }
        AgingSweepPtr = (AgingSweepPtr + 1) % LLC_SETS;
    }

    // Training logic (simplified representative strategy)
    if ((Misses_B + 5) < Misses_A) {
        PolicyCounter += 1;
    } else if ((Misses_A + 5) < Misses_B) {
        PolicyCounter -= 1;
    }

    if (PolicyCounter >= 0) {
        CurrentThreshold = 3;
    } else {
        CurrentThreshold = 2;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}