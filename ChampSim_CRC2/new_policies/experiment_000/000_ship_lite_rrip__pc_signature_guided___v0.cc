#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// RRIP parameters
static constexpr uint8_t RRPV_MAX = 3;

// SHiP-Lite parameters
static constexpr uint32_t SIG_COUNT = 1024;      // number of PC-signature entries (power of two)
static constexpr uint8_t SIG_CTR_BITS = 3;       // 3-bit saturating counters
static constexpr uint8_t SIG_CTR_MAX  = (1u << SIG_CTR_BITS) - 1; // 7
static constexpr uint8_t BAD_THRESHOLD = 4;      // >=4 => bad (insert at RRPV_MAX)

// Replacement metadata
static uint8_t  rrpv[LLC_SETS][LLC_WAYS];
static uint16_t line_sig[LLC_SETS][LLC_WAYS]; // stores 10-bit signature index
static uint8_t  reused[LLC_SETS][LLC_WAYS];   // 1 if line has seen at least one hit since fill

// Global signature predictor (3-bit counters)
static uint8_t sig_ctrs[SIG_COUNT];

// Simple hash to 10-bit signature index
static inline uint32_t pc_signature(uint64_t PC) {
    // Mix a few PC bits then mask to table size
    uint64_t x = PC ^ (PC >> 2) ^ (PC >> 10) ^ (PC >> 18);
    return static_cast<uint32_t>(x) & (SIG_COUNT - 1);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;   // cold by default
            line_sig[s][w] = 0;      // arbitrary; guarded by reused=1 initially
            reused[s][w] = 1;        // treat uninitialized lines as already "reused" to avoid false penalties
        }
    }
    // Initialize signature counters to neutral (midpoint)
    uint8_t neutral = BAD_THRESHOLD - 1; // 3
    for (uint32_t i = 0; i < SIG_COUNT; i++) {
        sig_ctrs[i] = neutral;
    }
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return an invalid way immediately if any
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid)
            return w;
    }

    // SRRIP victim selection: look for RRPV==MAX; if none, age all and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == RRPV_MAX)
                return w;
        }
        // Increment RRPV (saturating) for all ways in the set
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < RRPV_MAX)
                rrpv[set][w]++;
        }
    }
    // Unreachable
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t /*paddr*/,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t /*type*/,
    uint8_t hit
) {
    if (hit) {
        // Hit: promote and reward signature once (first hit only)
        rrpv[set][way] = 0;
        if (reused[set][way] == 0) {
            uint32_t sig = line_sig[set][way];
            if (sig_ctrs[sig] > 0) sig_ctrs[sig]--;
            reused[set][way] = 1;
        }
        return;
    }

    // Miss/Fill path:
    // 1) Penalize the evicted line's signature if it never hit
    if (reused[set][way] == 0) {
        uint32_t old_sig = line_sig[set][way];
        if (sig_ctrs[old_sig] < SIG_CTR_MAX) sig_ctrs[old_sig]++;
    }

    // 2) Compute new line's signature and decide insertion RRPV
    uint32_t sig = pc_signature(PC);
    line_sig[set][way] = static_cast<uint16_t>(sig);
    reused[set][way] = 0; // track first hit

    uint8_t ctr = sig_ctrs[sig];
    uint8_t ins_rrpv = (ctr >= BAD_THRESHOLD) ? RRPV_MAX : 1; // protect good, age bad
    rrpv[set][way] = ins_rrpv;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}