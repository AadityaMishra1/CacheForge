#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// RRIP parameters
static constexpr uint8_t RRPV_MAX = 3;

// SHiP-Lite parameters (tri-level insertion via 3-bit counters)
static constexpr uint32_t SIG_COUNT = 1024;         // number of PC-signature entries (power of two)
static constexpr uint8_t  SIG_CTR_BITS = 3;         // 3-bit saturating counters (0..7)
static constexpr uint8_t  SIG_CTR_MAX  = (1u << SIG_CTR_BITS) - 1; // 7
static constexpr uint8_t  GOOD_THRESHOLD = 2;       // <=2 => good
static constexpr uint8_t  BAD_THRESHOLD  = 5;       // >=5 => bad (insert deepest)

// Request type conventions (CRC2 typical)
static constexpr uint32_t TYPE_LOAD      = 0;
static constexpr uint32_t TYPE_RFO       = 1;
static constexpr uint32_t TYPE_PREFETCH  = 2;
static constexpr uint32_t TYPE_WRITEBACK = 3;

// Replacement metadata
static uint8_t  rrpv[LLC_SETS][LLC_WAYS];     // 2-bit conceptually (0..3)
static uint16_t line_sig[LLC_SETS][LLC_WAYS]; // stores 10-bit signature index
static uint8_t  reused[LLC_SETS][LLC_WAYS];   // 1 if line has seen at least one hit since fill

// Global signature predictor (3-bit saturating counters)
static uint8_t sig_ctrs[SIG_COUNT];

// Hash to 10-bit signature index using PC and set to reduce aliasing
static inline uint32_t pc_signature(uint64_t PC, uint32_t set) {
    uint64_t x = PC ^ (PC >> 2) ^ (PC >> 5) ^ (PC >> 13);
    x ^= (static_cast<uint64_t>(set) << 1) ^ (static_cast<uint64_t>(set) >> 3);
    x ^= (x >> 10);
    return static_cast<uint32_t>(x) & (SIG_COUNT - 1);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;   // cold by default
            line_sig[s][w] = 0;      // don't-care initially
            reused[s][w] = 1;        // avoid falsely penalizing uninitialized contents
        }
    }
    // Initialize signature counters to neutral (unknown)
    uint8_t neutral = 4; // middle of unknown band [3..4]
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid)
            return w;
    }

    // SRRIP victim search: prefer lines at RRPV_MAX; otherwise age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == RRPV_MAX)
                return w;
        }
        // Age all ways (saturating)
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
    uint32_t type,
    uint8_t hit
) {
    if (hit) {
        // Hit: promote to MRU and reward signature once (first hit only)
        rrpv[set][way] = 0;
        if (reused[set][way] == 0) {
            uint32_t sig = line_sig[set][way];
            if (sig_ctrs[sig] > 0) sig_ctrs[sig]--;
            reused[set][way] = 1;
        }
        return;
    }

    // Miss/Fill path:
    // 1) Train on the evicted line if it never hit
    if (reused[set][way] == 0) {
        uint32_t old_sig = line_sig[set][way];
        if (sig_ctrs[old_sig] < SIG_CTR_MAX) sig_ctrs[old_sig]++;
    }

    // 2) Compute new line's signature and choose insertion RRPV
    uint32_t sig = pc_signature(PC, set);
    line_sig[set][way] = static_cast<uint16_t>(sig);
    reused[set][way] = 0; // wait for first hit to reward

    uint8_t ctr = sig_ctrs[sig];
    uint8_t ins_rrpv;
    if (ctr >= BAD_THRESHOLD)       ins_rrpv = RRPV_MAX; // bad
    else if (ctr <= GOOD_THRESHOLD) ins_rrpv = 1;        // good
    else                             ins_rrpv = 2;        // unknown

    // Type-aware tweaks
    if (type == TYPE_WRITEBACK) {
        ins_rrpv = RRPV_MAX; // never protect writebacks
    } else if (type == TYPE_PREFETCH) {
        if (ins_rrpv < RRPV_MAX) ins_rrpv++; // shift one level deeper (clamped)
    }

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