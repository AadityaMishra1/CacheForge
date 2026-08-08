#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define CACHE_LINE_SIZE 64ULL
#define DEAD_THRESH 2          // PC‑cold threshold
#define STREAM_RUN_LEN 2       // consecutive forward strides to mark stream

// ============================================================================
// GLOBAL STATE ARRAYS (static)
// ============================================================================
// RRIP: 2‑bit per line (store in 8‑bit)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// Run‑state: 0 = normal, 1 = one forward stride, 2 = STREAM
static uint8_t run_state[LLC_SETS][LLC_WAYS];

// PC‑cold predictor (1024 entries, 8‑bit counters)
static uint8_t pc_cold[1024];

// Per‑set run detector state
static uint64_t last_addr[LLC_SETS];
static uint64_t last_pc[LLC_SETS];
static uint8_t stride_cnt[LLC_SETS];   // values 0‑3, we use only up to 2

// Initialize replacement state
void InitReplacementState() {
    for (int s = 0; s < LLC_SETS; ++s) {
        for (int w = 0; w < LLC_WAYS; ++w) {
            rrpv[s][w] = 2;          // default insertion RRPV
            run_state[s][w] = 0;
        }
        last_addr[s] = 0;
        last_pc[s] = 0;
        stride_cnt[s] = 0;
    }
    for (int i = 0; i < 1024; ++i) pc_cold[i] = 0;
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
    // 1) Look for an empty way
    for (uint32_t w = 0; w < LLC_WAYS; ++w)
        if (!current_set[w].valid)
            return w;

    // 2) Prefer evicting STREAM lines with RRPV==3
    for (uint32_t w = 0; w < LLC_WAYS; ++w)
        if (run_state[set][w] == 2 && rrpv[set][w] == 3)
            return w;

    // 3) Normal RRIP victim search
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; ++w)
            if (rrpv[set][w] == 3 && run_state[set][w] != 2)
                return w;                     // non‑stream victim

        // Increment RRPV of all non‑stream lines (saturate at 3)
        for (uint32_t w = 0; w < LLC_WAYS; ++w)
            if (run_state[set][w] != 2 && rrpv[set][w] < 3)
                ++rrpv[set][w];
    }
    // Unreachable
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
    // ---------- PC‑cold predictor ----------
    uint32_t pc_idx = (PC >> 2) & 0x3FF;          // 1024‑entry table
    if (pc_cold[pc_idx] < 255) ++pc_cold[pc_idx];

    // ---------- Forward‑run detector ----------
    // Detect forward stride of exactly one cache line from the same PC
    if (PC == last_pc[set] && paddr == last_addr[set] + CACHE_LINE_SIZE) {
        if (stride_cnt[set] < STREAM_RUN_LEN) ++stride_cnt[set];
        if (stride_cnt[set] >= STREAM_RUN_LEN) {
            run_state[set][way] = 2;            // mark as STREAM
            rrpv[set][way] = 3;                 // lowest priority
        }
    } else {
        stride_cnt[set] = 1;                    // start a new potential run
    }
    last_addr[set] = paddr;
    last_pc[set] = PC;

    // ---------- RRIP update ----------
    if (hit) {
        // On hit promote to RRPV = 0 and clear stream flag
        rrpv[set][way] = 0;
        run_state[set][way] = 0;
    } else {
        // On miss (insertion)
        // Use PC‑cold info to decide initial RRPV
        if (pc_cold[pc_idx] < DEAD_THRESH) {
            rrpv[set][way] = 3;                 // dead line, evict quickly
        } else {
            rrpv[set][way] = 2;                 // normal insertion
        }
        // Reset run_state (may be set to STREAM later by detector)
        if (run_state[set][way] != 2)           // keep STREAM if already set
            run_state[set][way] = 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // kept blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // kept blank
}