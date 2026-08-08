#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
// LTP table size (PC-indexed, 2-bit counters)
static constexpr uint32_t LTP_IDX_BITS = 10;             // 1024-entry LTP
static constexpr uint32_t LTP_SIZE     = (1u << LTP_IDX_BITS);
static constexpr uint8_t  LTP_INIT     = 1;              // slightly cold start (0..3)
static constexpr uint8_t  LTP_FRIENDLY_TH = 2;           // >=2 => friendly (insert younger)

// StreamGuard++: per-PC ±1-line stride detector
static constexpr uint32_t STREAM_IDX_BITS = 10;          // 1024 entry stream table
static constexpr uint32_t STREAM_SIZE     = (1u << STREAM_IDX_BITS);
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;        // >=2 => near-bypass
static constexpr uint8_t  LINE_LOW_BITS   = 12;          // track low 12 bits of line number

// Per-line age (0=youngest..3=oldest)
static constexpr uint8_t  AGE_MAX = 3;

// --------------- Per-line state ---------------
// age: 2 bits (0..3)
static uint8_t  AGE[LLC_SETS][LLC_WAYS];
// hitcnt: 2 bits (0=none,1=one,2=two-or-more)
static uint8_t  HITCNT[LLC_SETS][LLC_WAYS];
// pcsig: 10-bit PC index stored in 16-bit container
static uint16_t PCSIG[LLC_SETS][LLC_WAYS];
// sentinel: 1-bit; 1 => no-train/no-promote (streams/prefetches/WB)
static uint8_t  SENTINEL[LLC_SETS][LLC_WAYS];

// --------------- Global predictors ---------------
static uint8_t LTP[LTP_SIZE]; // 2-bit saturating counters (0..3), PC-indexed

// StreamGuard++ tables (PC-indexed)
static uint16_t ST_LAST_LINE[STREAM_SIZE]; // low LINE_LOW_BITS of line number
static uint8_t  ST_CONF[STREAM_SIZE];      // 2-bit confidence (0..3)
static uint8_t  ST_LAST_ABS1[STREAM_SIZE]; // 1-bit: last step had |delta|==1

// ---------------- Helpers ----------------
static inline uint32_t ltp_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 19) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (LTP_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 21);
    return static_cast<uint32_t>(x) & (STREAM_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc2(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec2(uint8_t &x) { if (x > 0) x--; }
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// Update stream detector and return confidence after this access (demand only)
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_lowN(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    // compute small signed delta within tracked window
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(prev));
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc2(ST_CONF[idx], 3); // consecutive ±1 step confirms stream
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        sat_dec2(ST_CONF[idx]);       // decay on non-sequential step
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]      = AGE_MAX;
            HITCNT[s][w]   = 0;
            PCSIG[s][w]    = 0;
            SENTINEL[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < LTP_SIZE; i++) {
        LTP[i] = LTP_INIT;
    }
    for (uint32_t i = 0; i < STREAM_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
}

// Compare two candidates for eviction; return true if (a) is more evictable than (b)
static inline bool more_evictable(
    uint32_t set,
    uint32_t a, uint32_t b
) {
    // 1) Prefer evicting sentinels
    if (SENTINEL[set][a] != SENTINEL[set][b]) return SENTINEL[set][a] > SENTINEL[set][b];

    // 2) Prefer evicting PCs predicted dead (lower LTP value)
    uint8_t ltp_a = LTP[ PCSIG[set][a] & (LTP_SIZE - 1) ];
    uint8_t ltp_b = LTP[ PCSIG[set][b] & (LTP_SIZE - 1) ];
    if (ltp_a != ltp_b) return ltp_a < ltp_b;

    // 3) Prefer lines with fewer observed hits (0 < 1 < 2+)
    uint8_t hc_a = HITCNT[set][a];
    uint8_t hc_b = HITCNT[set][b];
    if (hc_a != hc_b) return hc_a < hc_b;

    // 4) Tie-break by age: older is more evictable
    return AGE[set][a] > AGE[set][b];
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Choose most evictable way by composite priority
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best; // replaced block index
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
    const bool demand = is_demand(type);

    // Update stream detector on all demand accesses (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        // Never promote or train sentinels (streams/prefetches/WB)
        if (SENTINEL[set][way]) return;

        // Multi-hit gating: only 2nd+ hit promotes and counts as "strong"
        if (demand) {
            if (HITCNT[set][way] == 0) {
                // First demand hit: train PC towards friendly, no promotion
                uint32_t idx = PCSIG[set][way] & (LTP_SIZE - 1);
                sat_inc2(LTP[idx], 3);
                HITCNT[set][way] = 1;
                return;
            } else {
                // Second or later demand hit: promote to youngest
                HITCNT[set][way] = 2;
                AGE[set][way] = 0;
                return;
            }
        }
        // Non-demand hits (shouldn't occur): keep stable
        return;
    }

    // MISS path: train on evicted occupant (dead-on-evict), excluding sentinels
    if (victim_addr != 0) {
        if (!SENTINEL[set][way]) {
            if (HITCNT[set][way] == 0) {
                uint32_t idx = PCSIG[set][way] & (LTP_SIZE - 1);
                sat_dec2(LTP[idx]);
            }
        }
    }

    // Insertion policy for the incoming line
    if (type == ACCESS_WRITEBACK) {
        // Writebacks: never bypass; tail insert, sentinel, no train/promote
        AGE[set][way]      = AGE_MAX;
        HITCNT[set][way]   = 0;
        PCSIG[set][way]    = 0;
        SENTINEL[set][way] = 1;
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetches: tail insert, sentinel, no train/promote
        AGE[set][way]      = AGE_MAX;
        HITCNT[set][way]   = 0;
        PCSIG[set][way]    = 0;
        SENTINEL[set][way] = 1;
        return;
    }

    // Demand insertion: apply StreamGuard++ near-bypass
    if (stream_conf >= STREAM_CONF_BYPASS) {
        // Stream: tail insert + sentinel to forbid promotion/training
        AGE[set][way]      = AGE_MAX;
        HITCNT[set][way]   = 0;
        PCSIG[set][way]    = 0;
        SENTINEL[set][way] = 1;
        return;
    }

    // Normal demand: consult LTP friendliness
    uint32_t sig = ltp_index(PC);
    uint8_t ltp_val = LTP[sig];

    SENTINEL[set][way] = 0;     // trainable line
    PCSIG[set][way]    = static_cast<uint16_t>(sig);
    HITCNT[set][way]   = 0;

    if (ltp_val >= LTP_FRIENDLY_TH) {
        // Friendly PCs: insert young (but not absolute MRU)
        AGE[set][way] = 1;
    } else {
        // Unfriendly PCs: insert old to self-eliminate unless proven otherwise
        AGE[set][way] = AGE_MAX;
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}