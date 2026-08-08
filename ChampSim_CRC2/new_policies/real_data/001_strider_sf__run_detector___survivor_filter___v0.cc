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
// PC tables (compact to fit metadata budget)
static constexpr uint8_t  PC_BITS   = 8;                 // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE   = (1u << PC_BITS);

// Run detector: ±1/±2 stride; >=2 => stream-guard near-bypass
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;        // confidence threshold (0..3)
static constexpr uint8_t  LINE_LOW_BITS      = 8;        // low bits of line number to compare

// Survivor Filter (PC predictor)
static constexpr uint8_t  SF_WARM_TH = 2;                // >=2 => predicted warm
// Insertion depths
static constexpr uint8_t  RRPV_DISTANT = 3;              // tail
static constexpr uint8_t  RRPV_WARM    = 1;              // younger-than-tail
static constexpr uint8_t  RRPV_MRU     = 0;              // MRU on confirmed multi-hit

// ---------------- Per-line state ----------------
// 2-bit age (SRRIP-like RRPV)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];        // 0..3
// 2-bit demand hit counter (0,1,2+)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];      // 0..3 (we use up to 3 as 2+)
// 2-bit status: 0=normal, 1=stream-guard(no promote/train), 2=quarantine(prefetch/WB)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];      // 0..2
// 8-bit PC signature for Survivor Filter training
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];       // 0..255

// ---------------- Global predictors ----------------
// Survivor Filter: 2-bit counters per PC (0..3)
static uint8_t PC_CT[PC_SIZE];

// Run detector tables per PC
static uint8_t ST_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t ST_CONF[PC_SIZE];        // 2-bit confidence (0..3)
static uint8_t ST_LAST_ABS12[PC_SIZE];  // 1-bit: previous step was |delta|==1 or 2

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix then index
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 17);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint8_t line_lowN(uint64_t paddr) {
    return static_cast<uint8_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
// Update run-detector and return new confidence (demand only)
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint8_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(ST_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);

    if (abs12) {
        if (ST_LAST_ABS12[idx]) {
            sat_inc(ST_CONF[idx], 3);   // consecutive ±1/±2 confirms a run
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS12[idx] = 1;
        }
    } else {
        sat_dec(ST_CONF[idx]);          // decay on break
        ST_LAST_ABS12[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]   = RRPV_DISTANT;
            HITCNT[s][w] = 0;
            STATUS[s][w] = 0;
            PCSIG[s][w]  = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_CT[i]        = 1; // slightly cold start
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS12[i]= 0;
    }
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting stream/prefetch sentinels
    bool a_sentinel = (STATUS[set][a] != 0);
    bool b_sentinel = (STATUS[set][b] != 0);
    if (a_sentinel != b_sentinel) return a_sentinel;

    // 2) Prefer PCs predicted dead (lower Survivor Filter value)
    uint8_t sf_a = PC_CT[ PCSIG[set][a] & (PC_SIZE - 1) ];
    uint8_t sf_b = PC_CT[ PCSIG[set][b] & (PC_SIZE - 1) ];
    if (sf_a != sf_b) return sf_a < sf_b;

    // 3) Prefer fewer observed hits (0 < 1 < 2+)
    uint8_t hc_a = HITCNT[set][a];
    uint8_t hc_b = HITCNT[set][b];
    if (hc_a != hc_b) return hc_a < hc_b;

    // 4) Tie-break by age (older is more evictable)
    return RRPV[set][a] > RRPV[set][b];
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Composite priority victim selection (no unbounded loops)
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);

    // Update run detector on every demand access (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    if (hit) {
        // Never train or promote sentinels (stream-guard or prefetch/WB)
        if (STATUS[set][way] != 0) {
            return;
        }

        if (demand) {
            uint8_t prev = HITCNT[set][way];
            if (prev < 3) HITCNT[set][way] = static_cast<uint8_t>(prev + 1);

            // Second and later demand hits: confirm reuse and promote hard
            if (prev >= 1) {
                // Credit the filling PC in Survivor Filter on the second hit
                if (prev == 1) {
                    uint8_t pcs = PCSIG[set][way] & (PC_SIZE - 1);
                    sat_inc(PC_CT[pcs], 3);
                }
                RRPV[set][way] = RRPV_MRU;
            }
            // First demand hit: record only; no promotion
        }
        return;
    }

    // Miss/Fill path: train the evicted line (old contents) before overwrite
    {
        uint8_t old_status = STATUS[set][way];
        uint8_t old_hits   = HITCNT[set][way];
        uint8_t old_pcs    = PCSIG[set][way] & (PC_SIZE - 1);

        // Train only for normal lines (ignore stream-guard/prefetch/WB sentinels)
        if (old_status == 0) {
            if (old_hits == 0) {
                // Dead-on-evict => decrement Survivor Filter
                sat_dec(PC_CT[old_pcs]);
            }
            // For old_hits >= 2, the Survivor Filter was already credited on second hit
        }
    }

    // Insert new line
    uint32_t pcs = pc_index(PC);
    PCSIG[set][way]  = static_cast<uint8_t>(pcs);
    HITCNT[set][way] = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass writebacks; quarantine at tail, no train/promote
        STATUS[set][way] = 2;
        RRPV[set][way]   = RRPV_DISTANT;
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine at tail, no train/promote
        STATUS[set][way] = 2;
        RRPV[set][way]   = RRPV_DISTANT;
        return;
    }

    // Demand fill (LOAD/RFO)
    if (stream_conf >= STREAM_CONF_BYPASS) {
        // Stream-guard near-bypass: insert cold, never promote/train
        STATUS[set][way] = 1;
        RRPV[set][way]   = RRPV_DISTANT;
        return;
    }

    // Non-stream demand: use Survivor Filter for insertion depth
    STATUS[set][way] = 0;
    if (PC_CT[pcs] >= SF_WARM_TH) {
        RRPV[set][way] = RRPV_WARM;     // predicted warm
    } else {
        RRPV[set][way] = RRPV_DISTANT;  // predicted cold
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