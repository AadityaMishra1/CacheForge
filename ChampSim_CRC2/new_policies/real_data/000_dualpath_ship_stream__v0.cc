#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// SRRIP parameters
static uint8_t RRPV[LLC_SETS][LLC_WAYS];      // 2-bit (stored in 8-bit)
// SHiP: per-line signature (10-bit stored in 16-bit)
static uint16_t LINE_SIG[LLC_SETS][LLC_WAYS];
// Per-line capped hit count (0..3) to enforce multi-hit promotion
static uint8_t LINE_HITS[LLC_SETS][LLC_WAYS]; // 2-bit (stored in 8-bit)
// Per-line stream-mark: never promote streaming lines
static uint8_t LINE_STREAM[LLC_SETS][LLC_WAYS]; // 1-bit (stored in 8-bit)

// SHCT: PC-signature counters (3-bit saturating: 0..7)
static constexpr uint32_t SHIP_SIG_BITS = 10;
static constexpr uint32_t SHCT_SIZE = (1u << SHIP_SIG_BITS);
static uint8_t SHCT[SHCT_SIZE];

// Lightweight per-PC stream detector (delta==+1 line, i.e., 64B stride)
// Confident after two consecutive +1 steps (conf>=2)
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE];   // low 16 bits of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];        // 2-bit 0..3
static uint8_t  ST_LAST_PLUS1[STREAM_TABLE_SIZE];  // 0/1

// Type tags (best-effort; Champsim CRC2 commonly uses these)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline uint32_t ship_index(uint64_t pc) {
    // 10-bit signature from PC
    uint64_t x = pc ^ (pc >> 10) ^ (pc >> 20);
    return static_cast<uint32_t>(x) & (SHCT_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    // cache line number low16 (64B lines)
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu);
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline bool pc_is_stream_confident(uint64_t pc) {
    return ST_CONF[stream_index(pc)] >= 2;
}
static inline void update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    bool plus1 = (uint16_t)(ST_LAST_LINE[idx] + 1u) == curr;

    if (plus1) {
        if (ST_LAST_PLUS1[idx]) sat_inc(ST_CONF[idx], 3);
        ST_LAST_PLUS1[idx] = 1;
    } else {
        // decay confidence on break of stride
        sat_dec(ST_CONF[idx]);
        ST_LAST_PLUS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w] = 3;            // SRRIP: start as distant re-ref
            LINE_SIG[s][w] = 0;
            LINE_HITS[s][w] = 0;
            LINE_STREAM[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        SHCT[i] = 3; // neutral/weak-dead start
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_PLUS1[i] = 0;
    }
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
    // Return an invalid way immediately if available
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP victim selection: look for RRPV==3; if none, age and retry
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) return w;
        }
        // Age all ways (saturate at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    // Fallback (should be unreachable)
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
    // Snapshot predictors before we update them with this access
    const uint32_t sig = ship_index(PC);
    const bool stream_conf = pc_is_stream_confident(PC);

    if (hit) {
        // On hit: streaming lines never promote; others use multi-hit promotion
        if (LINE_STREAM[set][way]) {
            // Keep them low-priority; do not promote
            if (LINE_HITS[set][way] < 3) LINE_HITS[set][way]++;
        } else {
            // First re-reference trains SHCT as 'live'
            if (LINE_HITS[set][way] == 0) sat_inc(SHCT[LINE_SIG[set][way]], 7);
            if (LINE_HITS[set][way] < 3) LINE_HITS[set][way]++;

            // Multi-hit guarded promotion
            if (LINE_HITS[set][way] >= 2) {
                RRPV[set][way] = 0; // strong promotion
            } else if (RRPV[set][way] > 0) {
                RRPV[set][way]--;   // mild promotion
            }
        }
    } else {
        // Miss: install new line and train on evicted one if any
        // Train SHCT on the evicted occupant if the way previously held a valid line
        if (victim_addr != 0) {
            // For streaming-marked lines: only penalize if they were never hit
            if (LINE_STREAM[set][way]) {
                if (LINE_HITS[set][way] == 0) sat_dec(SHCT[LINE_SIG[set][way]]);
            } else {
                // For normal lines: dead-on-fill (no hits) -> decrement
                if (LINE_HITS[set][way] == 0) sat_dec(SHCT[LINE_SIG[set][way]]);
                // If it had at least one hit, we already incremented on the first hit
            }
        }

        // Install new metadata
        LINE_SIG[set][way] = sig;
        LINE_HITS[set][way] = 0;
        LINE_STREAM[set][way] = (type != ACCESS_WRITEBACK && stream_conf) ? 1 : 0;

        // Insertion policy
        uint8_t ins_rrpv = 3;
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = 3; // keep prefetch low priority
        } else if (LINE_STREAM[set][way]) {
            ins_rrpv = 3; // aggressive low priority for streams
        } else {
            // SHiP-guided insertion
            ins_rrpv = (SHCT[sig] >= 4) ? 2 : 3;
        }
        RRPV[set][way] = ins_rrpv;
    }

    // Update PC stream detector with this access
    update_pc_stream(PC, paddr);
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}