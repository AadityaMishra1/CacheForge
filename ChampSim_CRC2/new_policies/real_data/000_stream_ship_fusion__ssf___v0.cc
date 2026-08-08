#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---- SSF parameters ----
static constexpr uint32_t RRPV_MAX = 3;         // SRRIP-2b
static constexpr uint32_t SHCT_ENTRIES = 1024;  // PC-signature table size (10-bit index)
static constexpr uint32_t SHCT_INIT = 4;        // neutral initial confidence (3-bit counter)
static constexpr uint32_t SHCT_MAX = 7;

static constexpr uint32_t STREAM_ENTRIES = 1024; // per-PC stream detector entries (10-bit index)
static constexpr uint32_t STREAM_TAG_BITS = 12;  // small tag to reduce aliasing
static constexpr uint32_t STREAM_CONF_MAX = 3;   // saturating confidence
static constexpr uint32_t STREAM_TH = 2;         // need >=2 sequential steps to classify stream

// Access type hints (best-effort; values match common ChampSim encodings)
static constexpr uint32_t ACCESS_LOAD = 0;
static constexpr uint32_t ACCESS_RFO = 1;
static constexpr uint32_t ACCESS_PREFETCH = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---- Per-line metadata ----
static uint8_t  line_rrpv[LLC_SETS][LLC_WAYS];     // 2b effective (stored in 8b)
static uint16_t line_sig[LLC_SETS][LLC_WAYS];      // 10b signature (PC index)
static uint8_t  line_reuse[LLC_SETS][LLC_WAYS];    // 1b: seen at least one demand hit
static uint8_t  line_valid_meta[LLC_SETS][LLC_WAYS]; // 1b: slot previously held a valid line (for SHiP eviction update)

// ---- SHiP predictor ----
static uint8_t SHCT[SHCT_ENTRIES]; // 3b saturating counters

// ---- Per-PC stream detector ----
// Index by low 10 bits of PC; tag by next 12 bits; last_block keeps low 16 bits of line addr
static uint16_t stream_tag[STREAM_ENTRIES];     // 12b effective
static uint16_t stream_last_block[STREAM_ENTRIES]; // 16b of line address
static uint8_t  stream_conf[STREAM_ENTRIES];    // 2b confidence

// Helpers
static inline uint32_t ship_index(uint64_t PC) {
    return static_cast<uint32_t>(PC) & (SHCT_ENTRIES - 1);
}
static inline uint32_t stream_index(uint64_t PC) {
    return static_cast<uint32_t>(PC) & (STREAM_ENTRIES - 1);
}
static inline uint16_t stream_pc_tag(uint64_t PC) {
    return static_cast<uint16_t>((PC >> 10) & ((1u << STREAM_TAG_BITS) - 1));
}

// Update per-PC stream state and return whether current access is in a confident forward stream
static inline bool update_and_is_stream(uint64_t PC, uint64_t paddr_line) {
    uint32_t idx = stream_index(PC);
    uint16_t tag = stream_pc_tag(PC);

    if (stream_tag[idx] != tag) {
        // New PC region
        stream_tag[idx] = tag;
        stream_conf[idx] = 0;
        stream_last_block[idx] = static_cast<uint16_t>(paddr_line & 0xFFFFu);
        return false;
    }

    uint16_t last = stream_last_block[idx];
    uint16_t cur  = static_cast<uint16_t>(paddr_line & 0xFFFFu);

    // Detect +1 line stride (sequential forward)
    if (static_cast<uint16_t>(last + 1) == cur) {
        if (stream_conf[idx] < STREAM_CONF_MAX) stream_conf[idx]++;
    } else if (cur == last) {
        // same line access: keep confidence
    } else {
        stream_conf[idx] = 0;
    }

    stream_last_block[idx] = cur;
    return (stream_conf[idx] >= STREAM_TH);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            line_rrpv[s][w] = RRPV_MAX;
            line_sig[s][w] = 0;
            line_reuse[s][w] = 0;
            line_valid_meta[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < SHCT_ENTRIES; i++) {
        SHCT[i] = SHCT_INIT;
    }
    for (uint32_t i = 0; i < STREAM_ENTRIES; i++) {
        stream_tag[i] = 0;
        stream_last_block[i] = 0;
        stream_conf[i] = 0;
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
        if (!current_set[w].valid) return w;
    }

    // SRRIP: search for RRPV==MAX; if none, age and repeat
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (line_rrpv[set][w] >= RRPV_MAX) {
                return w;
            }
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (line_rrpv[set][w] < RRPV_MAX) line_rrpv[set][w]++;
        }
    }
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
    // Line address (64B blocks)
    uint64_t line_addr = (paddr >> 6);

    // Streaming classification (updates internal state)
    bool is_stream = update_and_is_stream(PC, line_addr);

    if (hit) {
        // Multi-hit gating: first demand hit only marks reuse; further hits promote.
        // Never promote lines accessed by streaming PCs.
        if (line_reuse[set][way] == 0) {
            line_reuse[set][way] = 1; // first reuse observed
            // do not promote on first hit
        } else {
            if (!is_stream) {
                line_rrpv[set][way] = 0; // MRU
            }
        }
        return;
    }

    // Miss fill path:
    // 1) Update SHiP for the evicted line if this slot was previously valid
    if (line_valid_meta[set][way]) {
        uint16_t old_sig = line_sig[set][way] & (SHCT_ENTRIES - 1);
        if (line_reuse[set][way]) {
            if (SHCT[old_sig] < SHCT_MAX) SHCT[old_sig]++;
        } else {
            if (SHCT[old_sig] > 0) SHCT[old_sig]--;
        }
    }

    // 2) Decide insertion priority for the incoming line
    uint32_t sig = ship_index(PC);
    bool predicted_dead = (SHCT[sig] <= 1); // cold PCs bypass to tail
    bool is_prefetch = (type == ACCESS_PREFETCH);

    uint8_t ins_rrpv;
    if (is_stream || predicted_dead || is_prefetch) {
        ins_rrpv = RRPV_MAX;   // lowest priority: approximate bypass
    } else {
        ins_rrpv = 1;          // conservative insertion for predicted-hot
    }

    // 3) Install new metadata
    line_rrpv[set][way] = ins_rrpv;
    line_sig[set][way] = static_cast<uint16_t>(sig);
    line_reuse[set][way] = 0;         // not yet reused
    line_valid_meta[set][way] = 1;    // slot now holds valid line
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}