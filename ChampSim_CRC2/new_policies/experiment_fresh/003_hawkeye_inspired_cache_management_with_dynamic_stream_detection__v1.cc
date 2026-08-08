#include <algorithm>
#include <cstring>
#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SAMPLE_RATE 16
#define PCIDX_MASK 0xFFF
#define MAX_SCORE 7
#define MAX_CONF 3

using std::min;
using std::max;

// Meta information per cache line
struct LineMetadata {
    uint8_t rrpv;
    uint8_t friendly;
    uint8_t second_touch;
};

// Metadata tables
LineMetadata line_metadata[LLC_SETS][LLC_WAYS];
struct PSTEntry {
    int8_t score;  // [-3, +4]
    uint8_t conf;  // [0 to 3]
    uint8_t stream;// Streaming flag
    uint8_t age_epoch;
} PST[4096];

uint8_t friend_count[LLC_SETS];

// Initialize replacement state
void InitReplacementState() {
    memset(line_metadata, 0, sizeof(line_metadata));
    memset(PST, 0, sizeof(PST));
    std::fill_n(friend_count, LLC_SETS, 0);
}

// Classify a PC into friendly or averse
inline int8_t classify_pc(uint64_t pc) {
    uint32_t idx = pc & PCIDX_MASK;
    if (PST[idx].stream && PST[idx].conf >= 2) return -1; // Strongly averse
    if (PST[idx].score >= 2 && PST[idx].conf >= 2) return 1; // Friendly
    return 0; // Uncertain
}

// Get victim in the set using RRIP with friend preference
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) return way;
    }

    uint32_t victim = 0;
    int max_rrpv = -1;
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (line_metadata[set][way].friendly == 0 && line_metadata[set][way].rrpv > max_rrpv) {
            victim = way;
            max_rrpv = line_metadata[set][way].rrpv;
        }
    }

    if (max_rrpv != 3) { // No victims with rrpv max, increment all
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            line_metadata[set][way].rrpv = min(line_metadata[set][way].rrpv + 1, 3U);
        }
    return way; // nested victim removed
    }
    return victim;
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
    uint32_t pc_idx = PC & PCIDX_MASK;
    auto &meta = line_metadata[set][way];

    if (hit) {
        if (meta.friendly) {
            meta.rrpv = max(meta.rrpv - 1, 0U); // Gentle promotion
        } else if (meta.second_touch) {
            meta.second_touch = 0;
            meta.friendly = 1;
            friend_count[set]++;
            meta.rrpv = max(meta.rrpv - 2, 0U);
        } else {
            meta.second_touch = 1;
            meta.rrpv = max(meta.rrpv - 1, 0U);
        }
        return;
    }

    int8_t classification = classify_pc(PC);
    int target_friends = min(max((LLC_WAYS * PST[pc_idx].score) / MAX_SCORE, 0), int(0.8 * LLC_WAYS));
    bool can_insert_friend = (friend_count[set] < target_friends);

    if (classification == -1) {
        if (current_set[way].valid) {
            meta.rrpv = 3;
            meta.friendly = 0;
        }
    } else if (classification == 1 && can_insert_friend) {
        meta.rrpv = 0;
        meta.friendly = 1;
        friend_count[set]++;
    } else {
        meta.rrpv = 2;
        meta.friendly = 0;
    }
    meta.second_touch = 0;
}

void PrintStats() {}

void PrintStats_Heartbeat() {}
