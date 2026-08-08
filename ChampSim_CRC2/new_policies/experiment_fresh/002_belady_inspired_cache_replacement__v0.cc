#include <algorithm>
#include <cstring>
#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SAMPLE_RATE 16
#define THR_P 3
#define THR_S 2
#define PCIDX_MASK 0xFFF
#define MAX_PFPRED 7
#define MAX_STREAM 3

using std::min;
using std::max;

struct LineMetadata {
    uint8_t pred_friendly;
    uint8_t was_hit;
    uint8_t pf;
    uint8_t age;
};

LineMetadata line_metadata[LLC_SETS][LLC_WAYS];
uint8_t PCPred[4096];
uint8_t PCStream[4096];
uint16_t signature[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

bool IsSampledSet(uint32_t set) {
    return (set % SAMPLE_RATE == 0);
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) {
            return way;
        }
    }

    uint32_t victim = 0;
    int max_age = -1;
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        LineMetadata &meta = line_metadata[set][way];
        if (meta.pred_friendly == 0) {
            if (meta.age > max_age) {
                victim = way;
                max_age = meta.age;
            } else if (meta.age == max_age && meta.pf == 1 && meta.was_hit == 0) {
                victim = way;
            }
        }
    }

    if (max_age != -1) return victim;

    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        LineMetadata &meta = line_metadata[set][way];
        if (meta.pred_friendly == 1 && meta.was_hit == 0) {
            if (meta.age > max_age) {
                victim = way;
                max_age = meta.age;
            }
        }
    }

    if (max_age != -1) return victim;

    max_age = -1;
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        LineMetadata &meta = line_metadata[set][way];
        if (meta.age > max_age) {
            victim = way;
            max_age = meta.age;
        }
    }

    return victim;
}

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
    uint32_t idx = PC & PCIDX_MASK;
    auto &meta = line_metadata[set][way];
    
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        auto &m = line_metadata[set][w];
        if (w == way) {
            m.age = 0;
        } else {
            m.age = min(m.age + 1, 3);
        }
    }

    if (hit) {
        meta.was_hit = 1;
        if (meta.pf == 1) {
            meta.pred_friendly = 1;
            PCStream[idx] = max(PCStream[idx] - 1, 0);
        }
        return;
    }

    if (IsSampledSet(set)) {
        uint16_t evicted_sig = signature[set / SAMPLE_RATE][way];
        if (meta.was_hit == 0) {
            PCPred[evicted_sig] = max(PCPred[evicted_sig] - 1, 0);
            PCStream[evicted_sig] = min(PCStream[evicted_sig] + 1, MAX_STREAM);
        } else {
            PCPred[evicted_sig] = min(PCPred[evicted_sig] + 1, MAX_PFPRED);
            PCStream[evicted_sig] = max(PCStream[evicted_sig] - 1, 0);
        }
    }

    bool predicted_averse = (PCPred[idx] <= THR_P) || (PCStream[idx] >= THR_S);
    meta.pf = (type == PREFETCH);
    meta.was_hit = 0;

    if (predicted_averse && meta.pf == 1) {
        meta.pred_friendly = 0;
        meta.age = 3;
    } else {
        meta.pred_friendly = predicted_averse ? 0 : 1;
        meta.age = meta.pred_friendly ? 2 : 3;
    }

    if (IsSampledSet(set)) {
        signature[set / SAMPLE_RATE][way] = idx;
    }
}

void InitReplacementState() {
    memset(line_metadata, 0, sizeof(line_metadata));
    std::fill_n(PCPred, 4096, 4);
    std::fill_n(PCStream, 4096, 1);
}

// Periodically print heartbeat (can be empty)
void PrintStats_Heartbeat() {}

// Print end-of-simulation statistics (can be empty)
void PrintStats() {}