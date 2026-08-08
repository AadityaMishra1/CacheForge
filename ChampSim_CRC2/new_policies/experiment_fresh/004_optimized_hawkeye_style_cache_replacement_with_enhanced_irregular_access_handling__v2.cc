#include <algorithm>
#include <cstring>
#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SAMPLE_RATE 4  // 25% sampled sets
#define PC_PRED_SIZE 8192
#define FS_THRESH 5
#define AVERSE_THRESH 2
#define STABILITY_THRESH 2
#define STREAM_K 4

using std::min;
using std::max;

struct LineMetadata {
    uint8_t RRPV : 2;
    uint8_t friend_flag : 1;
    uint8_t hit_seen : 1;
    uint8_t prefetch_bit : 1;
};

LineMetadata line_metadata[LLC_SETS][LLC_WAYS];
uint8_t PC_PRED[PC_PRED_SIZE][3];  // FS, ST, BP in 6 bits
uint16_t RDT[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

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

    while (true) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (line_metadata[set][way].RRPV == 3) {
                return way;
            }
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            line_metadata[set][way].RRPV = min(3, line_metadata[set][way].RRPV + 1);
        }
    }
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
    uint32_t idx = PC % PC_PRED_SIZE;
    auto &meta = line_metadata[set][way];
    uint8_t &FS = PC_PRED[idx][0], &ST = PC_PRED[idx][1], &BP = PC_PRED[idx][2];

    if (hit) {
        if (meta.prefetch_bit && type != PREFETCH) {
            meta.prefetch_bit = 0;
            if (FS >= FS_THRESH) {
                if (meta.hit_seen == 0) {
                    meta.hit_seen = 1;
                    meta.RRPV = max(0, meta.RRPV - 1);
                } else {
                    if (FS >= FS_THRESH) {
                        meta.RRPV = 0;
                    }
                }
            }
        }
    } else {
        if (IsSampledSet(set)) {
            uint16_t sig = PC & ((1 << 15) - 1);
            int pos = -1;
            for (int i = 0; i < LLC_WAYS; ++i) {
                if (RDT[set / SAMPLE_RATE][i] == sig) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) {
                pos = LLC_WAYS - 1;
                std::rotate(RDT[set / SAMPLE_RATE], RDT[set / SAMPLE_RATE] + 1, RDT[set / SAMPLE_RATE] + LLC_WAYS);
            }
            RDT[set / SAMPLE_RATE][0] = sig;

            if (pos < AVERSE_THRESH) {  // FRIENDLY
                FS = min(7, FS + 1);
                if (FS >= FS_THRESH) {
                    ST = min(3, ST + 1);
                }
                BP = 0;
            } else {  // AVERSE
                FS = max(0, FS - 1);
                if (ST >= STABILITY_THRESH && (++PC_PRED[idx][2]) >= STREAM_K) {
                    BP = 1;
                }
            }
        }

        uint8_t pred = (FS >= FS_THRESH) && (ST >= 1) ? 1 : 0;
        meta.prefetch_bit = (type == PREFETCH);
        meta.friend_flag = pred;
        meta.RRPV = pred ? 1 : 3;
        meta.hit_seen = 0;
        if (meta.prefetch_bit) {
            meta.RRPV = min(3, meta.RRPV + 1);
        }
    }
}

void InitReplacementState() {
    memset(line_metadata, 0, sizeof(line_metadata));
    memset(PC_PRED, 0, sizeof(PC_PRED));
    memset(RDT, -1, sizeof(RDT));  // Use -1 as uninitialized indicator
}

void PrintStats_Heartbeat() {}

void PrintStats() {}