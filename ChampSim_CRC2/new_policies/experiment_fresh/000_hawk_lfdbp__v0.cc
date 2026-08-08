#include <cstdint>
#include <algorithm>
#include <cstring>
#include "../inc/champsim_crc2.h"

using std::min;
using std::max;

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define MAX_RRPV 3

struct StreamerEntry {
    uint16_t last_line_idx;
    uint8_t run_len;
    uint8_t conf;
    bool dir;
};

uint32_t rrpv[LLC_SETS][LLC_WAYS];
uint8_t upt[512];
uint8_t dbp[512];
StreamerEntry streamer_table[256];

uint16_t hash9(uint64_t value) {
    return value & 0x1FF;
}

uint8_t hash16(uint64_t value) {
    return value & 0xFFFF;
}

void InitReplacementState() {
    std::fill(&rrpv[0][0], &rrpv[0][0] + LLC_SETS * LLC_WAYS, MAX_RRPV);
    std::fill(&upt[0], &upt[0] + 512, 0);
    std::fill(&dbp[0], &dbp[0] + 512, 0);

    for (auto &entry : streamer_table) {
        entry.last_line_idx = 0;
        entry.run_len = 0;
        entry.conf = 0;
        entry.dir = 0;
    }
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
        if (!current_set[way].valid) return way;
    }

    int32_t victim = -1;
    int32_t highest_rrpv = -1;

    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (rrpv[set][way] > highest_rrpv) {
            victim = way;
            highest_rrpv = rrpv[set][way];
        }
    }

    if (highest_rrpv < MAX_RRPV) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] < MAX_RRPV) {
                rrpv[set][way]++;
            }
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
    uint16_t pc_sig = hash9(PC ^ (paddr >> 12));
    uint8_t &dbp_counter = dbp[pc_sig];
    uint8_t &upt_counter = upt[pc_sig];

    if (hit) {
        if (rrpv[set][way] < MAX_RRPV) {
            rrpv[set][way] = max(rrpv[set][way] - 1, 0);
        }
        upt_counter = min(upt_counter + 1, 7u);
        dbp_counter = max(dbp_counter - 1, 0u);
    } else {
        int victim_rrpv = rrpv[set][way];
        if (upt_counter <= 3) {
            upt_counter = max(int(upt_counter) - 2, 0);
            dbp_counter = min(dbp_counter + 1, 3u);
        }
        
        if (streamer_table[pc_sig % 256].conf >= 2 && dbp_counter >= 2) {
            rrpv[set][way] = MAX_RRPV;
        } else if (upt_counter >= 4) {
            rrpv[set][way] = 0;
        } else {
            rrpv[set][way] = MAX_RRPV;
        }
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}
