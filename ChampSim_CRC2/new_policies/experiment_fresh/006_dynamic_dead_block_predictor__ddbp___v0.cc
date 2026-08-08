#include <cstdint>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define PRT_SIZE 8192
#define ADT_SIZE 4096
#define RSD_SIZE 1024

uint32_t rrpv[LLC_SETS][LLC_WAYS];
uint32_t dead_candidate[LLC_SETS][LLC_WAYS];
bool touched[LLC_SETS / 4][LLC_WAYS];
uint16_t pc_sig[LLC_SETS / 4][LLC_WAYS];
uint8_t PRT[PRT_SIZE], ADT[ADT_SIZE], RSD[RSD_SIZE];

void InitReplacementState() {
    std::fill_n(&PRT[0], PRT_SIZE, 4);
    std::fill_n(&ADT[0], ADT_SIZE, 1);
    std::fill_n(&RSD[0], RSD_SIZE, 1);
    for (int i = 0; i < LLC_SETS; ++i) {
        for (int j = 0; j < LLC_WAYS; ++j) {
            rrpv[i][j] = 3;
            dead_candidate[i][j] = false;
            if (i % 4 == 0) {
                touched[i >> 2][j] = false;
                pc_sig[i >> 2][j] = 0;
            }
        }
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    for (uint32_t way = 0; way < LLC_WAYS; ++way) {
        if (!current_set[way].valid) return way;
    }

    while (true) {
        for (uint32_t way = 0; way < LLC_WAYS; ++way) {
            if (rrpv[set][way] == 3 && dead_candidate[set][way]) {
                return way;
            }
        }

        for (uint32_t way = 0; way < LLC_WAYS; ++way) {
            if (rrpv[set][way] == 3) {
                return way;
            } else {
                rrpv[set][way]++;
            }
        }
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    uint32_t pc_sig_index = PC % PRT_SIZE;
    uint32_t addr_sig_index = (paddr >> LOG2_BLOCK_SIZE) % ADT_SIZE;
    uint32_t region_sig_index = (paddr >> LOG2_PAGE_SIZE) % RSD_SIZE;

    if (hit) {
        if (set % 4 == 0) touched[set >> 2][way] = true;
        if (dead_candidate[set][way]) {
            dead_candidate[set][way] = false;
            rrpv[set][way] = std::min(rrpv[set][way], 1u);
        } else {
            rrpv[set][way] = 0;
        }
    } else {
        bool is_sampled = (set % 4 == 0);
        if (is_sampled) {
            pc_sig[set >> 2][way] = pc_sig_index;
            touched[set >> 2][way] = false;
        }

        bool pc_dead_low = PRT[pc_sig_index] <= 2;
        bool addr_dead_low = ADT[addr_sig_index] >= 2;
        bool region_stream_low = RSD[region_sig_index] >= 2;
        bool is_prefetch = (type == PREFETCH);

        bool predicted_dead = region_stream_low || addr_dead_low || pc_dead_low || (is_prefetch && (pc_dead_low || addr_dead_low || region_stream_low));

        if (predicted_dead) {
            rrpv[set][way] = 3;
            dead_candidate[set][way] = true;
        } else if (PRT[pc_sig_index] >= 6) {
            rrpv[set][way] = 0;
        } else {
            rrpv[set][way] = 2;
        }
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}