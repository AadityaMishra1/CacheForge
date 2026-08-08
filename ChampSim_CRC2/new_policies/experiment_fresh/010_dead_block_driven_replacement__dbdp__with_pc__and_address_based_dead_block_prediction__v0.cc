#include <cstdint>
#include <algorithm>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define PST_SIZE 4096
#define DAT_SIZE 8192
#define BITS_PER_SET 2
#define BITS_PER_FLAG 1
#define RRPV_MAX 3
#define BITS_PER_SIG 12

using std::min;
using std::max;

// Predictor tables
uint8_t PST_demand[PST_SIZE];
uint8_t PST_pref[PST_SIZE];
uint8_t DAT[DAT_SIZE];

// Per-line metadata
uint8_t rrpv[LLC_SETS][LLC_WAYS];
uint8_t dead_flag[LLC_SETS][LLC_WAYS];
uint8_t hit_once[LLC_SETS][LLC_WAYS];

struct SampledMeta {
    uint8_t alive;
    uint16_t fill_sig;    // 12 bits
    uint8_t fill_is_pref; // 1 bit
};

SampledMeta sampled_meta[LLC_SETS / 4][LLC_WAYS];

// Bypass throttle
uint8_t bypass_throttle;

void InitReplacementState() {
    std::fill_n(&PST_demand[0], PST_SIZE, 0);
    std::fill_n(&PST_pref[0], PST_SIZE, 0);
    std::fill_n(&DAT[0], DAT_SIZE, 0);
    
    for (int i = 0; i < LLC_SETS; ++i) {
        for (int j = 0; j < LLC_WAYS; ++j) {
            rrpv[i][j] = RRPV_MAX;
            dead_flag[i][j] = 0;
            hit_once[i][j] = 0;
        }
    }
    
    bypass_throttle = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid)
            return way;
    }
    
    int victim_way = -1;
    while (victim_way == -1) {
        for (uint32_t way = 0; way < LLC_WAYS; ++way) {
            if (rrpv[set][way] == RRPV_MAX) {
                if (dead_flag[set][way]) return way;
                if (victim_way == -1) victim_way = way;
            }
        }
        
        if (victim_way == -1) {
            for (uint32_t way = 0; way < LLC_WAYS; ++way) {
                rrpv[set][way] = min(rrpv[set][way] + 1, (uint8_t)RRPV_MAX);
            }
        }
    }
    
    return victim_way;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    uint32_t sig = PC % PST_SIZE;
    uint32_t addr_hash = paddr % DAT_SIZE;
    
    if (hit) {
        rrpv[set][way] = max(rrpv[set][way] - 1, (uint8_t)0);
        if (hit_once[set][way] == 0) {
            hit_once[set][way] = 1;
        } else {
            dead_flag[set][way] = 0;
        }
        if (set % 4 == 0) {
            sampled_meta[set / 4][way].alive = 1;
        }
    } else {
        uint8_t *pc_ctr = type == PREFETCH ? &PST_pref[sig] : &PST_demand[sig];
        uint8_t addr_ctr = DAT[addr_hash];
        
        bool strong_dead = (*pc_ctr >= 6) || (addr_ctr >= 3);
        bool likely_dead = (*pc_ctr >= 4) || (addr_ctr == 2);
        bool friendly = (*pc_ctr <= 2) && (addr_ctr <= 1);
        
        bool can_bypass = strong_dead && (type == PREFETCH || bypass_throttle >= (1 << 6));
        if (can_bypass) return;

        rrpv[set][way] = strong_dead ? 3 : (likely_dead ? 2 : (friendly ? 1 : 2));
        dead_flag[set][way] = strong_dead || likely_dead;
        hit_once[set][way] = 0;

        if (set % 4 == 0) {
            sampled_meta[set / 4][way] = {0, static_cast<uint16_t>(sig), static_cast<uint8_t>(type == PREFETCH)};
        }

        bool outcome_dead = sampled_meta[set / 4][way].alive == 0;
        if (outcome_dead) (*pc_ctr) = min(*pc_ctr + 1, (uint8_t)7);
        else (*pc_ctr) = max(*pc_ctr - 1, (uint8_t)0);

        if (outcome_dead) DAT[addr_hash] = min(DAT[addr_hash] + 1, (uint8_t)3);
        else DAT[addr_hash] = max(DAT[addr_hash] - 1, (uint8_t)0);

        if (outcome_dead) bypass_throttle = min(bypass_throttle + 1, (uint8_t)255);
        else bypass_throttle = max(bypass_throttle - 1, (uint8_t)0);
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}