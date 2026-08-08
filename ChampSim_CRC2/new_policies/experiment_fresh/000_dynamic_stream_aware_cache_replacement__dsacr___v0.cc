#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SRC_SIZE 1024
#define MAX_RRPV 7

using std::min;
using std::max;

uint32_t rrpv[LLC_SETS][LLC_WAYS];
bool ft[LLC_SETS][LLC_WAYS / 4] = {0};
uint16_t sigID[LLC_SETS][LLC_WAYS / 4] = {0};
uint8_t nohit_streak[LLC_SETS] = {0};
uint16_t scan_region_sig[LLC_SETS] = {0};

struct SRCEntry {
    bool valid;
    uint16_t tag;
    uint8_t stream_score; // 2 bits
    uint8_t seq_conf; // 2 bits
    uint8_t direction; // 2 bits
    uint8_t last_offset; // 6 bits
};

SRCEntry SRC[SRC_SIZE];
uint64_t BOF_current[64] = {0};
uint64_t BOF_prev[64] = {0};
uint32_t bof_index = 0;

void InitReplacementState() {
    memset(rrpv, MAX_RRPV, sizeof(rrpv));
    for(auto& entry : SRC) {
        entry.valid = false;
    }
}

uint32_t (way /*nested victim removed*/uint32_t cpu, uint32_t set, const BLOCK* current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    for(uint32_t way = 0; way < LLC_WAYS; way++) {
        if(current_set[way].valid == false) {
            return way;
        }
    }

    while(true) {
        for(uint32_t way = 0; way < LLC_WAYS; way++) {
            if(rrpv[set][way] == MAX_RRPV) {
                return way;
            }
        }
        for(uint32_t way = 0; way < LLC_WAYS; way++) {
            rrpv[set][way]++;
        }
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    uint16_t region_id = paddr >> 12;
    uint8_t block_offset = (paddr >> 6) & 0x3F;
    uint16_t signature = (PC ^ region_id) & (SRC_SIZE - 1);
    SRCEntry &entry = SRC[signature];

    if(hit) {
        nohit_streak[set] = 0;
        rrpv[set][way] = 0;
        if(set % 4 == 0 && !ft[set][way]) {
            ft[set][way] = true;
        }
        return;
    } else {
        uint16_t sig_tag = PC ^ region_id;
        if(set % 4 == 0) {
            sigID[set][way] = signature;
            ft[set][way] = false;
        }

        if(!entry.valid || entry.tag != sig_tag) {
            entry.valid = true;
            entry.tag = sig_tag;
            entry.stream_score = 1;
            entry.seq_conf = 1;
            entry.last_offset = block_offset;
            entry.direction = 0;  // unknown
        } else {
            int delta = block_offset - entry.last_offset;
            entry.last_offset = block_offset;
            if(delta == 1) {
                entry.seq_conf = min(3, entry.seq_conf + 1);
                entry.direction = (entry.direction == 0 || entry.direction == 1) ? 1 : 0;
            } else if(delta == -1) {
                entry.seq_conf = min(3, entry.seq_conf + 1);
                entry.direction = (entry.direction == 0 || entry.direction == 2) ? 2 : 0;
            } else {
                entry.seq_conf = max(0, entry.seq_conf - 1);
                entry.direction = 0;
            }
            entry.stream_score = (ft[set][way]) ? max(0, entry.stream_score - 1) : min(3, entry.stream_score + 1);
        }
        
        if((entry.stream_score >= 2 || entry.seq_conf >= 2 || nohit_streak[set] >= 8) && !(BOF_current[bof_index] & (1ULL << (PC % 64)))) {
            if((entry.stream_score >= 3) || (entry.seq_conf >= 3)) {
                return;
            }
            rrpv[set][way] = MAX_RRPV - 1;
        } else {
            rrpv[set][way] = (entry.stream_score == 0) ? 1 : 2;
        }
        nohit_streak[set]++;
        BOF_current[bof_index] |= (1ULL << (PC % 64));
    }
}

void EpochMaintenance() {
    std::swap(BOF_current, BOF_prev);
    bof_index = (bof_index + 1) % 64;
    std::fill(std::begin(BOF_current), std::end(BOF_current), 0);
}

void PrintStats() {}

void PrintStats_Heartbeat() {}