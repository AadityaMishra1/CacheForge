#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

using std::min;
using std::max;

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define PSP_ENTRIES 1024
#define SHCT_ENTRIES 4096
#define PSEL_INIT 2048
#define RUNLEN_MAX 7
#define STREAM_CONF_MAX 7
#define STREAM_THRESHOLD 4
#define HIGH_REUSE_THRESHOLD 3

struct PSTEntry {
    uint16_t last_block;
    int8_t last_delta;
    uint8_t stream_conf;
    uint8_t run_len;
    bool valid;
};

uint8_t rrpv[LLC_SETS][LLC_WAYS];
uint8_t mt[LLC_SETS][LLC_WAYS];
uint16_t psig[LLC_SETS][LLC_WAYS];
bool uf[LLC_SETS][LLC_WAYS];

uint16_t shct[SHCT_ENTRIES];
PSTEntry pst[PSP_ENTRIES];
int32_t psel = PSEL_INIT;

void InitReplacementState() {
    std::memset(rrpv, 3, sizeof(rrpv));
    std::memset(mt, 0, sizeof(mt));
    std::memset(psig, 0, sizeof(psig));
    std::memset(uf, 0, sizeof(uf));
    std::memset(shct, 0, sizeof(shct));
    std::memset(pst, 0, sizeof(pst));
    psel = PSEL_INIT;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    for(uint32_t way = 0; way < LLC_WAYS; way++) {
        if(!current_set[way].valid) return way;
    }
    while (true) {
        for(uint32_t way = 0; way < LLC_WAYS; way++) {
            if(rrpv[set][way] == 3) return way;
        }
        for(uint32_t way = 0; way < LLC_WAYS; way++) {
            rrpv[set][way]++;
        }
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    uint16_t sig = PC % SHCT_ENTRIES;
    PSTEntry& entry = pst[PC % PSP_ENTRIES];

    // Stream prediction
    uint16_t block = paddr >> 6;
    if(entry.valid) {
        int delta = block - entry.last_block;
        entry.run_len = (abs(delta) == 1) ? min(entry.run_len + 1, RUNLEN_MAX) : 0;
        entry.stream_conf = (entry.run_len >= STREAM_THRESHOLD) ? min(entry.stream_conf + 1, STREAM_CONF_MAX) : entry.stream_conf;
    } else {
        entry.valid = true;
    }
    entry.last_block = block;

    bool is_stream = entry.stream_conf >= STREAM_THRESHOLD;

    // Replacement decisions
    if (hit) {
        if (!mt[set][way]) {
            mt[set][way] = 1;  // Mark as multi-touch on first hit
            rrpv[set][way] = max(rrpv[set][way] - 1, 0U);
        } else {
            rrpv[set][way] = max(rrpv[set][way] - 2, 0U);
        }
        entry.stream_conf = max(entry.stream_conf - 1, 0);  // Decrease on-hit for PC
    } else {
        int rrpv_val = (is_stream ? 3 : shct[sig] >= HIGH_REUSE_THRESHOLD ? 1 : (psel >= PSEL_INIT/2 ? 3 : 0));
        rrpv[set][way] = rrpv_val;
        mt[set][way] = 0;
        if (is_stream) {
            if (type == PREFETCH || psel >= PSEL_INIT/2) {
                return;  // Bypass decision
            }
        }

        if (shct[sig] < HIGH_REUSE_THRESHOLD || !is_stream) {
            psig[set][way] = sig;
            uf[set][way] = (shct[sig] < HIGH_REUSE_THRESHOLD);
        }
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}