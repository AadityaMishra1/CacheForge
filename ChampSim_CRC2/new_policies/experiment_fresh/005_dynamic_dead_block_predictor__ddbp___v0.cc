#include <algorithm>
#include <cstring>
#include <cstdint>
using std::min;
using std::max;

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define SHCT_SIZE 1024
#define RDT_SIZE 2048

uint32_t rrpv[LLC_SETS][LLC_WAYS];
uint8_t dead[LLC_SETS][LLC_WAYS];
uint8_t hit_once[LLC_SETS][LLC_WAYS];

uint16_t pc_signatures[LLC_SETS][LLC_WAYS];
uint8_t reuse_outcome[LLC_SETS][LLC_WAYS];
uint8_t prefetch_tag[LLC_SETS][LLC_WAYS];

uint8_t SHCT_demand[SHCT_SIZE];
uint8_t SHCT_pref[SHCT_SIZE];
uint8_t RDT[RDT_SIZE];

inline bool IsSampledSet(uint32_t set) {
    return (set % 4 == 0);
}

uint32_t HashPC(uint64_t PC) {
    return (PC >> 2) & (SHCT_SIZE - 1);
}

uint32_t HashRegion(uint64_t page_number) {
    return page_number & (RDT_SIZE - 1);
}

void InitReplacementState() {
    std::memset(rrpv, 2, sizeof(rrpv));
    std::memset(dead, 0, sizeof(dead));
    std::memset(hit_once, 0, sizeof(hit_once));
    std::memset(SHCT_demand, 0, sizeof(SHCT_demand));
    std::memset(SHCT_pref, 0, sizeof(SHCT_pref));
    std::memset(RDT, 0, sizeof(RDT));
}

std::pair<bool, bool> PredictDead(uint64_t PC, uint64_t Addr, bool is_prefetch) {
    uint32_t sig = HashPC(PC);
    uint32_t rix = HashRegion(Addr >> 12);
    uint8_t c_pc = is_prefetch ? SHCT_pref[sig] : SHCT_demand[sig];
    uint8_t c_rg = RDT[rix];
    uint8_t dead_score = 2 * c_pc + c_rg;
    bool strong_dead = (c_pc == 3) && (c_rg >= 2);
    bool likely_dead = (dead_score >= 6) || (c_pc >= 2 && c_rg >= 2);
    return {likely_dead, strong_dead};
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) {
            return way;
        }
    }
    
    int victim = -1;
    int max_rrpv = -1;
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (dead[set][way] == 1 && rrpv[set][way] > max_rrpv) {
            victim = way;
            max_rrpv = rrpv[set][way];
        }
    }
    if (victim != -1) return victim;
    
    while (true) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] == 3) return way;
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            rrpv[set][way] = min(3, rrpv[set][way] + 1);
        }
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC,
                            uint64_t victim_addr, uint32_t type, uint8_t hit) {
    if (!hit) {
        auto [likely_dead, strong_dead] = PredictDead(PC, paddr, type == 2);
        dead[set][way] = likely_dead;
        rrpv[set][way] = likely_dead ? 3 : 2;
        hit_once[set][way] = 0;
        if (IsSampledSet(set)) {
            pc_signatures[set][way] = HashPC(PC);
            reuse_outcome[set][way] = 0;
            prefetch_tag[set][way] = (type == 2);
        }
    } else {
        if (dead[set][way]) dead[set][way] = 0;
        if (hit_once[set][way] == 0) {
            hit_once[set][way] = 1;
            rrpv[set][way] = min(rrpv[set][way], 1);
        } else {
            rrpv[set][way] = 0;
        }
        if (IsSampledSet(set)) {
            reuse_outcome[set][way] = 1;
        }
    }
}

void OnEvict(uint32_t set, uint32_t way, uint64_t Addr_evicted) {
    if (IsSampledSet(set)) {
        uint32_t sig = pc_signatures[set][way];
        uint32_t rix = HashRegion(Addr_evicted >> 12);
        bool reused = reuse_outcome[set][way];
        uint8_t &ctr = prefetch_tag[set][way] ? SHCT_pref[sig] : SHCT_demand[sig];
        if (!reused) {
            ctr = min(3, ctr + 1);
            RDT[rix] = min(3, RDT[rix] + 1);
        } else {
            ctr = max(0, ctr - 1);
            RDT[rix] = max(0, RDT[rix] - 1);
        }
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}
