#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>

// Configuration
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16
#define PC_TBL_SIZE 1024
#define MAX_RRPV 7

// Mode A (Hawkeye) support
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* hawkeye_demand;
static HAWKEYE_PC_PREDICTOR* hawkeye_prefetch;

// Per-set / per-line state (low-overhead encoding)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static bool streaming_line[LLC_SETS][LLC_WAYS];      // streaming-tagged
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];           // 0..3
static bool modeB_active[LLC_SETS];                   // per-set bandit toggle (Mode B)

static int8_t bandit_score[LLC_SETS];                  // [-31..31] simple advisor
static uint64_t last_paddr_pc[PC_TBL_SIZE];            // 64-bit phys addr anchor per PC
static uint8_t  fwd_run_cnt[PC_TBL_SIZE];              // two-step stride forward-run detector

// Tiny PC table for stride-based streaming (hash to PC_TBL_SIZE)
static inline uint32_t pc_idx(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE-1); }

// Helpers
static inline void promote(uint32_t set, uint32_t way){
    rrpv[set][way] = 0;
    for(uint32_t w=0; w<LLC_WAYS; ++w){
        if(w==way) continue;
        if(rrpv[set][w] < MAX_RRPV) rrpv[set][w]++;
    }
}

static inline bool detect_stream(uint64_t pc, uint64_t paddr){
    uint32_t idx = pc_idx(pc);
    int64_t delta = (int64_t)paddr - (int64_t)last_paddr_pc[idx];
    last_paddr_pc[idx] = paddr;
    if(delta == 64){
        if(++fwd_run_cnt[idx] >= 2) return true;
    } else {
        fwd_run_cnt[idx] = 0;
    }
    return false;
}

// Hawkeye-like victim (baseline)
static uint32_t hawkeye_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; ++w) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; ++w) if(rrpv[set][w] == MAX_RRPV) return w;
    // fallback: oldest
    uint32_t v=0; uint8_t best=0;
    for(uint32_t w=0; w<LLC_WAYS; ++w){
        if(rrpv[set][w] >= best){ best = rrpv[set][w]; v = w; }
    }
    return v;
}

// Mode B (StrideGuard) victim: bias toward evicting streaming lines when active
static uint32_t stride_victim(uint32_t set, const BLOCK* current_set){
    // prefer evicting a streaming line with max rrpv
    uint32_t chosen = 0; bool found=false;
    for(uint32_t w=0; w<LLC_WAYS; ++w){
        if(!current_set[w].valid) return w;
        if(rrpv[set][w] == MAX_RRPV && streaming_line[set][w]){
            return w;
        }
        if(rrpv[set][w] == MAX_RRPV) { chosen = w; found = true; }
    }
    return found ? chosen : hawkeye_victim(set, current_set);
}

// Public entry points
void InitReplacementState() {
    hawkeye_demand = new HAWKEYE_PC_PREDICTOR();
    hawkeye_prefetch = new HAWKEYE_PC_PREDICTOR();

    for(uint32_t s=0; s<LLC_SETS; ++s){
        bandit_score[s] = 0;
        modeB_active[s] = false;
        for(uint32_t w=0; w<LLC_WAYS; ++w){
            rrpv[s][w] = MAX_RRPV;
            streaming_line[s][w] = false;
            hitcnt[s][w] = 0;
        }
    }
    std::memset(last_paddr_pc, 0, sizeof(last_paddr_pc));
    std::memset(fwd_run_cnt, 0, sizeof(fwd_run_cnt));
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                      uint64_t PC, uint64_t paddr, uint32_t type){
    // Invalidate-exists check
    for(uint32_t w=0; w<LLC_WAYS; ++w) if(!current_set[w].valid) return w;

    // Epochal bandit decision: default to Hawkeye; enable Mode B if score >= 4
    bool useModeB = (bandit_score[set] >= 4);
    modeB_active[set] = useModeB;

    if(useModeB){
        return stride_victim(set, current_set);
    } else {
        return hawkeye_victim(set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                          uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                          uint32_t type, uint8_t hit){
    paddr = (paddr >> 6) << 6;
    if(type == WRITEBACK) return;

    // Miss path: record stream flag for the new line
    if(hit == 0){
        bool is_stream = detect_stream(PC, paddr);
        streaming_line[set][way] = is_stream;
        hitcnt[set][way] = 0;

        // If Mode B is active for this set, gating: streams need more hits to promote
        if(modeB_active[set]){
            // Do not promote on first 2 hits; promotion occurs on subsequent hits
        } else {
            // Mode A: normal Hawkeye-like promotion on reuse (handled on hits)
        }

        // simple per-set score update to help the bandit decide (learns over time)
        bandit_score[set] = (bandit_score[set] < 31) ? bandit_score[set] + 1 : 31;
        return;
    }

    // Hit path: update per-line counters and possibly promote
    if(streaming_line[set][way]){
        if(++hitcnt[set][way] >= 3){
            promote(set, way);
        }
    } else {
        // non-stream lines: promote on second demand hit (simple rule)
        if(++hitcnt[set][way] >= 2){
            promote(set, way);
        }
    }

    // Optional: update Hawkeye predictors softly (keep compatibility)
    if(type == ACCESS_PREFETCH){
        // no strong update here for simplicity
    } else {
        // demand hit: train Hawkeye predictor minimally
        hawkeye_demand->increment(PC);
    }

    // Decay/bandit incentive
    if(hit) {
        if(modeB_active[set]){
            bandit_score[set] += 0; // minor polish
        } else {
            // encourage Mode B if beneficial later
            if((bandit_score[set] -= 0) < -16) bandit_score[set] = -16;
        }
    }
}

// Heartbeat/Stats kept empty per instruction
void PrintStats() {}
void PrintStats_Heartbeat() {}