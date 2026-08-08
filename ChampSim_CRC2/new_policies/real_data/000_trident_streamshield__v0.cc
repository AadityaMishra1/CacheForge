#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

// TRIDENT-StreamShield (exploration variant)
// - StreamShield: per-PC ±1/±2 forward-run detector -> stream near-bypass (hard tail), never promote
// - PC-UseScore: 2-bit expected-use (trained on 2+ demand hits vs. dead evictions)
// - Two-Stage Promotion: promote only on 2nd+ demand hit; prefetches quarantined at tail
// Eviction order: stream/quarantine -> colder PCs -> fewer hits -> older age

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags per CRC2
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS        = 9;        // 512-entry per-PC tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS  = 12;       // low bits of line number for run detector
static constexpr uint8_t  RG_BYPASS_TH   = 2;        // >=2 => treat as stream

static constexpr uint8_t  AGE_MAX        = 3;        // 2-bit age: 0=MRU .. 3=LRU
static constexpr uint8_t  AGE_YOUNG      = 1;        // insertion depth for warm PCs
static constexpr uint8_t  HITS_TO_PROMOTE = 2;       // promote on 2nd+ demand hit

// Status: 0=normal, 1=stream (never promote), 2=quarantine (prefetch tail)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state ----------------
static uint8_t AGE     [LLC_SETS][LLC_WAYS];   // 2 bits conceptual (0..3)
static uint8_t HITCNT  [LLC_SETS][LLC_WAYS];   // 2 bits (0..3, with 2+ meaning multi)
static uint8_t STATUS  [LLC_SETS][LLC_WAYS];   // ST_*
static uint8_t PCSIG   [LLC_SETS][LLC_WAYS];   // 8-bit PC index (hash)
static uint8_t VALIDLN [LLC_SETS][LLC_WAYS];   // 1-bit: metadata valid for training

// ---------------- Global predictors ----------------
// PC-UseScore: 2-bit warmness per PC (0..3); warm if >=2
static uint8_t USESCORE[PC_SIZE];

// RunGuard per-PC: last-line low bits and streaming confidence + tiny history flags
static uint16_t RG_LAST_LINE[PC_SIZE];  // 12 bits effective
static uint8_t  RG_CONF[PC_SIZE];       // 2-bit (0..3) stream confidence
static uint8_t  RG_LAST_ABS12[PC_SIZE]; // 1-bit: last step magnitude was 1 or 2
static uint8_t  RG_LAST_FWD[PC_SIZE];   // 1-bit: last step was forward

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // Low-cost hash to spread PCs into PC_SIZE entries
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 17) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
}

// Update RunGuard on a demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad    = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3); // two consecutive forward small steps => stream
        } else {
            // first sighting of potential stream
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);        // decay on breaks/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict sentinels first

    uint8_t usa = USESCORE[ PCSIG[set][a] ];
    uint8_t usb = USESCORE[ PCSIG[set][b] ];
    if (usa != usb) return (usa < usb); // colder PCs first

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);     // fewer hits first

    return AGE[set][a] > AGE[set][b];   // older age last tie-break
}

// ---------------- ChampSim hooks ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]     = AGE_MAX;
            HITCNT[s][w]  = 0;
            STATUS[s][w]  = ST_NORMAL;
            PCSIG[s][w]   = 0;
            VALIDLN[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        USESCORE[i]       = 1; // slightly cold start
        RG_LAST_LINE[i]   = 0;
        RG_CONF[i]        = 0;
        RG_LAST_ABS12[i]  = 0;
        RG_LAST_FWD[i]    = 0;
    }
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Composite selection
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    // Ignore writebacks completely
    if (type == ACCESS_WRITEBACK) return;

    const bool demand = is_demand(type);
    const uint32_t pcidx = pc_index(PC);

    // Age once per demand access to maintain recency bias
    if (demand) age_all(set);

    if (hit) {
        // Demand hit: update hit count; promotion only on 2nd+ demand hit
        if (demand) {
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;

            // Stream-sentinels never promote
            if (STATUS[set][way] != ST_STREAM) {
                if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                    // Shed quarantine (prefetch) and promote
                    STATUS[set][way] = ST_NORMAL;
                    AGE[set][way] = 0; // MRU
                    // Reinforce UseScore on proven reusers
                    sat_inc(USESCORE[ PCSIG[set][way] ], 3);
                }
            }
        }
        // On prefetch hits, do not promote; keep quarantine until a later demand hit
        return;
    }

    // Miss/fill path: train on eviction of prior occupant if it was valid
    if (VALIDLN[set][way]) {
        const uint8_t prev_sig   = PCSIG[set][way];
        const uint8_t prev_hits  = HITCNT[set][way];
        const uint8_t prev_stat  = STATUS[set][way];

        // Reuse if 2+ demand hits and not a stream-sentinel
        if (prev_hits >= 2 && prev_stat != ST_STREAM) {
            sat_inc(USESCORE[prev_sig], 3);
        } else {
            sat_dec(USESCORE[prev_sig]);
        }
    }

    // Decide insertion for the new line
    uint8_t new_status = ST_NORMAL;
    uint8_t new_age    = AGE_MAX;
    uint8_t new_hitcnt = 0;

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine at hard tail
        new_status = ST_QUAR;
        new_age    = AGE_MAX;
    } else if (demand) {
        // Update RunGuard and apply stream near-bypass
        uint8_t rgc = update_runguard(PC, paddr);
        if (rgc >= RG_BYPASS_TH) {
            new_status = ST_STREAM;     // never promote
            new_age    = AGE_MAX;       // hard tail
        } else {
            // PC-UseScore guided adaptive insertion
            if (USESCORE[pcidx] >= 2)   // warm
                new_age = AGE_YOUNG;
            else
                new_age = AGE_MAX;
            new_status = ST_NORMAL;
        }
    }

    // Install new metadata
    AGE[set][way]     = new_age;
    STATUS[set][way]  = new_status;
    HITCNT[set][way]  = new_hitcnt;
    PCSIG[set][way]   = static_cast<uint8_t>(pcidx & 0xFF); // store low 8 bits
    VALIDLN[set][way] = 1;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}