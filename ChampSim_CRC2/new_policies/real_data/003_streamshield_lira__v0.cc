#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables (knobs) ----------------
static constexpr uint8_t  PC_BITS         = 8;               // 256-entry per-PC tables
static constexpr uint32_t PC_SIZE         = (1u << PC_BITS);

// LiRA (per-PC liveness) 2-bit counter: 0..3
static constexpr uint8_t  LIRA_INIT       = 1;               // start slightly cold
static constexpr uint8_t  LIRA_WARM_TH    = 2;               // >=2 => warm (likely multi-use)

// StreamGate: per-PC ±1/±2 forward-run detector
static constexpr uint8_t  LINE_LOW_BITS   = 12;              // compare low bits of line#
static constexpr uint8_t  STREAM_CONF_TH  = 2;               // >=2 => stream near-bypass

// Age: 0=young/MRU .. 3=old/tail
static constexpr uint8_t  AGE_MAX         = 3;
static constexpr uint8_t  AGE_YOUNG       = 1;

// Multi-hit promotion gate
static constexpr uint8_t  HITS_TO_PROMOTE = 2;               // promote on 2nd+ demand hit

// Status tags
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state (compact intent) ----------------
// Note: Arrays use byte storage; report info-theoretic bits in storage section.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2 bits: ST_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: line held valid data previously

// ---------------- Global predictors ----------------
// LiRA per-PC friendliness (2-bit)
static uint8_t LIRA[PC_SIZE];                 // 0..3

// StreamGate per-PC state
static uint16_t SG_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  SG_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  SG_LAST_ABS12[PC_SIZE];       // 1-bit: last step |delta| in {1,2}
static uint8_t  SG_LAST_FWD[PC_SIZE];         // 1-bit: last step was forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix for low-bit entropy
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
}
// Update StreamGate on demand access; returns new confidence
static inline uint8_t update_streamgate(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(SG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (SG_LAST_ABS12[idx] && SG_LAST_FWD[idx]) {
            sat_inc(SG_CONF[idx], 3);        // two consecutive forward steps
        } else {
            if (SG_CONF[idx] == 0) SG_CONF[idx] = 1;
            SG_LAST_ABS12[idx] = 1;
            SG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(SG_CONF[idx]);               // decay on irregular/backwards/big stride
        SG_LAST_ABS12[idx] = 0;
        SG_LAST_FWD[idx]   = 0;
    }
    SG_LAST_LINE[idx] = curr;
    return SG_CONF[idx];
}

// Victim priority comparator
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting quarantined/stream-sentinel lines
    bool sa = (STATUS[set][a] != ST_NORMAL);
    bool sb = (STATUS[set][b] != ST_NORMAL);
    if (sa != sb) return sa;

    // 2) Prefer lines from colder PCs by LiRA score
    uint8_t la = LIRA[ PCSIG[set][a] ];
    uint8_t lb = LIRA[ PCSIG[set][b] ];
    if (la != lb) return (la < lb);

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        LIRA[i]          = LIRA_INIT;
        SG_LAST_LINE[i]  = 0;
        SG_CONF[i]       = 0;
        SG_LAST_ABS12[i] = 0;
        SG_LAST_FWD[i]   = 0;
    }
}

// Find victim in the set
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

    // Composite priority selection
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

// Update replacement state
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
    const bool demand = is_demand(type);

    // Update stream detector only on demand accesses
    if (demand) {
        update_streamgate(PC, paddr);
    }

    if (hit) {
        // On hit: enforce multi-hit gating and gentle recency
        age_all(set);

        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Confirmed reuse: clear sentinel and promote to MRU
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way]    = 0;
                // Reinforce LiRA
                uint32_t idx = pc_index(PC);
                sat_inc(LIRA[idx], 3);
            } else {
                // First demand hit: keep slightly young, do not MRU
                if (AGE[set][way] > AGE_YOUNG) AGE[set][way] = AGE_YOUNG;
            }
        } else {
            // Prefetch/writeback-triggered hit: never promote here
            // optionally keep current age; no LiRA training
        }
        return;
    }

    // Miss path (insertion): train on eviction if the victim held valid data
    if (LVALID[set][way]) {
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_pc   = PCSIG[set][way];
        if (old_hits >= HITS_TO_PROMOTE) sat_inc(LIRA[old_pc], 3);
        else sat_dec(LIRA[old_pc]);
    }

    // Decide insertion policy
    uint32_t idx = pc_index(PC);
    uint8_t ins_age = AGE_MAX;
    uint8_t ins_st  = ST_NORMAL;

    if (type == ACCESS_PREFETCH) {
        // Prefetch: always quarantine at tail
        ins_st  = ST_QUAR;
        ins_age = AGE_MAX;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass on WB: insert evictable at tail
        ins_st  = ST_QUAR;
        ins_age = AGE_MAX;
    } else { // Demand (LOAD/RFO)
        uint8_t sconf = SG_CONF[idx];
        if (sconf >= STREAM_CONF_TH) {
            // Stream near-bypass: tail insert with stream sentinel
            ins_st  = ST_STREAM;
            ins_age = AGE_MAX;
        } else {
            // Use LiRA liveness to choose depth
            if (LIRA[idx] >= LIRA_WARM_TH) ins_age = AGE_YOUNG;
            else                           ins_age = AGE_MAX;
            ins_st = ST_NORMAL;
        }
    }

    // Perform insertion
    age_all(set);
    AGE[set][way]    = ins_age;
    HITCNT[set][way] = 0;
    STATUS[set][way] = ins_st;
    PCSIG[set][way]  = static_cast<uint8_t>(idx);
    LVALID[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}