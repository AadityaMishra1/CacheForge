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
// PC-indexed tables
static constexpr uint8_t  PC_IDX_BITS     = 8;                // 256-entry tables
static constexpr uint32_t PC_SIZE         = (1u << PC_IDX_BITS);

// LifeScore predictor: 2-bit (0..3)
static constexpr uint8_t  LS_INIT         = 1;                // slightly cold start
static constexpr uint8_t  LS_WARM_TH      = 2;                // >=2 => warm (insert young)

// StreamGate: per-PC +1/+2 forward-run detector
static constexpr uint8_t  LINE_LOW_BITS   = 12;               // low bits of line# to compare
static constexpr uint8_t  SG_BYPASS_TH    = 2;                // >=2 => near-bypass (hard tail)

// Insertion age (0=MRU/young .. 3=LRU/old)
static constexpr uint8_t  AGE_MAX         = 3;
static constexpr uint8_t  AGE_YOUNG       = 1;                // slightly young for warm PCs

// Multi-hit promotion
static constexpr uint8_t  HITS_TO_PROMOTE = 2;                // promote on 2nd+ demand hit

// Per-line status
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };       // stream-guard and quarantine

// ---------------- Per-line state (compact) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits conceptual (0..3)
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 0,1,2(=2+ demand hits)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // ST_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index

// ---------------- Global predictors ----------------
// LifeScore (PC-indexed 2-bit counters)
static uint8_t LIFESCORE[PC_SIZE];

// StreamGate per-PC state
static uint16_t SG_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t  SG_CONF[PC_SIZE];        // 2-bit (0..3)
static uint8_t  SG_LAST_ABS12[PC_SIZE];  // 1-bit: last was |delta| in {1,2}
static uint8_t  SG_LAST_FWD[PC_SIZE];    // 1-bit: last step was forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix for indexing PCs into small tables
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update StreamGate on demand access; returns confidence
static inline uint8_t update_streamgate(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(SG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (SG_LAST_ABS12[idx] && SG_LAST_FWD[idx]) {
            sat_inc(SG_CONF[idx], 3);      // two consecutive forward steps → stream
        } else {
            if (SG_CONF[idx] == 0) SG_CONF[idx] = 1;
            SG_LAST_ABS12[idx] = 1;
            SG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(SG_CONF[idx]);             // decay on break/backward/irregular
        SG_LAST_ABS12[idx] = 0;
        SG_LAST_FWD[idx]   = 0;
    }
    SG_LAST_LINE[idx] = curr;
    return SG_CONF[idx];
}

// Candidate comparison: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict sentinels first (stream/quarantine)
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Evict lines from colder PCs (lower LifeScore)
    uint8_t lsa = LIFESCORE[ PCSIG[set][a] & (PC_SIZE - 1) ];
    uint8_t lsb = LIFESCORE[ PCSIG[set][b] & (PC_SIZE - 1) ];
    if (lsa != lsb) return (lsa < lsb);

    // 3) Evict lines with fewer demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age: older is more evictable
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]     = AGE_MAX;
            HITCNT[s][w]  = 0;
            STATUS[s][w]  = ST_NORMAL;
            PCSIG[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        LIFESCORE[i]     = LS_INIT;
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

    // Demand access updates stream detector (works on both hits and misses)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_streamgate(PC, paddr);
    }

    // On miss (fill), train LifeScore on the evicted line (using old metadata),
    // then install the new line with StreamGate + LifeScore guided insertion.
    if (!hit) {
        // Train LifeScore on victim's observed lifetime if the slot was previously used
        uint8_t old_pc = PCSIG[set][way];
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_stat = STATUS[set][way];
        if ((old_pc | old_hits | old_stat) != 0) {
            if (old_hits >= 1) sat_inc(LIFESCORE[old_pc & (PC_SIZE - 1)], 3);
            else               sat_dec(LIFESCORE[old_pc & (PC_SIZE - 1)]);
        }

        // Install new line
        uint8_t pcidx = static_cast<uint8_t>(pc_index(PC));
        PCSIG[set][way]  = pcidx;
        HITCNT[set][way] = 0;

        if (type == ACCESS_PREFETCH) {
            // Prefetch: quarantine at hard tail, never promote until confirmed by 2nd demand hit
            STATUS[set][way] = ST_QUAR;
            AGE[set][way]    = AGE_MAX;
        } else if (type == ACCESS_WRITEBACK) {
            // Writeback never bypass: keep quarantined at tail
            STATUS[set][way] = ST_QUAR;
            AGE[set][way]    = AGE_MAX;
        } else {
            // Demand fill: apply StreamGate near-bypass or LifeScore-guided insertion
            if (stream_conf >= SG_BYPASS_TH) {
                STATUS[set][way] = ST_STREAM;  // stream-guarded, no early promotion
                AGE[set][way]    = AGE_MAX;    // hard tail
            } else {
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way]    = (LIFESCORE[pcidx] >= LS_WARM_TH) ? AGE_YOUNG : AGE_MAX;
            }
        }
        return;
    }

    // On hit: apply HitGate and controlled promotion
    if (demand) {
        // Count only demand hits for promotion decisions
        if (HITCNT[set][way] == 0) {
            HITCNT[set][way] = 1;               // first demand hit observed
            if (AGE[set][way] > 1) AGE[set][way] = 1; // mild refresh, not MRU
        } else {
            // 2nd+ demand hit: promote to MRU and clear any sentinel
            HITCNT[set][way] = 2;
            age_all(set);
            AGE[set][way] = 0;                  // MRU
            if (STATUS[set][way] != ST_NORMAL) STATUS[set][way] = ST_NORMAL;
            // Reinforce LifeScore on confirmed reuse
            sat_inc(LIFESCORE[ PCSIG[set][way] & (PC_SIZE - 1) ], 3);
        }
    } else {
        // Prefetch or writeback hit: no promotion; keep quarantine/stream guards intact
        // slight optional refresh avoided to preserve scan resistance
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}