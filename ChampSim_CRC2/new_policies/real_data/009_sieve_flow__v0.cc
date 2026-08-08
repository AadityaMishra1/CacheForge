#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS             = 8;   // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE             = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS       = 12;  // compare low bits of line number
static constexpr uint8_t  STREAM_NEAR_TH      = 2;   // >=2 => near-bypass (stream/quarantine)
static constexpr uint8_t  STREAM_BYPASS_TH    = 3;   // >=3 => sustained run (strong near-bypass)

static constexpr uint8_t  HITS_TO_PROMOTE     = 2;   // promote on 2nd+ demand hit

// Insertion ages: 0=MRU (young) .. 3=LRU (old)
static constexpr uint8_t  AGE_MAX             = 3;   // hard tail
static constexpr uint8_t  AGE_YOUNG           = 1;   // slightly young for warm PCs

// VUT thresholds (2-bit counters: 0..3)
static constexpr uint8_t  VUT_INIT            = 1;   // start slightly cold
static constexpr uint8_t  VUT_WARM_TH         = 2;   // >=2 => warm PC

// Status: 0=normal, 1=stream (near-bypass), 2=quarantine (prefetch/WB)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state (compact; bytes in code, bits accounted in storage) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];  // 2 bits (0..3)
static uint8_t DHITS  [LLC_SETS][LLC_WAYS];  // 2 bits: demand hits 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];  // 2 bits: ST_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];  // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];  // 1-bit mirror of block validity

// ---------------- Global predictors ----------------
// Verified-Use Table (VUT): per-PC friendliness (2-bit 0..3)
static uint8_t VUT[PC_SIZE];

// Forward-run detector per PC
static uint16_t RD_LAST_LINE[PC_SIZE];  // low LINE_LOW_BITS of line#
static uint8_t  RD_CONF[PC_SIZE];       // 2-bit confidence (0..3)
static uint8_t  RD_LAST_ABS12[PC_SIZE]; // 1-bit: last step |delta| in {1,2}
static uint8_t  RD_LAST_FWD[PC_SIZE];   // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple bit-mix for entropy under hashing pressure
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
    // Set-local gentle aging used on fills to keep recency signal lightweight
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update forward-run detector on demand access; returns updated confidence
static inline uint8_t update_run(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RD_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RD_LAST_ABS12[idx] && RD_LAST_FWD[idx]) {
            sat_inc(RD_CONF[idx], 3); // consecutive forward small strides
        } else {
            if (RD_CONF[idx] == 0) RD_CONF[idx] = 1;
            RD_LAST_ABS12[idx] = 1;
            RD_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RD_CONF[idx]);       // decay on breaks/backward/large strides
        RD_LAST_ABS12[idx] = 0;
        RD_LAST_FWD[idx]   = 0;
    }
    RD_LAST_LINE[idx] = curr;
    return RD_CONF[idx];
}

// Eviction ordering: stream/quarantine first, then colder PC, then fewer demand hits, then older age
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    uint8_t va = VUT[ PCSIG[set][a] ];
    uint8_t vb = VUT[ PCSIG[set][b] ];
    if (va != vb) return (va < vb);

    uint8_t ha = DHITS[set][a];
    uint8_t hb = DHITS[set][b];
    if (ha != hb) return (ha < hb);

    return AGE[set][a] > AGE[set][b];
}

// ---------------- CRC2 interface ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            DHITS[s][w]  = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        VUT[i]           = VUT_INIT;
        RD_LAST_LINE[i]  = 0;
        RD_CONF[i]       = 0;
        RD_LAST_ABS12[i] = 0;
        RD_LAST_FWD[i]   = 0;
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Composite victim selection
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
    const bool demand = is_demand(type);

    if (hit) {
        // Demand-hit path: update run detector and verified-use promotion gate
        if (demand) {
            (void)update_run(PC, paddr);
            uint8_t prev = DHITS[set][way];
            if (prev < 2) DHITS[set][way] = static_cast<uint8_t>(prev + 1);

            if (DHITS[set][way] >= HITS_TO_PROMOTE) {
                // Verified reuse: promote and clear sentinels
                AGE[set][way] = 0; // MRU
                if (STATUS[set][way] != ST_NORMAL) STATUS[set][way] = ST_NORMAL;
                // Credit PC once at verification moment
                sat_inc(VUT[ PCSIG[set][way] ], 3);
            }
            // On first demand hit: no promotion (age unchanged)
        } else {
            // Prefetch and writeback hits do not affect promotion or DHITS
        }
        return;
    }

    // Miss/fill path (way holds victim): train on victim if valid
    if (LVALID[set][way]) {
        uint8_t old_hits = DHITS[set][way];
        uint8_t old_pc   = PCSIG[set][way];
        if (old_hits <= 1) sat_dec(VUT[old_pc]); // penalize ≤1-use lines
    }

    // Age others gently on fill, then install the new line
    age_all(set);

    uint32_t pc_idx = pc_index(PC);

    // Default initialization for the new line
    PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
    DHITS[set][way]  = 0;
    LVALID[set][way] = 1;

    if (type == ACCESS_PREFETCH) {
        // Quarantine all prefetches at tail; never promote from prefetch hit
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    if (type == ACCESS_WRITEBACK) {
        // Never bypass WB; insert at tail with quarantine to prefer early eviction
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Demand fill: stream detection then VUT-based insertion
    uint8_t conf = update_run(PC, paddr);

    if (conf >= STREAM_NEAR_TH) {
        // Stream/scans: near-bypass at tail with stream sentinel; sustained runs keep them evict-first
        STATUS[set][way] = ST_STREAM;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Non-stream: VUT-guided insertion depth
    if (VUT[pc_idx] >= VUT_WARM_TH) {
        STATUS[set][way] = ST_NORMAL;
        AGE[set][way]    = AGE_YOUNG;
    } else {
        STATUS[set][way] = ST_NORMAL;
        AGE[set][way]    = AGE_MAX;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}