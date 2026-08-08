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

// ---------------- Tunables ----------------
// PC-indexing
static constexpr uint8_t  PC_BITS            = 8;                // 256-entry PC tables
static constexpr uint32_t PC_SIZE            = (1u << PC_BITS);

// TinyLFU friendliness (3-bit, 0..7)
static constexpr uint8_t  FRIEND_BITS        = 3;
static constexpr uint8_t  FRIEND_MAX         = (1u << FRIEND_BITS) - 1; // 7
static constexpr uint8_t  FRIEND_INIT        = 1;                // slightly cold start
static constexpr uint8_t  FRIEND_WARM_TH     = 3;                // >=3 => warm
static constexpr uint8_t  FRIEND_COLD_QUAR   = 0;                // ==0 => optionally quarantine

// RunStride stream shield
static constexpr uint8_t  LINE_LOW_BITS      = 12;               // compare low line bits
static constexpr uint8_t  STREAM_TAIL_TH     = 2;                // >=2 => treat as stream near-bypass

// Multi-hit promotion
static constexpr uint8_t  HITS_TO_PROMOTE    = 2;                // promote on 2nd demand hit

// Age depth (0=young/MRU .. 3=old/LRU)
static constexpr uint8_t  AGE_MAX            = 3;
static constexpr uint8_t  AGE_YOUNG          = 1;                // for warm PCs

// TinyLFU aging (global)
static constexpr uint32_t AGING_PERIOD_FILLS = 4096;             // every N fills, decay all FRIEND by 1

// ---------------- Per-line state (compact conceptual bits) ----------------
// Note: Stored in byte arrays for simplicity; storage accounted minimally in report.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2 bits: 0=normal,1=stream,2=quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: line valid snapshot for eviction training

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors ----------------
// TinyLFU per-PC friendliness
static uint8_t FRIEND[PC_SIZE];               // 3-bit conceptual (0..7)

// RunStride per-PC state
static uint16_t RS_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit: last step |delta| in {1,2}
static uint8_t  RS_LAST_FWD[PC_SIZE];         // 1-bit: last step forward

// TinyLFU aging clock
static uint64_t fill_clock = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix for entropy
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc_u8(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec_u8(uint8_t &v) { if (v > 0) v--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update RunStride on demand access; returns updated confidence
static inline uint8_t update_runstride(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad    = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc_u8(RS_CONF[idx], 3);     // two consecutive forward steps
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RS_CONF[idx]);            // decay on break/backward/large stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// Compare candidates: true if 'a' is more evictable than 'b'
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Sentinels first (stream/quarantine)
    bool sa = (STATUS[set][a] != ST_NORMAL);
    bool sb = (STATUS[set][b] != ST_NORMAL);
    if (sa != sb) return sa;

    // 2) Colder PCs first (lower TinyLFU)
    uint8_t fa = FRIEND[ PCSIG[set][a] ];
    uint8_t fb = FRIEND[ PCSIG[set][b] ];
    if (fa != fb) return (fa < fb);

    // 3) Fewer demand hits first
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Older age last
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
        FRIEND[i]        = FRIEND_INIT;
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
    }
    fill_clock = 0;
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
    // Return first invalid if present
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
    const bool demand   = is_demand(type);
    const bool prefetch = (type == ACCESS_PREFETCH);
    const bool wb       = (type == ACCESS_WRITEBACK);
    const uint32_t pidx = pc_index(PC);

    // Demand accesses update RunStride
    uint8_t rs_conf = RS_CONF[pidx];
    if (demand) rs_conf = update_runstride(PC, paddr);

    if (hit) {
        // Age others once per access
        age_all(set);

        // TinyLFU increment on demand hits
        if (demand) sat_inc_u8(FRIEND[pidx], FRIEND_MAX);

        // Demand hit accounting
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            // Promote only on 2nd+ demand hit; clear sentinels then
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way] = 0; // MRU
            }
            // Else: no promotion on first hit by design
        }
        // Prefetch/WB hits: no promotion, no counters
        return;
    }

    // Miss path: train on victim (previous occupant in this way), then insert
    if (LVALID[set][way]) {
        uint8_t v_pc = PCSIG[set][way];
        uint8_t v_hits = HITCNT[set][way];
        if (v_hits >= 2) {
            sat_inc_u8(FRIEND[v_pc], FRIEND_MAX);
        } else {
            sat_dec_u8(FRIEND[v_pc]);
        }
    }

    // Global TinyLFU aging on fills
    fill_clock++;
    if ((fill_clock % AGING_PERIOD_FILLS) == 0) {
        for (uint32_t i = 0; i < PC_SIZE; i++) {
            sat_dec_u8(FRIEND[i]); // periodic decay
        }
    }

    // Age others once per access
    age_all(set);

    // Decide insertion state
    uint8_t init_status = ST_NORMAL;
    uint8_t init_age    = AGE_MAX;

    if (prefetch) {
        init_status = ST_QUAR;                   // quarantine prefetches
        init_age    = AGE_MAX;
    } else if (wb) {
        init_status = ST_QUAR;                   // never bypass WB, but keep low priority
        init_age    = AGE_MAX;
    } else if (demand && rs_conf >= STREAM_TAIL_TH) {
        init_status = ST_STREAM;                 // stream near-bypass
        init_age    = AGE_MAX;
    } else {
        // PC-based friendliness
        if (FRIEND[pidx] >= FRIEND_WARM_TH) {
            init_age = AGE_YOUNG;                // slightly young for warm PCs
        } else {
            init_age = AGE_MAX;                  // hard tail for cold PCs
            if (FRIEND[pidx] == FRIEND_COLD_QUAR) {
                init_status = ST_QUAR;           // extra guard for extremely cold PCs
            }
        }
    }

    // Increment TinyLFU on demand access (miss counts as an access)
    if (demand) sat_inc_u8(FRIEND[pidx], FRIEND_MAX);

    // Install metadata
    AGE[set][way]    = init_age;
    STATUS[set][way] = init_status;
    HITCNT[set][way] = 0;
    PCSIG[set][way]  = static_cast<uint8_t>(pidx);
    LVALID[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}