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
// PC tables (compact)
static constexpr uint8_t  PC_BITS   = 8;                // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE   = (1u << PC_BITS);

// Run detector: ±1/±2 stride; >=2 => stream-guard near-bypass
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;       // confidence threshold (0..3)
static constexpr uint8_t  LINE_LOW_BITS      = 8;       // low bits of line number to compare

// PC Cold Filter
static constexpr uint8_t  CF_WARM_TH = 2;               // >=2 => predicted warm

// Insertion depths (2Q age within a set, 0=youngest..3=oldest)
static constexpr uint8_t  AGE_MAX    = 3;               // tail
static constexpr uint8_t  AGE_YOUNG  = 1;               // slightly young for warm PCs

// Status: 0=normal, 1=stream-guard(no promote/train), 2=quarantine(prefetch/WB)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// Queue bit: 0=Probation, 1=Protected
// Hits-to-promote: demand-only; promote on reaching 2+
static constexpr uint8_t  HITS_TO_PROMOTE = 2;

// ---------------- Per-line state ----------------
// Queue membership
static uint8_t QUEUE_BIT[LLC_SETS][LLC_WAYS];   // 0 or 1
// Status sentinel
static uint8_t STATUS[LLC_SETS][LLC_WAYS];      // 0..2
// Age within a set (small LRU-like age)
static uint8_t AGE[LLC_SETS][LLC_WAYS];         // 0..3
// Demand hit counter (0,1,2+)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];      // 0..3 (we use up to 3 as 2+)
// PC signature (8-bit index)
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];       // 0..255

// ---------------- Global predictors ----------------
// PC Cold Filter: 2-bit counters per PC (0..3)
static uint8_t PC_CT[PC_SIZE];

// Run detector tables per PC
static uint8_t ST_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t ST_CONF[PC_SIZE];        // 2-bit confidence (0..3)
static uint8_t ST_LAST_ABS12[PC_SIZE];  // 1-bit: previous step was |delta|==1 or 2

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint8_t line_lowN(uint64_t paddr) {
    return static_cast<uint8_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
// Update run-detector and return new confidence (demand only)
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint8_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(ST_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);

    if (abs12) {
        if (ST_LAST_ABS12[idx]) {
            sat_inc(ST_CONF[idx], 3);   // consecutive ±1/±2 confirms a run
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS12[idx] = 1;
        }
    } else {
        sat_dec(ST_CONF[idx]);          // decay on break
        ST_LAST_ABS12[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Age all lines in a set (bounded)
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting stream/prefetch/WB sentinels
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer Probation over Protected
    uint8_t qa = QUEUE_BIT[set][a];
    uint8_t qb = QUEUE_BIT[set][b];
    if (qa != qb) return (qa < qb); // 0 (Probation) evictable before 1 (Protected)

    // 3) Prefer PCs predicted colder (lower counter)
    uint8_t cta = PC_CT[ PCSIG[set][a] & (PC_SIZE - 1) ];
    uint8_t ctb = PC_CT[ PCSIG[set][b] & (PC_SIZE - 1) ];
    if (cta != ctb) return (cta < ctb);

    // 4) Prefer fewer observed hits (0 < 1 < 2+)
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 5) Tie-break by age: older is more evictable
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            QUEUE_BIT[s][w] = 0;       // Probation
            STATUS[s][w]    = ST_NORMAL;
            AGE[s][w]       = AGE_MAX; // old by default
            HITCNT[s][w]    = 0;
            PCSIG[s][w]     = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_CT[i]         = 1; // slightly cold start
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS12[i] = 0;
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

    // Composite priority victim selection (no unbounded loops)
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

    // Update run detector on every demand access (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        // Age everyone (bounded), then decide on the accessed line
        age_all(set);

        // If sentinel (stream/prefetch/WB), do not promote or train; also do not refresh its age
        if (STATUS[set][way] != ST_NORMAL) {
            return;
        }

        // Demand hits: multi-hit gating
        if (demand) {
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;

            // On 2nd+ demand hit, promote to Protected and credit PC Cold Filter (once per line)
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                if (QUEUE_BIT[set][way] == 0) {
                    // First time crossing the gate
                    sat_inc(PC_CT[ PCSIG[set][way] & (PC_SIZE - 1) ], 3);
                }
                QUEUE_BIT[set][way] = 1; // Protected
            }
        }

        // Refresh age on any hit of a normal line
        AGE[set][way] = 0;
        return;
    }

    // MISS path: train on eviction (previous occupant of 'way')
    {
        uint8_t prev_status = STATUS[set][way];
        uint8_t prev_hits   = HITCNT[set][way];
        uint8_t prev_sig    = PCSIG[set][way];
        // Dead-on-evict training for normal demand fills only
        if (prev_status == ST_NORMAL && prev_hits == 0) {
            sat_dec(PC_CT[ prev_sig & (PC_SIZE - 1) ]);
        }
    }

    // Insert new block
    age_all(set); // age existing lines before inserting

    uint32_t pc_idx = pc_index(PC);

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; quarantine (no train/promote)
        STATUS[set][way]    = ST_QUAR;
        QUEUE_BIT[set][way] = 0;        // Probation
        AGE[set][way]       = AGE_MAX;  // tail
        HITCNT[set][way]    = 0;
        PCSIG[set][way]     = 0;        // avoid training by WB context
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine: tail, probation, no promotion/training
        STATUS[set][way]    = ST_QUAR;
        QUEUE_BIT[set][way] = 0;
        AGE[set][way]       = AGE_MAX;
        HITCNT[set][way]    = 0;
        PCSIG[set][way]     = static_cast<uint8_t>(pc_idx);
        return;
    }

    // Demand fill
    const bool stream_like = (stream_conf >= STREAM_CONF_BYPASS);

    STATUS[set][way]    = stream_like ? ST_STREAM : ST_NORMAL;
    QUEUE_BIT[set][way] = 0;                    // all inserts start in Probation (2Q)
    HITCNT[set][way]    = 0;
    PCSIG[set][way]     = static_cast<uint8_t>(pc_idx);

    // Insertion depth: near-bypass for streams and cold PCs; slightly young for warm PCs
    if (stream_like) {
        AGE[set][way] = AGE_MAX;                // hard tail for streams; never promoted
    } else {
        uint8_t pred = PC_CT[pc_idx];
        AGE[set][way] = (pred >= CF_WARM_TH) ? AGE_YOUNG : AGE_MAX;
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