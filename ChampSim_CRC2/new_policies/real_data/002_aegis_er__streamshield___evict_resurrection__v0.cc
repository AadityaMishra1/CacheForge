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
// PC tables sizing (keep small to fit budget)
static constexpr uint32_t PC_BITS   = 8;                 // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE   = (1u << PC_BITS);
static constexpr uint8_t  PC_INIT   = 1;                 // slightly cold start (0..3)
static constexpr uint8_t  PC_WARM_TH = 2;                // >=2 => warm (insert higher)

// StreamShield: per-PC ±1-line stride detector
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;        // >=2 => stream-guard near-bypass
static constexpr uint8_t  LINE_LOW_BITS = 8;             // track low 8 bits of line number

// Evict-Resurrection Bloom filter
static constexpr uint32_t BLOOM_BITS     = 4096;         // bits per plane
static constexpr uint32_t BLOOM_WORDS    = (BLOOM_BITS / 64);
static constexpr uint32_t BLOOM_HASHES   = 2;            // use 2 hashes
static constexpr uint32_t EPOCH_PERIOD   = 2048;         // fills between plane rotation

// Per-line age (0=youngest..3=oldest)
static constexpr uint8_t AGE_MAX = 3;

// Status encodings: 0=normal, 1=stream-guard (no train/promote), 2=PF/WB sentinel (no train/promote)
enum : uint8_t { ST_NORMAL = 0, ST_STREAM = 1, ST_SENTINEL = 2 };

// ---------------- Per-line state (packed logically) ----------------
// AGE: 2 bits (0..3)
static uint8_t  AGE[LLC_SETS][LLC_WAYS];
// HITCNT: 2 bits (0..3) demand hits only; 0=none, 1=one, 2+=two or more
static uint8_t  HITCNT[LLC_SETS][LLC_WAYS];
// STATUS: 2 bits (ST_NORMAL/ST_STREAM/ST_SENTINEL)
static uint8_t  STATUS[LLC_SETS][LLC_WAYS];
// PCSIG: 8-bit PC index for training on eviction
static uint8_t  PCSIG[LLC_SETS][LLC_WAYS];
// PROTECT: 1-bit protection (Evict-Resurrection or multi-hit)
static uint8_t  PROTECT[LLC_SETS][LLC_WAYS];

// Track whether selected victim was valid at selection time (per-set)
static uint8_t  LAST_VICTIM_VALID[LLC_SETS]; // 1-bit logical

// ---------------- Global predictors ----------------
// PC-cold predictor (2-bit saturating counters, 0..3)
static uint8_t PC_CT[PC_SIZE];

// StreamShield tables (PC-indexed)
static uint8_t  ST_LAST_LINE[PC_SIZE]; // low LINE_LOW_BITS of line number
static uint8_t  ST_CONF[PC_SIZE];      // 2-bit confidence (0..3) stored in byte
static uint8_t  ST_LAST_ABS1[PC_SIZE]; // 0/1: last step had |delta|==1

// Evict-Resurrection Bloom filter (2 planes with epoch rotation)
static uint64_t BLOOM[2][BLOOM_WORDS];
static uint8_t  BLOOM_ACTIVE = 0;
static uint32_t fills_since_epoch = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 15) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint8_t line_lowN(uint64_t paddr) {
    return static_cast<uint8_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline uint64_t line_addr(uint64_t paddr) {
    return (paddr >> 6);
}
static inline void sat_inc2(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec1(uint8_t &x) { if (x > 0) x--; }

static inline uint32_t mix32(uint64_t x) {
    // 64->32 bit mix (xorshift/mul)
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}
static inline uint32_t bloom_h1(uint64_t la) { return mix32(la * 0x9e3779b97f4a7c15ULL) & (BLOOM_BITS - 1); }
static inline uint32_t bloom_h2(uint64_t la) { return mix32((la ^ (la >> 7)) * 0xbf58476d1ce4e5b9ULL) & (BLOOM_BITS - 1); }

static inline void bloom_clear_plane(uint8_t plane) {
    for (uint32_t i = 0; i < BLOOM_WORDS; i++) BLOOM[plane][i] = 0ULL;
}
static inline void bloom_rotate_epoch() {
    BLOOM_ACTIVE ^= 1u;
    bloom_clear_plane(BLOOM_ACTIVE);
    fills_since_epoch = 0;
}
static inline void bloom_insert(uint64_t la) {
    uint32_t i1 = bloom_h1(la);
    uint32_t i2 = bloom_h2(la);
    BLOOM[BLOOM_ACTIVE][i1 >> 6] |= (1ULL << (i1 & 63u));
    BLOOM[BLOOM_ACTIVE][i2 >> 6] |= (1ULL << (i2 & 63u));
}
static inline bool bloom_query(uint64_t la) {
    uint32_t i1 = bloom_h1(la);
    uint32_t i2 = bloom_h2(la);
    uint8_t p0 = (BLOOM[0][i1 >> 6] >> (i1 & 63u)) & 1u;
    uint8_t q0 = (BLOOM[0][i2 >> 6] >> (i2 & 63u)) & 1u;
    uint8_t p1 = (BLOOM[1][i1 >> 6] >> (i1 & 63u)) & 1u;
    uint8_t q1 = (BLOOM[1][i2 >> 6] >> (i2 & 63u)) & 1u;
    return ((p0 & q0) | (p1 & q1)) != 0;
}

// Update stream detector and return confidence after this access (demand only)
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint8_t curr = line_lowN(paddr);
    int8_t delta = static_cast<int8_t>(curr - ST_LAST_LINE[idx]);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) sat_inc2(ST_CONF[idx], 3);
        else { if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; ST_LAST_ABS1[idx] = 1; }
    } else {
        sat_dec1(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        LAST_VICTIM_VALID[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]     = AGE_MAX;
            HITCNT[s][w]  = 0;
            STATUS[s][w]  = ST_NORMAL;
            PCSIG[s][w]   = 0;
            PROTECT[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_CT[i]        = PC_INIT;
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    bloom_clear_plane(0);
    bloom_clear_plane(1);
    BLOOM_ACTIVE = 0;
    fills_since_epoch = 0;
}

// Evictability comparator: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict guarded sentinels first (stream/prefetch+WB)
    if (STATUS[set][a] != STATUS[set][b]) return STATUS[set][a] > STATUS[set][b];

    // 2) Avoid evicting protected lines (Evict-Resurrection or multi-hit)
    if (PROTECT[set][a] != PROTECT[set][b]) return PROTECT[set][a] < PROTECT[set][b];

    // 3) Prefer predicted-cold PCs (lower counter more evictable)
    uint8_t pa = PC_CT[ PCSIG[set][a] & (PC_SIZE - 1) ];
    uint8_t pb = PC_CT[ PCSIG[set][b] & (PC_SIZE - 1) ];
    if (pa != pb) return pa < pb;

    // 4) Fewer hits are more evictable
    if (HITCNT[set][a] != HITCNT[set][b]) return HITCNT[set][a] < HITCNT[set][b];

    // 5) Older age is more evictable
    return AGE[set][a] > AGE[set][b];
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
        if (!current_set[w].valid) {
            LAST_VICTIM_VALID[set] = 0; // selecting an invalid way
            return w;
        }
    }

    // Choose most evictable way by composite priority
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    LAST_VICTIM_VALID[set] = 1; // we are evicting a valid line
    return best;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);
    uint8_t stream_conf = 0;
    if (demand) {
        // Update StreamShield on all demand accesses (hit or miss)
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        uint8_t st = STATUS[set][way];

        // Never promote or train sentinels (streams/prefetch/WB)
        if (st == ST_STREAM || st == ST_SENTINEL) {
            return;
        }

        // Multi-hit promotion gating: only 2nd+ demand hit promotes
        if (demand) {
            uint8_t prev = HITCNT[set][way];
            if (prev == 0) {
                // First demand hit: warm PC predictor, no promotion
                uint32_t pidx = pc_index(PC);
                sat_inc2(PC_CT[pidx], 3);
                HITCNT[set][way] = 1;
            } else {
                // 2nd+ demand hit: strong promotion to MRU and protect
                if (HITCNT[set][way] < 3) HITCNT[set][way]++;
                AGE[set][way] = 0;
                PROTECT[set][way] = 1;
            }
        }
        return;
    }

    // MISS/FILL path: train on the evicted line (if any valid victim was present)
    if (LAST_VICTIM_VALID[set]) {
        // Remember victim's line address in Bloom filter (to detect harmful evictions)
        bloom_insert(line_addr(victim_addr));

        // Train PC-cold on dead-on-evict (ignore sentinels)
        if (STATUS[set][way] == ST_NORMAL && HITCNT[set][way] == 0) {
            uint8_t sig = PCSIG[set][way] & (PC_SIZE - 1);
            sat_dec1(PC_CT[sig]);
        }
    }

    // Decide insertion policy for the new line
    uint32_t pidx = pc_index(PC);
    uint8_t st_new = ST_NORMAL;
    uint8_t age_new = AGE_MAX;
    uint8_t protect_new = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on WB; tail insert, no train/promote
        st_new = ST_SENTINEL;
        age_new = AGE_MAX;
    } else if (type == ACCESS_PREFETCH) {
        // Prefetch inserts at tail; never train or promote
        st_new = ST_SENTINEL;
        age_new = AGE_MAX;
    } else if (demand && (stream_conf >= STREAM_CONF_BYPASS)) {
        // StreamShield: near-bypass (tail) for streaming PCs; never train/promote
        st_new = ST_STREAM;
        age_new = AGE_MAX;
    } else {
        // Not a streaming PC; check Evict-Resurrection
        bool resurrect = bloom_query(line_addr(paddr));
        if (resurrect) {
            // Insert young and protect (counteract recent harmful eviction)
            st_new = ST_NORMAL;
            age_new = 1; // gentle young
            protect_new = 1;
            sat_inc2(PC_CT[pidx], 3); // bias PC warmer on resurrection
        } else {
            // PC-based insertion: cold -> old; warm -> higher
            if (PC_CT[pidx] >= PC_WARM_TH) {
                age_new = 1; // warm insertion
            } else {
                age_new = AGE_MAX; // cold insertion
            }
        }
    }

    // Install new line state
    AGE[set][way]     = age_new;
    HITCNT[set][way]  = 0;
    STATUS[set][way]  = st_new;
    PCSIG[set][way]   = static_cast<uint8_t>(pidx);
    PROTECT[set][way] = protect_new;

    // Epoch rotation for Bloom filter based on fills
    fills_since_epoch++;
    if (fills_since_epoch >= EPOCH_PERIOD) {
        bloom_rotate_epoch();
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