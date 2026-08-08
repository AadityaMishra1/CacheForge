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

// ---------------- Per-line metadata (packed conceptually: 6 bits) ----------------
// refcnt: 2-bit CLOCK use counter (0..3)
// friendly: 1-bit class (0=averse, 1=friendly)
// hitcnt: 2-bit capped hit counter: 0=none,1=one hit,2=2+ hits
// no_train: 1-bit sentinel (streams/prefetch/wb): no-train/no-promote
static uint8_t refcnt[LLC_SETS][LLC_WAYS];
static uint8_t friendly[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t no_train[LLC_SETS][LLC_WAYS];

// ---------------- Global PC-classifier (Hawkeye-like) ----------------
static constexpr uint32_t PC_TABLE_SIZE = 2048; // 2K entries, 3-bit each
static uint8_t PCCTR[PC_TABLE_SIZE];            // 0..7, init ~neutral

// ---------------- StreamGuard: per-PC stride detector ----------------
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;
static uint32_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // 64B line number
static int32_t  ST_LAST_DELTA[STREAM_TABLE_SIZE];
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit (0..3)

// ---------------- Sampled OPTgen: 5% sets, compact recency stack ----------------
static constexpr uint32_t SAMPLE_MOD = 20; // ~5.0% -> 103 of 2048 sets
static constexpr uint32_t REUSE_STACK_LEN = 32; // recency window
static constexpr uint32_t SAMPLED_SETS = (LLC_SETS / SAMPLE_MOD) + 1;
static uint32_t SAMPLE_STACK[SAMPLED_SETS][REUSE_STACK_LEN];
static uint8_t  SAMPLE_COUNT[SAMPLED_SETS]; // 0..32

// ---------------- Helpers ----------------
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x)               { if (x > 0) x--; }

static inline uint32_t pc_hash(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_TABLE_SIZE - 1);
}
static inline uint32_t stream_hash(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint32_t line_number(uint64_t paddr) { return static_cast<uint32_t>(paddr >> 6); }

static inline bool is_sampled_set(uint32_t set) { return (set % SAMPLE_MOD) == 0; }
static inline uint32_t sampled_index(uint32_t set) { return set / SAMPLE_MOD; }

static inline uint32_t hash_line_id(uint64_t paddr) {
    uint64_t ln = (paddr >> 6);
    uint64_t h = ln ^ (ln >> 7) ^ (ln >> 17) ^ 0x9e3779b97f4a7c15ULL;
    return static_cast<uint32_t>(h); // 32-bit token
}

// StreamGuard update on demand access; returns confidence after update
static inline uint8_t update_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_hash(pc);
    uint32_t curr = line_number(paddr);
    int32_t delta = static_cast<int32_t>(curr - ST_LAST_LINE[idx]);

    if (delta != 0) {
        if (delta == ST_LAST_DELTA[idx]) {
            sat_inc(ST_CONF[idx], 3); // consecutive same-delta step
        } else {
            ST_CONF[idx] = 1;         // first observation of new stride
            ST_LAST_DELTA[idx] = delta;
        }
        ST_LAST_LINE[idx] = curr;
    } else {
        // same line: do not strengthen confidence; mild decay
        sat_dec(ST_CONF[idx]);
    }
    return ST_CONF[idx];
}

// Train OPTgen-style PC classifier on sampled sets using approximate stack distance
static inline void optgen_train(uint32_t set, uint64_t paddr, uint64_t pc, bool is_demand) {
    if (!is_demand) return;
    if (!is_sampled_set(set)) return;

    uint32_t sidx = sampled_index(set);
    uint32_t id   = hash_line_id(paddr);
    uint8_t  &cnt = PCCTR[pc_hash(pc)];

    // Search in stack
    int found = -1;
    uint8_t n = SAMPLE_COUNT[sidx];
    for (uint8_t i = 0; i < n; i++) {
        if (SAMPLE_STACK[sidx][i] == id) { found = i; break; }
    }

    if (found >= 0) {
        // If reuse distance within associativity => friendly, else averse
        if (found < LLC_WAYS) sat_inc(cnt, 7); else sat_dec(cnt);

        // Move-to-front (MRU)
        uint32_t tmp = SAMPLE_STACK[sidx][found];
        for (int j = found; j > 0; j--) SAMPLE_STACK[sidx][j] = SAMPLE_STACK[sidx][j-1];
        SAMPLE_STACK[sidx][0] = tmp;
    } else {
        // Cold/new => likely averse
        sat_dec(cnt);
        // Insert at MRU
        if (n < REUSE_STACK_LEN) {
            for (int j = n; j > 0; j--) SAMPLE_STACK[sidx][j] = SAMPLE_STACK[sidx][j-1];
            SAMPLE_STACK[sidx][0] = id;
            SAMPLE_COUNT[sidx] = n + 1;
        } else {
            for (int j = REUSE_STACK_LEN - 1; j > 0; j--) SAMPLE_STACK[sidx][j] = SAMPLE_STACK[sidx][j-1];
            SAMPLE_STACK[sidx][0] = id;
        }
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            refcnt[s][w]   = 0;
            friendly[s][w] = 0;
            hitcnt[s][w]   = 0;
            no_train[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TABLE_SIZE; i++) PCCTR[i] = 3; // neutral
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_LAST_DELTA[i] = 0;
        ST_CONF[i]       = 0;
    }
    for (uint32_t i = 0; i < SAMPLED_SETS; i++) {
        SAMPLE_COUNT[i] = 0;
        for (uint32_t j = 0; j < REUSE_STACK_LEN; j++) SAMPLE_STACK[i][j] = 0;
    }
}

// Find victim in the set: CLOCK-like with averse-first preference
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Two passes: averse first (friendly==0), then friendly
    for (int pass = 0; pass < 2; pass++) {
        // Bounded rounds of aging to ensure termination
        for (int round = 0; round < 4; round++) {
            for (uint32_t w = 0; w < LLC_WAYS; w++) {
                if ((friendly[set][w] == static_cast<uint8_t>(pass)) && (refcnt[set][w] == 0)) {
                    return w;
                }
            }
            // Age only the class in this pass
            for (uint32_t w = 0; w < LLC_WAYS; w++) {
                if (friendly[set][w] == static_cast<uint8_t>(pass)) {
                    if (refcnt[set][w] > 0) refcnt[set][w]--;
                }
            }
        }
    }

    // Fallback: pick the way with smallest refcnt, prefer averse
    uint32_t best_w = 0;
    uint8_t  best_score = 255;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint8_t score = (friendly[set][w] ? 8 : 0) + refcnt[set][w]; // averse preferred
        if (score < best_score) { best_score = score; best_w = w; }
    }
    return best_w;
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
    const bool is_demand = (type == ACCESS_LOAD) || (type == ACCESS_RFO);

    // Update stream detector and OPTgen trainer (demand only)
    uint8_t stream_conf = 0;
    if (is_demand) {
        stream_conf = update_stream(PC, paddr);
        optgen_train(set, paddr, PC, true);
    }

    // HIT behavior
    if (hit) {
        // No-op for sentinels (streams/prefetch/wb)
        if (no_train[set][way]) return;

        // Multi-hit promotion: first hit warms; second hit -> MRU-equivalent and friendly
        if (hitcnt[set][way] == 0) {
            hitcnt[set][way] = 1;
            if (refcnt[set][way] < 3) refcnt[set][way]++; // mild bump
        } else {
            hitcnt[set][way] = 2;
            refcnt[set][way] = 3;      // strong promotion
            friendly[set][way] = 1;    // proven live
        }
        return;
    }

    // MISS insertion policy
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on WB: low priority, sentinel (no train/promote)
        refcnt[set][way]   = 0;
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 1;
        return;
    }
    if (type == ACCESS_PREFETCH) {
        // Prefetches: low priority, no train/promote
        refcnt[set][way]   = 0;
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 1;
        return;
    }

    // Demand insertions
    if (stream_conf >= 2) {
        // StreamGuard: effective bypass via low-priority sentinel
        refcnt[set][way]   = 0;
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 1;
        return;
    } else if (stream_conf == 1) {
        // Early stream hint: also lowest priority, sentinel
        refcnt[set][way]   = 0;
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 1;
        return;
    }

    // PC-based decision (OPTgen-trained)
    uint8_t c = PCCTR[pc_hash(PC)];
    // thresholds: <=2 averse, 3-4 neutral, >=5 friendly
    if (c >= 5) {
        refcnt[set][way]   = 2; // favorable insertion
        friendly[set][way] = 1;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 0;
    } else if (c <= 2) {
        refcnt[set][way]   = 0; // far insertion
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 0;
    } else {
        refcnt[set][way]   = 1; // neutral
        friendly[set][way] = 0;
        hitcnt[set][way]   = 0;
        no_train[set][way] = 0;
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