#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Design parameters (tuned)
#define SAMPLE_RATE 32               // 3.125% sampled sets
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define SHCT_MAX 7                   // 3-bit SHiP counter
#define LIFE_MAX 3                   // 2-bit lifetime counters
#define STREAM_MAX 3                 // 2-bit stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define FRIEND_THR 3                 // SHCT >= 3 => friendly
#define SHORT_THR 2                  // LIFE_SHORT >= 2 => short
#define LONG_THR 2                   // LIFE_LONG >= 2 => long
#define STREAM_THR 2                 // PC_STREAM >= 2 => scan/streamy
#define DECAY_PERIOD 4096            // faster periodic decay
#define MISS_BURST_THR 64            // burst length that triggers partial decay
#define PARTIAL_DECAY_CHUNK 64       // entries decayed per burst trigger

// Access types (from champsim)
#ifndef LOAD
#define LOAD 0
#endif
#ifndef RFO
#define RFO 1
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif
#ifndef WRITEBACK
#define WRITEBACK 3
#endif

// Per-line metadata (minimized)
struct LineMeta {
    uint8_t rrpv;   // 0..3 (SRRIP)
    uint8_t hits;   // 2-bit saturating: 0..3 (we treat 2+ as multi-hit)
    uint8_t pf;     // prefetch flag
};

static LineMeta meta[LLC_SETS][LLC_WAYS];

// Sampled-set signature table (stores PC index for eviction training), 11-bit index; 0x7FF is invalid
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

// Global per-PC predictors
static uint8_t SHCT[PC_IDX_SIZE];          // 3-bit friendliness predictor
static uint8_t LIFE_SHORT[PC_IDX_SIZE];    // 2-bit short-life score
static uint8_t LIFE_LONG[PC_IDX_SIZE];     // 2-bit long-life score
static uint8_t PC_STREAM[PC_IDX_SIZE];     // 2-bit streaming/scan score
static uint8_t DOA_CONF[PC_IDX_SIZE];      // 2-bit DOA confidence

// Lightweight stride/scan detector state
static uint32_t PC_LAST_LINE[PC_IDX_SIZE]; // last line addr
static int16_t  PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];   // 2-bit run length (0..3)

// Epoch and burst counters for decay
static uint64_t event_ctr = 0;
static uint32_t miss_burst = 0;
static uint32_t decay_ptr = 0;

// Helpers
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix then mask to 11 bits
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}

static inline bool is_sampled(uint32_t set) {
    return (set % SAMPLE_RATE) == 0;
}

static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

// Partial decay helper: decay a chunk of entries to adapt quickly without heavy loops
static inline void decay_some(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = decay_ptr++ & (PC_IDX_SIZE - 1);
        SHCT[idx]       >>= 1;
        LIFE_SHORT[idx] >>= 1;
        LIFE_LONG[idx]  >>= 1;
        PC_STREAM[idx]  >>= 1;
        DOA_CONF[idx]   >>= 1;
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3; // cold
            meta[s][w].hits = 0;
            meta[s][w].pf = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[i] = FRIEND_THR;       // neutral
        LIFE_SHORT[i] = 1;          // slight bias to short
        LIFE_LONG[i] = 1;           // slight bias to long
        PC_STREAM[i] = 1;           // mild stream suspicion
        DOA_CONF[i] = 0;            // start with no DOA confidence
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t s = 0; s < (LLC_SETS / SAMPLE_RATE); s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_sig[s][w] = 0x7FF; // invalid
        }
    }
    event_ctr = 0;
    miss_burst = 0;
    decay_ptr = 0;
}

// Find victim in the set (SRRIP with safe aging)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // SRRIP: look for RRPV==3, else age
    for (int iter = 0; iter < 4; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        // age all ways (saturating at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // Fallback (should not happen)
    return 0;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    event_ctr++;

    LineMeta &m = meta[set][way];

    // Periodic decay to adapt to phase changes
    if ((event_ctr % DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
            SHCT[i]       >>= 1;
            LIFE_SHORT[i] >>= 1;
            LIFE_LONG[i]  >>= 1;
            PC_STREAM[i]  >>= 1;
            DOA_CONF[i]   >>= 1;
        }
    }

    if (hit) {
        // Reset miss burst on a hit
        miss_burst = 0;

        // Multi-hit promotion with first-hit hesitation
        bool is_demand = (type == LOAD) || (type == RFO);
        if (is_demand) {
            // First demand hit on a prefetched line: clear pf, but do not promote
            if (m.pf && m.hits == 0) {
                m.pf = 0;
            }
            if (m.hits < 3) m.hits++;
            if (m.hits >= 2) {
                m.rrpv = 0; // MRU only on second demand hit
            } else {
                // first hit: cap at RRPV=1 (scan resistant)
                if (m.rrpv > 1) m.rrpv = 1;
            }
        }
        return;
    }

    // Miss path
    miss_burst++;
    if (miss_burst >= MISS_BURST_THR) {
        decay_some(PARTIAL_DECAY_CHUNK);
        miss_burst = 0;
    }

    // Train on evicted line if set is sampled
    if (is_sampled(set)) {
        uint32_t sidx = set / SAMPLE_RATE;
        uint16_t old_idx = sample_sig[sidx][way];
        if (old_idx != 0x7FF) {
            uint8_t old_hits = m.hits; // hits accumulated by the victim
            uint8_t was_pf = m.pf;

            if (old_hits == 0) {
                // Dead on arrival
                sat_dec(SHCT[old_idx]); // less friendly
                // Stronger weighting to short-life on 0-hit
                sat_inc(LIFE_SHORT[old_idx], LIFE_MAX);
                sat_inc(LIFE_SHORT[old_idx], LIFE_MAX); // extra bump
                if (LIFE_LONG[old_idx] > 0) LIFE_LONG[old_idx]--;
                // Streaming suspicion up
                sat_inc(PC_STREAM[old_idx], STREAM_MAX);
                // DOA confidence up (dampen if was prefetch)
                if (!was_pf) sat_inc(DOA_CONF[old_idx], DOA_MAX);
                else if (DOA_CONF[old_idx] > 0) DOA_CONF[old_idx]--; // softer on pf
            } else if (old_hits == 1) {
                // Short reuse
                sat_inc(SHCT[old_idx], SHCT_MAX);
                sat_inc(LIFE_SHORT[old_idx], LIFE_MAX);
                if (LIFE_LONG[old_idx] > 0) LIFE_LONG[old_idx]--;
                // Slight stream neutrality
                if (PC_STREAM[old_idx] > 0) PC_STREAM[old_idx]--;
                // Slightly reduce DOA confidence
                if (DOA_CONF[old_idx] > 0) DOA_CONF[old_idx]--;
            } else { // 2+ hits: longer life
                sat_inc(SHCT[old_idx], SHCT_MAX);
                sat_inc(LIFE_LONG[old_idx], LIFE_MAX);
                if (LIFE_SHORT[old_idx] > 0) LIFE_SHORT[old_idx]--;
                if (PC_STREAM[old_idx] > 0) PC_STREAM[old_idx]--;
                // Reduce DOA confidence more aggressively
                if (DOA_CONF[old_idx] > 0) DOA_CONF[old_idx]--;
            }
        }
    }

    // Compute insertion policy for the incoming line
    uint32_t idx = pc_index(PC);
    bool friendly  = (SHCT[idx] >= FRIEND_THR);
    bool longlife  = (LIFE_LONG[idx] >= LONG_THR);
    bool shortlife = (LIFE_SHORT[idx] >= SHORT_THR);
    bool streamy   = (PC_STREAM[idx] >= STREAM_THR);
    bool doa       = (DOA_CONF[idx] >= 2); // confidence-gated DOA

    uint8_t ins_rrpv = 2; // default mid-cold
    if (type == WRITEBACK) {
        // Never bypass on writeback; neutral insertion
        ins_rrpv = 2;
    } else {
        if (doa || (streamy && !friendly) || shortlife) {
            ins_rrpv = 3; // soft-bypass
        } else if (friendly && longlife && !streamy) {
            ins_rrpv = 1; // warmer insertion
        } else {
            ins_rrpv = 2;
        }
        if (type == PREFETCH && ins_rrpv < 3) {
            ins_rrpv++; // prefetch lines insert colder by one
        }
    }

    // Install metadata for the filled line
    m.rrpv = ins_rrpv;
    m.hits = 0;
    m.pf   = (type == PREFETCH) ? 1 : 0;

    // Record PC signature for training if set is sampled
    if (is_sampled(set)) {
        uint32_t sidx = set / SAMPLE_RATE;
        sample_sig[sidx][way] = static_cast<uint16_t>(idx & 0x7FF);
    }

    // Update stride-based scan detector on every miss
    uint64_t line = (paddr >> 6);
    int32_t stride = static_cast<int32_t>(line - PC_LAST_LINE[idx]);
    if (stride == PC_LAST_STRIDE[idx] && (stride <= 2 && stride >= -2)) {
        if (PC_STR_RUN[idx] < 3) PC_STR_RUN[idx]++;
    } else {
        PC_STR_RUN[idx] = 1;
    }
    if (PC_STR_RUN[idx] >= 3) {
        sat_inc(PC_STREAM[idx], STREAM_MAX);
    } else {
        sat_dec(PC_STREAM[idx]);
    }
    PC_LAST_STRIDE[idx] = static_cast<int16_t>(stride);
    PC_LAST_LINE[idx] = static_cast<uint32_t>(line);
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}