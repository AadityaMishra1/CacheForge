#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

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

// ---------------- Tunables ----------------
#define UTIL_SIZE    1024   // PC utility table entries (2-bit counters)
#define STREAM_SIZE  1024   // stream sentinel entries (last line + run)
#define AGE_MAX      3      // 2-bit age: 0..3 (0=young/MRU-like)
#define RUN_MAX      3      // 2-bit stride run counter: 0..3
#define UTIL_MAX     3      // 2-bit utility counter: 0..3
#define UTIL_HOT_THR 2      // >=2 => predicted useful
#define STREAM_THR   2      // >=2 consecutive +1 strides => streamy

// ---------------- Per-line metadata (conceptually packed; see storage section) ----------------
struct LineMeta {
    uint8_t age;     // 2-bit: 0..3 (insertion at tail=3)
    uint8_t mhits;   // 2-bit: demand hit count during residency (0..3)
    uint8_t pf;      // 1-bit: filled by prefetch
    uint8_t stream;  // 1-bit: stream-clamped (no promotion)
    uint8_t pc_sig;  // 8-bit: hashed PC signature (for training/admission)
    uint8_t valid;   // 1-bit: our notion of validity (for training on eviction)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC utility (usefulness) predictor ----------------
static uint8_t pc_util[UTIL_SIZE]; // 2-bit counters, 0..3

// ---------------- Stream sentinel: PC⊕page -> last line (low 16b) and +1 run ----------------
static uint16_t stream_last_line[STREAM_SIZE]; // low 16 bits of line#
static uint8_t  stream_run[STREAM_SIZE];       // 0..3 run length

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }

static inline uint32_t hash_mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}
static inline uint32_t pc_index(uint64_t PC) {
    return (hash_mix64(PC) ^ (PC >> 2) ^ (PC << 13)) & (UTIL_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t PC, uint64_t paddr) {
    uint64_t page = (paddr >> 12);
    uint64_t mix = (PC << 7) ^ (PC >> 11) ^ (page * 0x9e3779b97f4a7c15ULL);
    return static_cast<uint32_t>(mix) & (STREAM_SIZE - 1);
}
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

// Predict streaminess then update the sentinel; WRITEBACK does not influence it
static inline bool stream_predict_and_update(uint32_t sidx, uint64_t line, uint32_t type) {
    uint16_t cur = static_cast<uint16_t>(line & 0xFFFFu);
    uint16_t prev = stream_last_line[sidx];
    uint16_t delta = static_cast<uint16_t>(cur - prev); // modulo 2^16
    bool predicted_stream = (stream_run[sidx] >= STREAM_THR);

    if (type != WRITEBACK) {
        if (delta == 1) {
            if (stream_run[sidx] < RUN_MAX) stream_run[sidx]++;
        } else if (cur != prev) {
            if (stream_run[sidx] > 0) stream_run[sidx]--;
        }
        stream_last_line[sidx] = cur;
    }
    return predicted_stream;
}

// Victim scoring: higher score => better victim
static inline uint32_t score_candidate(const LineMeta &m, uint8_t util_pred) {
    uint32_t s = 0;
    // Prefer predicted-useless lines
    s += (util_pred < UTIL_HOT_THR) ? 16 : 0;
    // Prefer lines with no demand hits
    s += (m.mhits == 0) ? 8 : ((m.mhits == 1) ? 2 : 0);
    // Prefer stream-clamped and prefetch lines
    s += m.stream ? 6 : 0;
    s += m.pf ? 4 : 0;
    // Age as tie-breaker
    s += m.age;
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].age = AGE_MAX;   // cold at start
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].pc_sig = 0;
            meta[s][w].valid = 0;
        }
    }
    for (uint32_t i = 0; i < UTIL_SIZE; i++) pc_util[i] = UTIL_HOT_THR - 1; // slightly cold start
    for (uint32_t i = 0; i < STREAM_SIZE; i++) {
        stream_last_line[i] = 0;
        stream_run[i] = 0;
    }
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Score all candidates; prefer predicted-useless with additional low-reuse signals
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        const LineMeta &m = meta[set][w];
        uint8_t util = pc_util[m.pc_sig & (UTIL_SIZE - 1)];
        uint32_t sc = score_candidate(m, util);
        if (w == 0 || sc > best_score) {
            best_score = sc;
            best_way = w;
        }
    }
    return best_way;
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
    uint32_t pc_idx = pc_index(PC);
    uint32_t sidx   = stream_index(PC, paddr);
    uint64_t line   = line_number(paddr);

    if (hit) {
        // On hit: update stream sentinel and apply multi-hit promotion gating
        (void)stream_predict_and_update(sidx, line, type);

        LineMeta &m = meta[set][way];

        if (is_demand(type)) {
            // demand hit: clear prefetch quarantine on first demand use
            if (m.pf) m.pf = 0;

            // record a demand hit (bounded)
            if (m.mhits < 3) m.mhits++;

            // stream-clamped lines never promote
            if (!m.stream) {
                if (m.mhits >= 2) {
                    // strong promotion on second demand hit
                    m.age = 0;
                    // credit usefulness to the fill PC signature
                    uint8_t &u = pc_util[m.pc_sig & (UTIL_SIZE - 1)];
                    if (u < UTIL_MAX) u++;
                } else {
                    // soft promotion: do not jump to 0 on first hit
                    if (m.age > 0) m.age--;
                }
            }
        } else {
            // prefetch "hit": do not promote
            (void)0;
        }
        return;
    }

    // MISS path: choose insertion policy, train on eviction, then install new metadata
    // Train evicted line's fill PC usefulness using its residency outcome
    LineMeta old = meta[set][way]; // snapshot before overwrite
    if (old.valid) {
        uint8_t &u = pc_util[old.pc_sig & (UTIL_SIZE - 1)];
        if (old.mhits == 0) { // useless
            if (u > 0) u--;
        } else {              // useful
            if (u < UTIL_MAX) u++;
        }
    }

    // Predict streaminess using state before this access and update sentinel
    bool is_stream = stream_predict_and_update(sidx, line, type);

    // Prepare new line metadata
    LineMeta nw;
    nw.valid  = 1;
    nw.pc_sig = static_cast<uint8_t>(pc_idx & 0xFF);
    nw.mhits  = 0;
    nw.stream = is_stream ? 1 : 0;
    nw.pf     = (type == PREFETCH) ? 1 : 0;

    if (type == WRITEBACK) {
        // never bypass writebacks; moderate priority insert
        nw.age = (AGE_MAX > 1) ? (AGE_MAX - 1) : AGE_MAX;
        nw.stream = 0;
        nw.pf = 0;
    } else if (type == PREFETCH) {
        // quarantine prefetches at tail
        nw.age = AGE_MAX;
    } else {
        // demand miss: admission based on usefulness and stream gate
        uint8_t util = pc_util[pc_idx];
        if (is_stream) {
            // emulate bypass: tail insert, clamp promotions
            nw.age = AGE_MAX;
        } else if (util < UTIL_HOT_THR) {
            // predicted useless: tail insert
            nw.age = AGE_MAX;
        } else {
            // predicted useful: modestly near-MRU insert
            nw.age = (AGE_MAX > 1) ? (AGE_MAX - 1) : AGE_MAX;
        }
    }

    meta[set][way] = nw;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}