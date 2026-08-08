#include <cstdint>
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
#define UTIL_SIZE     256    // PC utility table entries (2-bit counters) - matches 8b per-line PC sig
#define STREAM_SIZE   1024   // stream sentinel entries (PC⊕page)
#define AGE_MAX       3      // 2-bit age: 0..3 (0=young/MRU-like)
#define RUN_MAX       3      // 2-bit stride run counter: 0..3
#define UTIL_MAX      3      // 2-bit utility counter: 0..3
#define UTIL_HOT_THR  2      // >=2 => predicted useful
#define STREAM_THR    2      // >=2 consecutive +1 strides => streamy
#define EPOCH_LEN     32768  // epoch length in LLC accesses (non-WB)

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t mix32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}
static inline uint32_t pc_index(uint64_t PC) {
    return (mix32(PC ^ (PC << 7) ^ (PC >> 11)) & (UTIL_SIZE - 1));
}
static inline uint32_t stream_index(uint64_t PC, uint64_t paddr) {
    uint64_t page = (paddr >> 12);
    uint64_t mix = (PC << 7) ^ (PC >> 11) ^ (page * 0x9e3779b97f4a7c15ULL);
    return (uint32_t)mix & (STREAM_SIZE - 1);
}

// ---------------- Per-line metadata (conceptual packing; see storage section) ----------------
struct LineMeta {
    uint8_t age;     // 2-bit RRPV-like: 0..3
    uint8_t mhits;   // 2-bit demand hit count: 0..3
    uint8_t pf;      // 1-bit: filled by prefetch (quarantine)
    uint8_t stream;  // 1-bit: stream-clamped (no promotion)
    uint8_t pc_sig;  // 8-bit: hashed PC signature (util index)
    uint8_t valid;   // 1-bit: line ever installed (for safe training)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC usefulness predictor with epoch pruning ----------------
static uint8_t pc_util[UTIL_SIZE];     // 2-bit counters
static uint8_t pc_epoch_bit[UTIL_SIZE]; // effectively 1-bit per entry: last touched epoch parity
static uint8_t cur_epoch = 0;
static uint64_t epoch_accesses = 0;

// ---------------- Stream sentinel: PC⊕page -> last line and +1 run ----------------
static uint16_t stream_last_line[STREAM_SIZE]; // low 16 bits of line#
static uint8_t  stream_run[STREAM_SIZE];       // 0..3 run length

// Predict streaminess, then update the sentinel; WRITEBACK does not influence it
static inline bool stream_predict_and_update(uint32_t sidx, uint64_t line, uint32_t type) {
    uint16_t cur = (uint16_t)(line & 0xFFFFu);
    uint16_t prev = stream_last_line[sidx];
    uint16_t delta = (uint16_t)(cur - prev); // modulo 2^16
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

static inline void train_util(uint8_t sig, bool useful) {
    uint8_t &c = pc_util[sig];
    if (useful) {
        if (c < UTIL_MAX) c++;
    } else {
        if (c > 0) c--;
    }
}

// Victim scoring: higher score => better victim
static inline uint32_t score_candidate(const LineMeta &m, bool util_hot_epoch) {
    uint32_t s = 0;
    // Prefer predicted-cold lines
    s += (!util_hot_epoch) ? 32 : 0;
    // Prefer stream-clamped and prefetch lines
    s += m.stream ? 16 : 0;
    s += m.pf ? 8 : 0;
    // Prefer lines with fewer demand hits
    s += (m.mhits == 0) ? 8 : ((m.mhits == 1) ? 2 : 0);
    // Age as tie-breaker
    s += m.age;
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].age = AGE_MAX;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].pc_sig = 0;
            meta[s][w].valid = 0;
        }
    }
    for (uint32_t i = 0; i < UTIL_SIZE; i++) {
        pc_util[i] = UTIL_HOT_THR - 1; // slightly cold start
        pc_epoch_bit[i] = 0;
    }
    for (uint32_t i = 0; i < STREAM_SIZE; i++) {
        stream_last_line[i] = 0;
        stream_run[i] = 0;
    }
    cur_epoch = 0;
    epoch_accesses = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Score all candidates
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        const LineMeta &m = meta[set][w];
        uint8_t sig = m.pc_sig & (UTIL_SIZE - 1);
        bool util_hot_epoch = (pc_util[sig] >= UTIL_HOT_THR) && (pc_epoch_bit[sig] == cur_epoch);
        uint32_t sc = score_candidate(m, util_hot_epoch);
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
    // Epoch maintenance (only count non-WB accesses)
    if (type != WRITEBACK) {
        epoch_accesses++;
        if (epoch_accesses % EPOCH_LEN == 0) {
            cur_epoch ^= 1;
        }
    }

    uint32_t pc_idx = pc_index(PC);
    uint32_t sidx   = stream_index(PC, paddr);
    uint64_t line   = line_number(paddr);

    if (hit) {
        // On hit: update stream sentinel (no effect from WB)
        (void)stream_predict_and_update(sidx, line, type);

        LineMeta &m = meta[set][way];

        // Demand hit: record multi-hit and optional gentle touch
        if (is_demand(type)) {
            // mark PC as touched this epoch (after using it for decisions)
            pc_epoch_bit[pc_idx] = cur_epoch;

            if (m.pf) m.pf = 0; // clear quarantine on first demand use
            if (m.mhits < 3) m.mhits++;

            if (!m.stream) {
                if (m.mhits >= 2) {
                    m.age = 0; // strong promo on confirmed reuse
                } else {
                    // gentle touch on first demand hit (no full promotion)
                    if (m.age > 0) m.age--;
                }
            } else {
                // stream-clamped lines never promote
                m.age = AGE_MAX;
            }
        }
        // No promotion on prefetch hits by design
        return;
    }

    // Miss/Fill path: first train on the evicted line (if valid)
    LineMeta &victim = meta[set][way];
    if (victim.valid) {
        bool useful = (victim.mhits > 0); // any demand reuse during residency
        uint8_t vsig = victim.pc_sig & (UTIL_SIZE - 1);
        train_util(vsig, useful);
    }

    // Stream prediction (updates sentinel) and epoch-pruned utility for current PC
    bool predicted_stream = stream_predict_and_update(sidx, line, type);
    bool pc_touched_epoch = (pc_epoch_bit[pc_idx] == cur_epoch);
    bool util_hot_epoch = (pc_util[pc_idx] >= UTIL_HOT_THR) && pc_touched_epoch;
    bool stream_clamp = predicted_stream && !util_hot_epoch;

    // Insert new line metadata
    LineMeta &m = meta[set][way];
    m.valid = 1;
    m.pc_sig = (uint8_t)(pc_idx & 0xFF);
    m.mhits = 0;
    m.stream = stream_clamp ? 1 : 0;
    m.pf = (type == PREFETCH) ? 1 : 0;

    if (type == WRITEBACK) {
        // writebacks never bypass; moderate insertion
        m.age = (AGE_MAX > 2) ? 2 : AGE_MAX;
    } else if (m.pf) {
        // prefetch quarantine
        m.age = AGE_MAX;
    } else if (stream_clamp) {
        // cold stream -> tail insert and no promotion
        m.age = AGE_MAX;
    } else if (util_hot_epoch) {
        // trusted hot PC this epoch -> favorable insertion
        m.age = 1;
    } else {
        // default conservative insertion
        m.age = AGE_MAX;
    }

    // Finally, mark PC as touched for this epoch if demand
    if (is_demand(type)) {
        pc_epoch_bit[pc_idx] = cur_epoch;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}