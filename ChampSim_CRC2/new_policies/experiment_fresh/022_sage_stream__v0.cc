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
#define PC_IDX_SIZE 64            // 64-entry PC classifier/sentinel (6-bit per-line pc_sig)
#define LAST_LINE_BITS 12         // low bits of last line number for stride (+1) sentinel
#define SAMPLE_RATE_SHIFT 5       // 1/32 sampled sets for learner training

// ---------------- Per-line metadata (conceptual bit budget: 14 bits/line) ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (4 bits) recency stack position
    uint8_t mhits;    // 0..3 (2 bits) demand-hit count (promotion gated until >=2)
    uint8_t pf;       // 0/1 (1 bit) filled by prefetch
    uint8_t stream;   // 0/1 (1 bit) stream-clamped: never promote
    uint8_t pc_sig;   // 0..63 (6 bits) PC index at fill for training
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC-based learner: 2-bit keep/evict classifier ----------------
static uint8_t pc_keep[PC_IDX_SIZE]; // 0=strong-evict, 3=strong-keep

// ---------------- PC stride(+1) streaming sentinel ----------------
static uint16_t pc_last_line[PC_IDX_SIZE]; // LAST_LINE_BITS of last line#
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3 consecutive +1 stride run

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline uint32_t pc_index(uint64_t pc) {
    // lightweight mix; PC_IDX_SIZE is power-of-two
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 17) ^ (pc << 9);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMPLE_RATE_SHIFT) - 1)) == 0);
}

// Stride(+1) streaming sentinel; updates state and returns streaming prediction
static inline bool stream_predict_and_update(uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;
    uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
    uint16_t cur = static_cast<uint16_t>(line) & mask;
    uint16_t prev = pc_last_line[pc_idx];
    uint16_t delta = static_cast<uint16_t>((cur - prev) & mask);

    if (delta == 1) {
        if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
    } else if (cur != prev) {
        if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
    }
    pc_last_line[pc_idx] = cur;
    return (pc_run[pc_idx] >= 2); // 2+ steps of +1 stride => confident stream
}

// LRU helpers
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru < old) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = 0;
}
static inline void lru_insert_pos(uint32_t set, uint32_t way, uint8_t pos) {
    if (pos >= LLC_WAYS) pos = LLC_WAYS - 1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru <= pos) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = pos;
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                        // older preferred
    s += (m.mhits == 0) ? 8 : 0;       // single/no-hit preferred
    s += m.stream ? 6 : 0;             // stream-clamped preferred
    s += m.pf ? 4 : 0;                 // prefetch quarantined
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;     // stack order
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].pc_sig = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_keep[i] = 1;             // slightly biased to evict until proven
        pc_last_line[i] = 0;
        pc_run[i] = 0;
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
    // Prefer invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Score-based victim among valid ways
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (w == 0 || sc > best_score) {
            best_score = sc;
            best_way = w;
        }
    }

    // Sampled training on imminent eviction outcome (dead vs reused)
    if (is_sampled_set(set)) {
        const LineMeta &m = meta[set][best_way];
        uint8_t &ctr = pc_keep[m.pc_sig];
        if (m.mhits == 0) { // evicted before any reuse
            if (ctr > 0) ctr--;
        } else {            // observed reuse
            if (ctr < 3) ctr++;
        }
    }

    return best_way; // replaced block index
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
    const bool sampled = is_sampled_set(set);
    LineMeta &m = meta[set][way];

    if (hit) {
        // On hit: multi-hit gating and stream clamp
        if (is_demand(type)) {
            if (m.mhits < 3) m.mhits++;
            if (!m.stream) {
                if (m.mhits >= 2) {
                    lru_make_mru(set, way); // promote only after second demand hit
                }
                // else: first demand hit does not promote
            }
            // Train keep on observed reuse (fill PC signature)
            if (sampled) {
                uint8_t &ctr = pc_keep[m.pc_sig];
                if (ctr < 3) ctr++;
            }
        } else {
            // prefetch hit: no promotion, no training
        }
        return;
    }

    // Miss: determine insertion policy
    const uint32_t pc_idx = pc_index(PC);
    const uint64_t line = line_number(paddr);
    const bool stream_pred = stream_predict_and_update(pc_idx, line, type);
    const uint8_t keep_conf = pc_keep[pc_idx];

    m.pc_sig = static_cast<uint8_t>(pc_idx);
    m.pf = (type == PREFETCH) ? 1 : 0;
    m.stream = stream_pred ? 1 : 0;
    m.mhits = 0; // fresh line, no demand hits yet

    uint8_t ins_pos = LLC_WAYS - 1; // default tail

    if (type == WRITEBACK) {
        // never bypass writebacks; keep low priority
        ins_pos = LLC_WAYS - 2;
        m.pf = 0;
        m.stream = 0;
    } else if (type == PREFETCH) {
        // quarantine prefetches at tail
        ins_pos = LLC_WAYS - 1;
    } else {
        // demand: combine stream and PC learner
        if (stream_pred) {
            // confident stream: aggressive tail insert, never promote
            ins_pos = LLC_WAYS - 1;
        } else if (keep_conf <= 1) {
            // predicted cold: near-tail insert
            ins_pos = LLC_WAYS - 2;
        } else {
            // predicted keep: middle insertion to preserve useful recency
            ins_pos = LLC_WAYS / 2;
        }
    }

    lru_insert_pos(set, way, ins_pos);
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}