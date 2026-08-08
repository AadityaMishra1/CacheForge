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

// ---------------- Tunables (SAGE-Stream++) ----------------
#define PC_IDX_SIZE 128          // expanded learner/stream sentinel
#define LAST_LINE_BITS 12        // track low bits of line# for stride(+1) detection
#define SAMPLE_RATE_SHIFT 5      // 1/32 sampled sets for learner training
#define DECAY_INTERVAL (1u << 16) // periodic learner decay cadence (in updates)

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (4 bits) recency stack
    uint8_t mhits;    // 0..3 (2 bits) demand-hit count
    uint8_t pf;       // 0/1 (1 bit) filled by prefetch
    uint8_t stream;   // 0/1 (1 bit) stream-clamped: never promote
    uint8_t pc_sig;   // 0..127 (7 bits) PC index at fill for training
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC learner: 2-bit keep/evict ----------------
static uint8_t pc_keep[PC_IDX_SIZE]; // 0=strong-evict, 3=strong-keep

// ---------------- Per-PC stride(+1) streaming sentinel ----------------
static uint16_t pc_last_line[PC_IDX_SIZE]; // track low LAST_LINE_BITS of last line#
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3 run-length of +1 strides

// ---------------- Global periodic decay ----------------
static uint32_t update_count = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline uint32_t pc_index(uint64_t pc) {
    // lightweight mix into power-of-two table
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 17) ^ (pc << 9);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMPLE_RATE_SHIFT) - 1)) == 0);
}

// Stride(+1) stream sentinel: update and return current run length (0..3)
static inline uint8_t stream_run_and_update(uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return pc_run[pc_idx]; // do not learn on writebacks
    uint16_t mask = static_cast<uint16_t>((1u << LAST_LINE_BITS) - 1u);
    uint16_t cur = static_cast<uint16_t>(line) & mask;
    uint16_t prev = pc_last_line[pc_idx];
    uint16_t delta = static_cast<uint16_t>((cur - prev) & mask);

    if (delta == 1) {
        if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
    } else if (cur != prev) {
        if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
    }
    pc_last_line[pc_idx] = cur;
    return pc_run[pc_idx];
}

// LRU helpers
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru < old) {
            uint8_t np = static_cast<uint8_t>(meta[set][w].lru + 1);
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
            uint8_t np = static_cast<uint8_t>(meta[set][w].lru + 1);
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = pos;
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                          // recency
    s += (m.mhits == 0) ? 32 : 0;        // strongly prefer zero-hit
    s += m.stream ? 24 : 0;              // stream-clamped
    s += m.pf ? 16 : 0;                  // prefetch quarantine
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = static_cast<uint8_t>(w);
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].pc_sig = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_keep[i] = 1;      // slight bias to evict until proven
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    update_count = 0;
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

    // Sampled training on imminent eviction outcome (dead vs. reused)
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

    // Update stream sentinel per access (except writeback)
    uint32_t pc_idx = pc_index(PC);
    uint64_t ln = line_number(paddr);
    uint8_t run = stream_run_and_update(pc_idx, ln, type);

    if (hit) {
        // On hit: demand-hit gating and stream clamp
        if (is_demand(type)) {
            if (m.mhits < 3) m.mhits++;

            // Train positively on confirmed multi-hit in sampled sets
            if (sampled && m.mhits >= 2) {
                if (pc_keep[m.pc_sig] < 3) pc_keep[m.pc_sig]++;
            }

            // Clear prefetch quarantine once confirmed reuse
            if (m.mhits >= 2 && m.pf) m.pf = 0;

            // Promotion policy: never promote stream-clamped or before 2nd demand hit
            if (!m.stream && m.mhits >= 2 && m.pf == 0) {
                lru_make_mru(set, way);
            } // else: keep position (no-op) to avoid pollution
        } else {
            // Non-demand hits (should be rare): keep conservative
            // do not promote
        }
    } else {
        // Miss: new fill handling (never bypass on WRITEBACK)
        m.mhits = 0;
        m.pf = (type == PREFETCH) ? 1 : 0;
        m.stream = 0;
        m.pc_sig = static_cast<uint8_t>(pc_idx);

        // Learner prediction
        uint8_t pred = pc_keep[pc_idx]; // 0..3

        // Streaming decisions: 3+ => strongest streaming (soft-bypass via tail); 2+ => stream clamp
        bool strong_stream = (run >= 3);
        bool streamish = (run >= 2);

        uint8_t ins_pos = LLC_WAYS - 1; // default tail

        if (type == WRITEBACK) {
            // Respect rule: never bypass on writeback; insert conservatively mid
            ins_pos = LLC_WAYS - 4; // somewhat old
        } else if (strong_stream) {
            // Strong sequential: tail-insert and clamp
            m.stream = 1;
            ins_pos = LLC_WAYS - 1;
        } else if (streamish) {
            // Stream clamp: tail-insert and never promote
            m.stream = 1;
            ins_pos = LLC_WAYS - 1;
        } else if (type == PREFETCH) {
            // Prefetch quarantine: very low priority insertion
            ins_pos = LLC_WAYS - 2;
        } else {
            // Demand insert guided by PC learner
            if (pred >= 2) {
                // hot PC: shallow insert near MRU (but not MRU)
                ins_pos = 3;
            } else if (pred == 1) {
                // neutral PC: mid-depth
                ins_pos = LLC_WAYS / 2;
            } else {
                // cold PC: tail to minimize pollution
                ins_pos = LLC_WAYS - 1;
            }
        }

        lru_insert_pos(set, way, ins_pos);
    }

    // Periodic global decay to adapt across phases
    update_count++;
    if ((update_count & (DECAY_INTERVAL - 1)) == 0) {
        for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
            if (pc_keep[i] > 0) pc_keep[i]--; // drift toward "evict" unless reinforced
        }
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