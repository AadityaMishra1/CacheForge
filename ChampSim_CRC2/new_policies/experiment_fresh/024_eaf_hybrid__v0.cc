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
#define PC8_SIZE 256                 // 8-bit PC signature table for coldness (2-bit counters)
#define STREAM_PC_SIZE 1024          // per-PC stream sentinel entries (12-bit last_line + 2-bit run)
#define LAST_LINE_BITS 12            // track low 12 bits of line number for stride detection
#define SAMPLE_STRIDE 8              // 1/8 set sampling for per-line fill PC signatures
#define SAMPLE_SETS (LLC_SETS / SAMPLE_STRIDE)
#define DECAY_PERIOD 2048            // periodic decay cadence for tables and set pressure

// ---------------- Per-line metadata (packed conceptually to 10 bits) ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (recency bucket; 0=MRU)
    uint8_t ttl;      // 0..3 (survival priority)
    uint8_t mhits;    // 0..3 (demand-hit count; multi-hit gating)
    uint8_t pf;       // 0/1 (filled by prefetch)
    uint8_t stream;   // 0/1 (stream-clamped: never promote)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Sampled per-line fill PC signatures (8-bit) ----------------
static uint8_t fill_pc8[SAMPLE_SETS][LLC_WAYS]; // only for sampled sets

// ---------------- PC coldness predictor (2-bit per 8-bit PC signature) ----------------
static uint8_t pc_cold[PC8_SIZE]; // 0=hot ... 3=strong cold

// ---------------- Per-PC streaming sentinel ----------------
static uint16_t pc_last_line[STREAM_PC_SIZE]; // LAST_LINE_BITS of line#
static uint8_t  pc_run[STREAM_PC_SIZE];       // 0..3 forward +1 stride run

// ---------------- Per-set stream pressure (0..3) ----------------
static uint8_t set_pressure[LLC_SETS];

// ---------------- Transient decision (per-set) ----------------
static uint8_t pending_stream_ins[LLC_SETS];  // 0/1: insert as stream-clamped

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }

static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint8_t pc_sig8(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 17) ^ (pc << 13);
    x ^= (x >> 7);
    return static_cast<uint8_t>(x & 0xFF);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc << 3);
    return static_cast<uint32_t>(x) & (STREAM_PC_SIZE - 1);
}
static inline int32_t sample_index(int32_t set) {
    return ((set & (SAMPLE_STRIDE - 1)) == 0) ? (set / SAMPLE_STRIDE) : -1;
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

// Victim scoring (higher is better victim)
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                         // older preferred
    s += (3 - m.ttl) * 6;               // low TTL preferred
    s += (m.mhits == 0) ? 7 : ((m.mhits == 1) ? 3 : 0); // zero/single-hit preferred
    s += m.pf ? 5 : 0;                  // prefetch quarantined
    s += m.stream ? 7 : 0;              // stream-clamped preferred
    return s;
}

// Lookahead stream prediction (does not mutate state)
static inline bool predict_stream(uint32_t pc_idx, uint64_t line_low, uint8_t set_press, uint32_t type) {
    if (type == WRITEBACK) return false;
    uint16_t prev = pc_last_line[pc_idx];
    uint8_t run  = pc_run[pc_idx];
    uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
    uint16_t delta = static_cast<uint16_t>((line_low - prev) & mask);
    uint8_t run_next = run;
    if (delta == 1) {
        if (run_next < 3) run_next++;
    } else if (line_low != prev) {
        if (run_next > 0) run_next--;
    }
    return (run_next >= 2) || (set_press >= 2);
}

// periodic light decay for pc_cold, stream runs, and set pressure
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_cold
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t idx = (decay_ptr + i) & (PC8_SIZE - 1);
        if (pc_cold[idx] > 0) pc_cold[idx]--;
    }
    // Bleed stream runs sparsely
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr + (i * 3)) & (STREAM_PC_SIZE - 1);
        if (pc_run[idx] > 0) pc_run[idx]--;
    }
    // Bleed set pressure sparsely
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t s = (decay_ptr + (i * 7)) & (LLC_SETS - 1);
        if (set_pressure[s] > 0) set_pressure[s]--;
    }
    decay_ptr++;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;
            meta[s][w].ttl = 1;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
        }
        set_pressure[s] = 0;
        pending_stream_ins[s] = 0;
    }
    for (uint32_t i = 0; i < PC8_SIZE; i++) pc_cold[i] = 1; // slightly cold start
    for (uint32_t i = 0; i < STREAM_PC_SIZE; i++) {
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    for (uint32_t s = 0; s < SAMPLE_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) fill_pc8[s][w] = 0;
    }
    event_ctr = 0;
    decay_ptr = 0;
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
    // Prefer any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            // still compute stream decision to guide insertion
            uint32_t pc_idx = stream_index(PC);
            uint64_t ln = line_number(paddr);
            uint16_t cur = static_cast<uint16_t>(ln & ((1u << LAST_LINE_BITS) - 1u));
            pending_stream_ins[set] = predict_stream(pc_idx, cur, set_pressure[set], type) ? 1 : 0;
            return w;
        }
    }

    // Predict streaming for this access to decide insertion behavior
    uint32_t pc_idx = stream_index(PC);
    uint64_t ln = line_number(paddr);
    uint16_t cur = static_cast<uint16_t>(ln & ((1u << LAST_LINE_BITS) - 1u));
    pending_stream_ins[set] = predict_stream(pc_idx, cur, set_pressure[set], type) ? 1 : 0;

    // Score-based victim among valid ways
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sscore = score_line(meta[set][w]);
        if (w == 0 || sscore > best_score) {
            best_score = sscore;
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
    event_ctr++;
    periodic_decay();

    if (hit) {
        // Demand hits train and may promote
        if (is_demand(type)) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;

            // Only promote strongly after second demand hit and if not stream-clamped
            if (meta[set][way].stream == 0 && meta[set][way].pf == 0 && meta[set][way].mhits >= 2) {
                meta[set][way].ttl = 3;
                lru_make_mru(set, way);
            } else {
                // gentle nudge on first demand hit if not stream-clamped
                if (meta[set][way].stream == 0) {
                    uint8_t cur = meta[set][way].lru;
                    uint8_t target = (cur > 2) ? (cur - 2) : 0;
                    lru_insert_pos(set, way, target);
                }
            }
            // After two demand hits on a prefetch line, clear pf tag
            if (meta[set][way].pf && meta[set][way].mhits >= 2) meta[set][way].pf = 0;
        } else {
            // Non-demand hits: minimal movement
            uint8_t cur = meta[set][way].lru;
            uint8_t target = (cur > 1) ? (cur - 1) : 0;
            lru_insert_pos(set, way, target);
        }
        return;
    }

    // Miss path: train PC coldness using evicted line (sampled sets only)
    int32_t si = sample_index(static_cast<int32_t>(set));
    if (si >= 0) {
        uint8_t victim_pc8 = fill_pc8[si][way];
        // If victim had zero demand hits or was a pure prefetch, punish; if >=2 hits, reward
        if ((meta[set][way].mhits == 0) || (meta[set][way].pf != 0)) {
            if (pc_cold[victim_pc8] < 3) pc_cold[victim_pc8]++;
        } else if (meta[set][way].mhits >= 2) {
            if (pc_cold[victim_pc8] > 0) pc_cold[victim_pc8]--;
        }
    }

    // Update stream sentinel and set pressure with the actual access
    uint32_t pc_idx = stream_index(PC);
    uint64_t ln = line_number(paddr);
    uint16_t cur = static_cast<uint16_t>(ln & ((1u << LAST_LINE_BITS) - 1u));
    uint16_t prev = pc_last_line[pc_idx];
    uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
    uint16_t delta = static_cast<uint16_t>((cur - prev) & mask);
    if (type != WRITEBACK) {
        if (delta == 1) {
            if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
            if (set_pressure[set] < 3 && pc_run[pc_idx] >= 2) set_pressure[set]++;
        } else if (cur != prev) {
            if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
        }
    }
    pc_last_line[pc_idx] = cur;

    // Decide insertion policy
    const bool is_pref = (type == PREFETCH);
    const bool stream_now = pending_stream_ins[set] != 0;
    const uint8_t sig8 = pc_sig8(PC);
    const bool pc_is_cold = (pc_cold[sig8] >= 2);

    // Install new metadata
    meta[set][way].pf = is_pref ? 1 : 0;
    meta[set][way].mhits = 0;
    meta[set][way].stream = stream_now ? 1 : 0;

    if (is_pref || stream_now || pc_is_cold) {
        meta[set][way].ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1); // tail
    } else {
        meta[set][way].ttl = 2;
        lru_insert_pos(set, way, LLC_WAYS - 4); // near tail, allow survival
    }

    // Record fill PC signature in sampled sets
    if (si >= 0) {
        fill_pc8[si][way] = sig8;
    }

    // Clear transient flag
    pending_stream_ins[set] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}