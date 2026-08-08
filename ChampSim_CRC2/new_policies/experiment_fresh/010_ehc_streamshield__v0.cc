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
#define PC_IDX_SIZE 2048          // power-of-two PC-indexed tables
#define STREAM_RUN_THR 2          // >=2 consecutive +1 strides => stream
#define SCAN_DEBT_MAX 3           // per-set stream pressure 0..3
#define DECAY_PERIOD 2048         // periodic decay cadence for stream pressure

// ---------------- Per-line metadata (packed conceptually to 11 bits) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (recency bucket; 0=MRU)
    uint8_t ehc;     // 0..7 expected hits remaining
    uint8_t mhits;   // 0..3 demand hit count (multi-hit gating)
    uint8_t pf;      // 0/1 (filled by prefetch)
    uint8_t stream;  // 0/1 (stream-clamped: never promote)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC-based predictors ----------------
static uint8_t  pc_ehc[PC_IDX_SIZE];      // 3-bit expected-hit counter per PC (0..7)
static uint16_t pc_last_line[PC_IDX_SIZE]; // low 16 bits of last line number
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3 (+1 stride run)

// ---------------- Per-set stream pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- Time ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t pc_index(uint64_t pc) {
    // simple mix; PC_IDX_SIZE is power of two
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
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
    meta[set][way] = {pos, meta[set][way].ehc, meta[set][way].mhits, meta[set][way].pf, meta[set][way].stream};
    meta[set][way].lru = pos;
}
static inline void lru_make_lru(uint32_t set, uint32_t way) {
    lru_insert_pos(set, way, LLC_WAYS - 1);
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

    uint16_t cur = static_cast<uint16_t>(line & 0xFFFF);
    uint16_t prev = pc_last_line[pc_idx];
    uint16_t delta = static_cast<uint16_t>(cur - prev);

    if (delta == 1) {
        if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
    } else if (cur != prev) {
        if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
    }
    pc_last_line[pc_idx] = cur;

    bool pc_stream = (pc_run[pc_idx] >= STREAM_RUN_THR);
    if (pc_stream) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    } else {
        // light bleed of per-set pressure
        if ((event_ctr & 0x3F) == 0 && set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
    return pc_stream || (set_scan_debt[set] >= 2);
}

// Victim scoring (higher is better victim)
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                              // prefer older
    s += (m.ehc == 0) ? 16 : 0;              // predicted no more hits => evict
    s += (m.mhits == 0) ? 8 : ((m.mhits == 1) ? 3 : 0); // single/no-hit preferred
    s += m.pf ? 6 : 0;                       // prefetch quarantined
    s += m.stream ? 8 : 0;                   // stream-clamped preferred
    return s;
}

// periodic decay for set stream pressure (global cadence)
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    for (uint32_t s = 0; s < LLC_SETS; s += 64) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;  // stack position
            meta[s][w].ehc = 0;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_ehc[i] = 1;           // start slightly cold: expect few hits
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    event_ctr = 0;
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
    (void)cpu; (void)victim_addr; // unused
    event_ctr++;
    periodic_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);

    // Update stream sentinel and get prediction for this access
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    LineMeta &m = meta[set][way];

    if (hit) {
        // Demand hits train PC EHC as "useful"
        if (is_demand(type)) {
            if (pc_ehc[pc_idx] < 7) pc_ehc[pc_idx]++;

            // Decrement expected-hits remaining for this line
            if (m.ehc > 0) m.ehc--;

            // Multi-hit gating
            if (m.mhits < 3) m.mhits++;

            // Promotion: allowed only after second demand hit and if not stream/prefetch
            if (!m.pf && !m.stream && m.mhits >= 2) {
                lru_make_mru(set, way);
            }
        } else {
            // prefetch hit: do not promote
        }
    } else {
        // Miss fill: initialize per-line metadata from PC predictors
        m.pf = (type == PREFETCH) ? 1 : 0;
        m.stream = stream_pred ? 1 : 0;
        m.mhits = 0;

        uint8_t pe = pc_ehc[pc_idx]; // expected hits for this PC (0..7)
        m.ehc = pe;

        // Admission and insertion position
        if (m.pf || m.stream || pe == 0) {
            // aggressive tail insertion for scans/cold PCs/prefetches
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else if (pe == 1) {
            lru_insert_pos(set, way, LLC_WAYS - 2);
        } else {
            // middle insertion to avoid polluting MRU while allowing reuse to surface
            uint8_t mid = (LLC_WAYS / 2) + 2;
            if (mid >= LLC_WAYS) mid = LLC_WAYS - 2;
            lru_insert_pos(set, way, mid);
        }

        // Modest penalty to consistently cold demand PCs
        if (is_demand(type) && pe == 0 && pc_ehc[pc_idx] > 0) {
            pc_ehc[pc_idx]--; // keep cold PCs cold
        }

        // Writebacks: treat as tail inserts with no tags
        if (type == WRITEBACK) {
            m.pf = 0;
            m.stream = 0;
            m.mhits = 0;
            m.ehc = 0;
            lru_make_lru(set, way);
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