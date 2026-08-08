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
#define PC_IDX_SIZE 2048               // PC hash entries
#define CMS_ROWS 3
#define CMS_COLS 1024                  // 10-bit index per row
#define CMS_MAX 63                     // 6-bit saturating counters
#define STREAM_RUN_THR 2               // >=2 consecutive +1 strides => stream
#define SCAN_DEBT_MAX 3                // 0..3 per-set scan debt
#define DECAY_PERIOD 4096              // periodic decay cadence

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (0=MRU)
    uint8_t ttl;      // 0..3 survival priority
    uint8_t mhits;    // 0..3 demand hit count (multi-hit gating)
    uint8_t pf;       // 0/1 filled by prefetch
    uint8_t stream;   // 0/1 stream-clamped (no promotion)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set stream pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed stream sentinel ----------------
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low bits of last line#
static uint8_t  PC_RUN[PC_IDX_SIZE];       // 0..3 (+1 stride run)

// ---------------- TinyLFU (PC admission) ----------------
static uint8_t cms[CMS_ROWS][CMS_COLS]; // 6-bit counters per bucket
static uint16_t decay_ptr = 0;

// ---------------- Epoch ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline bool is_demandish(uint32_t type) { return (type == LOAD) || (type == RFO) || (type == PREFETCH); }

static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint64_t line_number(uint64_t paddr) {
    return (paddr >> 6); // 64B line
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
static inline void lru_make_lru(uint32_t set, uint32_t way) {
    lru_insert_pos(set, way, LLC_WAYS - 1);
}

// TinyLFU hashing
static inline uint32_t cms_h(uint64_t pc, uint32_t r) {
    uint64_t x = pc + 0x9e3779b97f4a7c15ULL + (static_cast<uint64_t>(r) << 32);
    x ^= (x >> 33); x *= 0xff51afd7ed558ccdULL;
    x ^= (x >> 33); x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    return static_cast<uint32_t>(x) & (CMS_COLS - 1);
}
static inline uint8_t tlfu_query(uint64_t pc) {
    uint8_t m = CMS_MAX;
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_h(pc, r);
        uint8_t v = cms[r][idx];
        if (v < m) m = v;
    }
    return m;
}
static inline void tlfu_inc(uint64_t pc) {
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_h(pc, r);
        if (cms[r][idx] < CMS_MAX) cms[r][idx]++;
    }
}
// approximate periodic decay to bound counts
static inline void tlfu_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    // decay a sparse stripe to keep overhead low
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_COLS; i += 8) {
            uint32_t idx = (decay_ptr + i) & (CMS_COLS - 1);
            if (cms[r][idx] > 0) cms[r][idx]--;
        }
    }
    decay_ptr = (decay_ptr + 1) & (CMS_COLS - 1);

    // bleed scan debt sparsely to recover after scans
    for (uint32_t s = 0; s < LLC_SETS; s += 64) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

    uint16_t cur = static_cast<uint16_t>(line & 0xFFFF);
    uint16_t prev = PC_LAST_LINE[pc_idx];
    int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(prev);

    if (stride == 1) {
        if (PC_RUN[pc_idx] < 3) PC_RUN[pc_idx]++;
    } else {
        if (PC_RUN[pc_idx] > 0) PC_RUN[pc_idx]--;
    }
    PC_LAST_LINE[pc_idx] = cur;

    if (PC_RUN[pc_idx] >= STREAM_RUN_THR) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    } else {
        // occasional bleed
        if ((event_ctr & 0x1F) == 0 && set_scan_debt[set] > 0) set_scan_debt[set]--;
    }

    return (PC_RUN[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                         // older is better victim
    s += (3 - m.ttl) * 8;               // low TTL preferred
    s += (m.mhits == 0) ? 8 : 0;        // single/no-hit preferred
    s += m.pf ? 4 : 0;                  // prefetch quarantined
    s += m.stream ? 6 : 0;              // stream-clamped preferred
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;   // diverse starting stack
            meta[s][w].ttl = 0;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_LAST_LINE[i] = 0;
        PC_RUN[i] = 0;
    }
    for (uint32_t r = 0; r < CMS_ROWS; r++)
        for (uint32_t c = 0; c < CMS_COLS; c++)
            cms[r][c] = 0;
    decay_ptr = 0;
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
    // Choose highest-score eviction candidate
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if ((w == 0) || (sc > best_score)) {
            best_score = sc;
            best_way = w;
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
    (void)cpu; (void)victim_addr; // unused
    event_ctr++;
    tlfu_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);

    // Train TinyLFU on demand references (both hits and misses)
    if (is_demand(type)) {
        tlfu_inc(PC);
    }

    // Update streaming sentinel for this access
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    if (hit) {
        LineMeta &m = meta[set][way];

        // streaming lines are clamped: no promotion
        if (m.stream) {
            // keep TTL low and avoid stack promotion to resist scans
            if (m.ttl > 0) m.ttl--; // gentle decay on hit to streaming
            return;
        }

        // handle prefetch-to-demand transitions
        if (is_demand(type) && m.pf) {
            m.pf = 0; // drop prefetch quarantine tag on demand use
        }

        // multi-hit gating: never promote TTL on first demand hit
        if (is_demand(type)) {
            if (m.mhits == 0) {
                m.mhits = 1;
                // recency benefit for the first hit, but TTL unchanged
                lru_make_mru(set, way);
            } else {
                if (m.mhits < 3) m.mhits++;
                if (m.ttl < 3) m.ttl++;
                lru_make_mru(set, way);
            }
        } else {
            // prefetch hit: do not promote
        }
        return;
    }

    // Miss path: initialize inserted line metadata
    LineMeta &m = meta[set][way];
    m.mhits = 0;
    m.stream = 0;
    m.pf = (type == PREFETCH) ? 1 : 0;

    if (type == WRITEBACK) {
        // Never bypass writebacks; moderate protection and middle insertion
        m.ttl = 2;
        m.stream = 0;
        lru_insert_pos(set, way, 8);
        return;
    }

    // Demand or Prefetch fill decisions
    uint8_t q = tlfu_query(PC); // TinyLFU frequency estimate

    if (stream_pred) {
        // Aggressive stream clamp: tail insert, TTL=0, never promote
        m.stream = 1;
        m.ttl = 0;
        lru_make_lru(set, way);
        return;
    }

    if (type == PREFETCH) {
        // Quarantine prefetches at tail with low TTL
        m.ttl = 0;
        lru_make_lru(set, way);
        return;
    }

    // Demand fill with TinyLFU admission
    if (q <= 1) {
        // Soft-bypass: tail insert, lowest TTL (resists pollution)
        m.ttl = 0;
        lru_make_lru(set, way);
    } else {
        // Admitted: modest TTL and mid-stack position
        m.ttl = 1;
        lru_insert_pos(set, way, 8);
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