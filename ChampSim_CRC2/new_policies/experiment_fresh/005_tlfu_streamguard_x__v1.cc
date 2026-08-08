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

// ---------------- Tunables (refined) ----------------
#define PC_IDX_SIZE 1024                 // PC hash entries (10-bit)
#define CMS_ROWS 3
#define CMS_COLS 1024                    // 10-bit index per row
#define CMS_MAX 15                       // 4-bit saturating counters
#define STREAM_RUN_THR 2                 // >=2 consecutive +1 strides => stream
#define SCAN_DEBT_MAX 3                  // 0..3 per-set scan debt
#define DECAY_PERIOD 4096                // TinyLFU decay cadence
#define TLFU_BASE_THR 2                  // frequency threshold
#define SAMPLE_EVERY 16                  // sample 1/16 sets for SHiP training
#define NSAMP (LLC_SETS / SAMPLE_EVERY)

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
static uint8_t  PC_LAST_LINE[PC_IDX_SIZE]; // low 8 bits of last line#
static uint8_t  PC_RUN[PC_IDX_SIZE];       // 0..3 (+1 stride run)

// ---------------- TinyLFU (PC admission) ----------------
static uint8_t cms[CMS_ROWS][CMS_COLS]; // 4-bit counters per bucket
static uint16_t decay_ptr = 0;

// ---------------- Sampled SHiP-like dead-block predictor ----------------
static uint8_t ship_ctr[PC_IDX_SIZE]; // 2-bit saturating (0..3)

struct SampleEntry {
    uint16_t sig; // 10-bit PC index stored
    uint8_t valid;
    uint8_t hit;
};
static SampleEntry sampler[NSAMP][LLC_WAYS];

// ---------------- Epoch ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline bool is_demandish(uint32_t type) { return (type == LOAD) || (type == RFO) || (type == PREFETCH); }
static inline bool is_sampled_set(uint32_t set) { return ((set & (SAMPLE_EVERY - 1)) == 0); }
static inline uint32_t samp_index(uint32_t set) { return set / SAMPLE_EVERY; }

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
// approximate periodic decay to bound counts and bleed scan debt
static inline void tlfu_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
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

    uint8_t cur = static_cast<uint8_t>(line & 0xFF);
    uint8_t prev = PC_LAST_LINE[pc_idx];
    uint8_t stride = static_cast<uint8_t>(cur - prev);

    if (stride == 1) {
        if (PC_RUN[pc_idx] < 3) PC_RUN[pc_idx]++;
    } else {
        if (PC_RUN[pc_idx] > 0) PC_RUN[pc_idx]--;
    }
    PC_LAST_LINE[pc_idx] = cur;

    if (PC_RUN[pc_idx] >= STREAM_RUN_THR) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    } else {
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
            meta[s][w] = {static_cast<uint8_t>(w), 0, 0, 0, 0};
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_LAST_LINE[i] = 0;
        PC_RUN[i] = 0;
        ship_ctr[i] = 1; // neutral
    }
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t c = 0; c < CMS_COLS; c++) {
            cms[r][c] = 0;
        }
    }
    for (uint32_t si = 0; si < NSAMP; si++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sampler[si][w].sig = 0;
            sampler[si][w].valid = 0;
            sampler[si][w].hit = 0;
        }
    }
    event_ctr = 0;
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
        if (!current_set[w].valid) return w;
    }
    // Otherwise choose the highest score
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (sc > best_score) {
            best_score = sc;
            best_way = w;
        } else if (sc == best_score) {
            // tie-breaker: older (higher LRU) first
            if (meta[set][w].lru > meta[set][best_way].lru)
                best_way = w;
        }
    }
    return best_way;
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
    event_ctr++;
    tlfu_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);

    // TinyLFU update on any demandish access
    if (is_demandish(type)) tlfu_inc(PC);

    // Update stream sentinel (used for both hits and fills)
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    // If hit path
    if (hit) {
        // SHiP online conservative reward: a hit from this PC suggests not-dead
        if (is_demand(type) && ship_ctr[pc_idx] > 0) ship_ctr[pc_idx]--;

        // sampled-set hit marking
        if (is_sampled_set(set)) {
            uint32_t si = samp_index(set);
            sampler[si][way].hit = 1;
        }

        // No promotion for stream-clamped or prefetch-only hits
        if (meta[set][way].stream) return;

        if (is_demand(type)) {
            if (meta[set][way].mhits == 0) {
                meta[set][way].mhits = 1; // arm on first demand hit
                // do not promote yet
            } else {
                // second demand hit: promote and boost TTL
                if (meta[set][way].ttl < 3) meta[set][way].ttl = 3;
                lru_make_mru(set, way);
                if (meta[set][way].mhits < 3) meta[set][way].mhits++;
            }
        } else {
            // prefetch hit: never promote; leave mhits unchanged here
        }
        return;
    }

    // Miss/Fill path: train sampler (old occupant) before overwriting entry
    if (is_sampled_set(set)) {
        uint32_t si = samp_index(set);
        SampleEntry &e = sampler[si][way];
        if (e.valid) {
            uint32_t sig = e.sig & (PC_IDX_SIZE - 1);
            if (e.hit == 0) {
                if (ship_ctr[sig] < 3) ship_ctr[sig]++; // dead -> penalize
            } else {
                if (ship_ctr[sig] > 0) ship_ctr[sig]--; // live -> reward
            }
        }
        // new entry for the line being filled
        e.sig = static_cast<uint16_t>(pc_idx);
        e.valid = 1;
        e.hit = 0;
    }

    // Decide insertion based on predictors
    uint8_t thr = TLFU_BASE_THR + ((set_scan_debt[set] >= 2) ? 1 : 0);
    uint8_t freq = tlfu_query(PC);
    bool ship_cold = (ship_ctr[pc_idx] >= 2);
    bool pf = (type == PREFETCH);

    uint8_t ins_ttl = 0;
    uint8_t ins_pos = LLC_WAYS - 1; // default tail insert
    uint8_t ins_stream = 0;
    uint8_t ins_mhits = 0;

    if (type == WRITEBACK) {
        // never bypass on WB; treat as useful
        ins_ttl = 2;
        ins_pos = 4;
        ins_stream = 0;
        ins_mhits = 1;
        pf = 0;
    } else if (stream_pred) {
        // confident stream -> clamp like bypass: TTL=0, tail insert, no promotion
        ins_ttl = 0;
        ins_pos = LLC_WAYS - 1;
        ins_stream = 1;
        ins_mhits = 0;
    } else if (pf) {
        // quarantine prefetches at tail
        ins_ttl = 0;
        ins_pos = LLC_WAYS - 1;
        ins_stream = 0;
        ins_mhits = 0;
    } else if (ship_cold && freq < thr) {
        // hybrid admission says cold -> tail insert TTL=0
        ins_ttl = 0;
        ins_pos = LLC_WAYS - 1;
        ins_stream = 0;
        ins_mhits = 0;
    } else {
        // admitted demand line
        if (freq >= (uint8_t)(thr + 2) && set_scan_debt[set] == 0) {
            ins_ttl = 2;
            ins_pos = 6; // slightly closer to MRU
        } else {
            ins_ttl = 1;
            ins_pos = 10; // mid-low insertion
        }
        ins_stream = 0;
        ins_mhits = 0;
    }

    // Materialize insertion
    meta[set][way].ttl = ins_ttl;
    meta[set][way].mhits = ins_mhits;
    meta[set][way].pf = pf ? 1 : 0;
    meta[set][way].stream = ins_stream;
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