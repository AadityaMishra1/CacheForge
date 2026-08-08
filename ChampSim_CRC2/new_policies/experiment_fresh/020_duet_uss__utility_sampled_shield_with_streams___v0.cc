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
#define PC_TBL_SIZE 2048          // PC-indexed tables (power of two)
#define SAMPLE_RATE 32            // 1 out of 32 sets are sampled (64 sets)
#define STREAM_RUN_THR 2          // >=2 consecutive +1 strides => streaming
#define DECAY_PERIOD 4096         // periodic decay cadence

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ttl;     // 0..3 survival priority
    uint8_t mhits;   // 0..3 demand-hit count (multi-hit gating)
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped (no promotion)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// Internal allocation tracker (maps to hardware valid; not counted as extra storage)
static uint8_t allocated[LLC_SETS][LLC_WAYS];

// ---------------- Sampler shadow tags (only used in sampled sets) ----------------
static uint16_t sample_sig[LLC_SETS][LLC_WAYS]; // PC index stored on fill
static uint8_t  sample_seen[LLC_SETS][LLC_WAYS]; // saw at least one demand hit

// ---------------- PC utility predictor (2-bit per PC) ----------------
static uint8_t pc_util[PC_TBL_SIZE]; // 0=strong cold .. 3=strong hot

// ---------------- Streaming sentinel (PC-indexed stride/run) ----------------
static uint16_t pc_last_line[PC_TBL_SIZE]; // low bits of last line ID
static uint8_t  pc_run[PC_TBL_SIZE];       // 0..3 consecutive +1 run
static uint8_t  set_scan_debt[LLC_SETS];   // 0..3 per-set scan pressure

// ---------------- Time/decay ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline bool is_demandish(uint32_t type) { return (type == LOAD) || (type == RFO) || (type == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline bool is_sampled(uint32_t set) { return ((set & (SAMPLE_RATE - 1)) == 0); }

static inline uint32_t pc_index(uint64_t pc) {
    // simple mix; PC_TBL_SIZE is power of two
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_TBL_SIZE - 1);
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
    s += m.lru;                         // older is better victim
    s += (3 - m.ttl) * 6;               // low TTL preferred
    s += (m.mhits == 0) ? 6 : ((m.mhits == 1) ? 2 : 0); // zero/single-hit preferred
    s += m.pf ? 4 : 0;                  // prefetch quarantined
    s += m.stream ? 8 : 0;              // stream-clamped preferred
    return s;
}

// periodic light decay for pc_util, pc_run, and scan debt
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_util and pc_run
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (decay_ptr + i) & (PC_TBL_SIZE - 1);
        if (pc_util[idx] > 0) pc_util[idx]--;
        if (pc_run[idx] > 0) pc_run[idx]--;
    }
    // Bleed scan debt sparsely across sets
    for (uint32_t s = decay_ptr; s < LLC_SETS; s += 64) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    decay_ptr = (decay_ptr + 1) & (PC_TBL_SIZE - 1);
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

    uint16_t cur = static_cast<uint16_t>(line & 0xFFFFu);
    uint16_t prev = pc_last_line[pc_idx];
    uint16_t delta = static_cast<uint16_t>(cur - prev); // modulo 2^16
    if (delta == 1) {
        if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
    } else {
        if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
    }
    pc_last_line[pc_idx] = cur;

    if (pc_run[pc_idx] >= STREAM_RUN_THR) {
        if (set_scan_debt[set] < 3) set_scan_debt[set]++;
    } else {
        if ((event_ctr & 0x1F) == 0 && set_scan_debt[set] > 0) set_scan_debt[set]--;
    }

    return (pc_run[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;
            meta[s][w].ttl = 1;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            allocated[s][w] = 0;
            sample_sig[s][w] = 0;
            sample_seen[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_util[i] = 1;       // slightly cold by default
        pc_last_line[i] = 0;
        pc_run[i] = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Score-based victim among valid ways
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t s = score_line(meta[set][w]);
        if (w == 0 || s > best_score) {
            best_score = s;
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
    (void)cpu; (void)victim_addr;
    event_ctr++;
    periodic_decay();

    const bool sampled = is_sampled(set);
    const uint32_t pc_idx = pc_index(PC);
    const uint64_t line = line_number(paddr);

    // Update streaming sentinel for this access (writebacks excluded)
    const bool stream_now = update_stream_sentinel(set, pc_idx, line, type);

    if (hit) {
        // Record sampler observation
        if (sampled && is_demand(type)) {
            sample_seen[set][way] = 1;
        }

        // No promotion on prefetch hits or stream-clamped lines
        if (!is_demand(type)) return;

        if (meta[set][way].stream) {
            // Keep clamped: do not promote
            // Optionally refresh tiny TTL to avoid instant eviction under short bursts
            if (meta[set][way].ttl < 1) meta[set][way].ttl = 1;
            return;
        }

        // Multi-hit gating: first demand hit sets mhits=1 without promotion
        if (meta[set][way].mhits == 0) {
            meta[set][way].mhits = 1;
            if (meta[set][way].ttl < 1) meta[set][way].ttl = 1;
            return;
        }

        // Second and later demand hits: promote and strengthen TTL
        lru_make_mru(set, way);
        if (meta[set][way].mhits < 3) meta[set][way].mhits++;
        if (meta[set][way].ttl < 3) meta[set][way].ttl++;
        return;
    }

    // Miss path: before installing, if we are evicting a valid line in a sampled set, train utility
    if (sampled && allocated[set][way]) {
        uint16_t esig = sample_sig[set][way];
        uint8_t saw_hit = sample_seen[set][way];
        if (saw_hit) {
            if (pc_util[esig] < 3) pc_util[esig]++;
        } else {
            if (pc_util[esig] > 0) pc_util[esig]--;
        }
    }

    // Decide admission/insertion policy
    bool predicted_hot = (pc_util[pc_idx] >= 2);
    bool is_pf = (type == PREFETCH);

    // Default fresh metadata
    meta[set][way].pf = is_pf ? 1 : 0;
    meta[set][way].stream = 0;
    meta[set][way].mhits = 0;

    if (type == WRITEBACK) {
        // Never bypass writebacks: insert at tail, neutral TTL
        lru_insert_pos(set, way, LLC_WAYS - 1);
        meta[set][way].ttl = 0;
        meta[set][way].stream = 0;
    } else if (stream_now) {
        // Aggressive stream clamp: tail insert, TTL=0, never promote
        lru_insert_pos(set, way, LLC_WAYS - 1);
        meta[set][way].ttl = 0;
        meta[set][way].stream = 1;
    } else if (is_pf) {
        // Prefetch quarantine: tail insert, low TTL
        lru_insert_pos(set, way, LLC_WAYS - 1);
        meta[set][way].ttl = 0;
    } else if (!predicted_hot) {
        // Cold PC: tail insert, low TTL, wait for 2nd hit to promote
        lru_insert_pos(set, way, LLC_WAYS - 1);
        meta[set][way].ttl = 0;
    } else {
        // Hot PC: admit near-MRU with modest TTL, but still multi-hit gated
        uint8_t near_mru_pos = (LLC_WAYS >= 4) ? 2 : 0;
        lru_insert_pos(set, way, near_mru_pos);
        meta[set][way].ttl = 1;
    }

    // Update sampler bookkeeping for the installed line
    if (sampled) {
        sample_sig[set][way] = pc_idx;
        sample_seen[set][way] = 0;
    }
    allocated[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}