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
#define PC_IDX_SIZE 2048            // 2K-entry PC coldness table (2-bit)
#define FOOT_SIG_SIZE 2048          // 2K-entry PC⊕page footprint sentinel
#define LAST_LINE_BITS 12           // low bits of last line number (mod 4096)
#define DECAY_PERIOD 4096           // periodic decay cadence

// ---------------- Per-line metadata (packed conceptually to 10 bits) ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (recency bucket; 0=MRU)
    uint8_t ttl;      // 0..3 (survival priority)
    uint8_t mhits;    // 0..3 (demand-hit count; multi-hit gating)
    uint8_t pf;       // 0/1 (filled by prefetch)
    uint8_t stream;   // 0/1 (stream-clamped: never promote)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC coldness predictor (2-bit per PC) ----------------
static uint8_t pc_cold[PC_IDX_SIZE]; // 0=strong cold ... 3=hot

// ---------------- Footprint streaming sentinel (PC⊕page) ----------------
static uint16_t foot_last_line[FOOT_SIG_SIZE]; // LAST_LINE_BITS of line#
static uint8_t  foot_run[FOOT_SIG_SIZE];       // 0..3 forward +1 stride run

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint16_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }

static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t pc_index(uint64_t pc) {
    // simple mix; PC_IDX_SIZE is power of two
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}

static inline uint32_t foot_sig(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12);
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7) ^ (page << 13);
    return static_cast<uint32_t>(mix) & (FOOT_SIG_SIZE - 1);
}

// periodic light decay for pc_cold and footprint run
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_cold
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (decay_ptr + i) & (PC_IDX_SIZE - 1);
        if (pc_cold[idx] > 0) pc_cold[idx]--;
    }
    // Bleed footprint runs sparsely
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr + (i << 1)) & (FOOT_SIG_SIZE - 1);
        if (foot_run[idx] > 0) foot_run[idx]--;
    }
    decay_ptr = (decay_ptr + 1) & (PC_IDX_SIZE - 1);
}

// Update footprint sentinel; return streaming prediction
static inline bool update_stream_sentinel(uint32_t fsig, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

    uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
    uint16_t cur = static_cast<uint16_t>(line) & mask;
    uint16_t prev = foot_last_line[fsig];
    // compute delta modulo 2^LAST_LINE_BITS
    uint16_t delta = static_cast<uint16_t>((cur - prev) & mask);

    if (delta == 1) {
        if (foot_run[fsig] < 3) foot_run[fsig]++;
    } else if (cur != prev) {
        if (foot_run[fsig] > 0) foot_run[fsig]--;
    }
    foot_last_line[fsig] = cur;

    return (foot_run[fsig] >= 2);
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

// Victim scoring (higher is better victim)
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                         // old lines preferred
    s += (3 - m.ttl) * 6;               // low TTL preferred
    s += (m.mhits == 0) ? 6 : ((m.mhits == 1) ? 2 : 0); // single/no-hit preferred
    s += m.pf ? 5 : 0;                  // prefetch quarantined
    s += m.stream ? 7 : 0;              // stream-clamped preferred
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;     // stack position
            meta[s][w].ttl = 1;     // modest default
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) pc_cold[i] = 1; // start slightly cold
    for (uint32_t i = 0; i < FOOT_SIG_SIZE; i++) {
        foot_last_line[i] = 0;
        foot_run[i] = 0;
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
    event_ctr++;
    periodic_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);
    uint32_t fsig = foot_sig(pc_idx, paddr);
    bool stream_pred = update_stream_sentinel(fsig, line, type);

    LineMeta &m = meta[set][way];

    if (hit) {
        // On hit: multi-hit gating; never promote stream-tagged or prefetch lines
        if (is_demand(type)) {
            if (m.mhits == 0) {
                m.mhits = 1;
                // modest positive feedback for the PC
                if (pc_cold[pc_idx] < 3) pc_cold[pc_idx]++;
            } else {
                // second and later demand hits: unlock promotion unless quarantined
                if (!m.stream && !m.pf) {
                    if (m.mhits < 3) m.mhits++;
                    lru_make_mru(set, way);
                    if (m.ttl < 3) m.ttl++; // reward survival
                    if (pc_cold[pc_idx] <= 2) pc_cold[pc_idx] = 3; // strong hot feedback
                } else {
                    // quarantined: track reuse but no promotion
                    if (m.mhits < 3) m.mhits++;
                }
            }
        } else {
            // PREFETCH/WRITEBACK hits: no promotion
            // keep mhits unchanged to preserve demand-only gating
        }
        return;
    }

    // Miss fill / insertion
    // Default soft-bypass for streams or cold PCs by tail insertion, no promotion
    uint8_t pc_state = pc_cold[pc_idx]; // 0..3
    bool is_stream = (type != WRITEBACK) && stream_pred;

    // By default, assume cold: small negative feedback on fill for demandish refs
    if (is_demandish(type) && pc_cold[pc_idx] > 0) pc_cold[pc_idx]--;

    // Initialize line metadata
    m.mhits = 0;
    m.pf = (type == PREFETCH) ? 1 : 0;
    m.stream = is_stream ? 1 : 0;

    // TTL and insertion position
    if (type == WRITEBACK) {
        // Never bypass writebacks; insert near tail conservatively
        m.ttl = 1;
        lru_make_lru(set, way);
    } else if (m.pf) {
        // Prefetch quarantine: lowest priority
        m.ttl = 0;
        lru_make_lru(set, way);
    } else if (is_stream) {
        // Stream clamp: lowest priority, never promote
        m.ttl = 0;
        lru_make_lru(set, way);
    } else {
        // Demand insertion guided by PC coldness
        if (pc_state <= 1) {
            m.ttl = pc_state; // 0 or 1
            lru_make_lru(set, way); // tail insert to avoid pollution
        } else {
            m.ttl = 2; // moderate TTL
            // still insert near tail to encourage confirmation before staying
            lru_insert_pos(set, way, LLC_WAYS - 2);
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