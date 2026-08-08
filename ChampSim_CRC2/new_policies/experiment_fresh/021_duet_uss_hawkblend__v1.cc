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
#define PC_TBL_SIZE 2048          // PC-indexed tables (power of two)
#define SAMPLE_RATE 32            // 1 out of 32 sets sampled (64 sets for 2048 total)
#define DECAY_PERIOD 4096         // periodic decay cadence (events)
#define STREAM_BASE_THR 2         // base run threshold for streaming
#define STREAM_HARD_THR 3         // hardened threshold when reuse observed

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ttl;     // 0..3 survival priority
    uint8_t mhits;   // 0..3 demand-hit count (multi-hit gating)
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped (no promotion)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Sampler shadow tags (only used in sampled sets) ----------------
static uint16_t sample_sig[LLC_SETS][LLC_WAYS]; // PC index
static uint8_t  sample_seen[LLC_SETS][LLC_WAYS]; // saw at least one demand hit

// ---------------- PC utility predictor (3-bit per PC, 0..7) ----------------
static uint8_t pc_util[PC_TBL_SIZE];

// ---------------- Streaming sentinel (PC-indexed stride/run + hardening) ----------------
static uint16_t pc_last_line[PC_TBL_SIZE]; // low bits of last line ID
static uint8_t  pc_run[PC_TBL_SIZE];       // 0..3 consecutive +1 run
static uint8_t  pc_harden[PC_TBL_SIZE];    // 0..3 reuse-driven hardening for stream threshold
static uint8_t  set_scan_debt[LLC_SETS];   // 0..3 per-set scan pressure

// ---------------- Time/decay ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline bool is_sampled(uint32_t set) { return ((set & (SAMPLE_RATE - 1)) == 0); }

static inline uint32_t pc_index(uint64_t pc) {
    // simple mix; PC_TBL_SIZE is power of two
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_TBL_SIZE - 1);
}

// saturating helpers
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }
static inline void sat_add_down(uint8_t &x, uint8_t dec) { x = (x > dec) ? uint8_t(x - dec) : uint8_t(0); }

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
    s += m.lru;                               // older is better victim
    s += (3 - m.ttl) * 8;                     // low TTL preferred
    s += (m.mhits == 0) ? 8 : ((m.mhits == 1) ? 4 : 0); // zero/single-hit preferred
    s += m.pf ? 6 : 0;                        // prefetch quarantined
    s += m.stream ? 10 : 0;                   // stream-clamped preferred
    return s;
}

// periodic light decay for pc_util, pc_run, hardening, and scan debt
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_util, pc_run, and harden
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (decay_ptr + i) & (PC_TBL_SIZE - 1);
        if (pc_util[idx] > 0) pc_util[idx]--;      // soften old hot PCs
        if (pc_run[idx] > 0) pc_run[idx]--;        // relax run confidence
        if (pc_harden[idx] > 0) pc_harden[idx]--;  // relax hardened threshold as phases change
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

    uint8_t thr = (pc_harden[pc_idx] > 0) ? STREAM_HARD_THR : STREAM_BASE_THR;

    if (pc_run[pc_idx] >= thr) {
        if (set_scan_debt[set] < 3) set_scan_debt[set]++;
    } else {
        // light bleed of debt
        if ((event_ctr & 0x1F) == 0 && set_scan_debt[set] > 0) set_scan_debt[set]--;
    }

    return (pc_run[pc_idx] >= thr) || (set_scan_debt[set] >= 2);
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
            sample_sig[s][w] = 0;
            sample_seen[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_util[i] = 3;       // slightly cold by default in [0..7]
        pc_last_line[i] = 0;
        pc_run[i] = 0;
        pc_harden[i] = 0;
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

    // Choose the highest scoring victim
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
    (void)cpu; (void)victim_addr;
    event_ctr++;
    periodic_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);

    // Update streaming sentinel for this access
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    if (hit) {
        // Demand hit handling
        if (is_demand(type)) {
            // SHiP-like training at first demand re-reference (only for sampled sets)
            if (is_sampled(set) && sample_seen[set][way] == 0) {
                sample_seen[set][way] = 1;
                if (pc_util[pc_idx] < 7) pc_util[pc_idx]++; // reward reuse
                sat_inc(pc_harden[pc_idx], 3);              // harden stream threshold for this PC
            }

            // Multi-hit gating: only promote after second demand hit; never promote if stream or prefetch
            if (!meta[set][way].pf && !meta[set][way].stream) {
                if (meta[set][way].mhits < 3) meta[set][way].mhits++;
                if (meta[set][way].mhits >= 2) {
                    if (meta[set][way].ttl < 3) meta[set][way].ttl++;
                    lru_make_mru(set, way);
                }
            } else {
                // keep quarantined: minor age relief but no promotion
                if (meta[set][way].ttl > 0) meta[set][way].ttl--;
            }
        } else {
            // Non-demand hits (e.g., prefetch hit, writeback acknowledgment): keep at place
            // Intentionally avoid promotion to preserve quarantine behavior
        }
        return;
    }

    // Miss path: train on eviction (for sampled sets), then insert new line
    if (is_sampled(set)) {
        // Eviction feedback: DOA (no demand hit) => strong negative; else neutral
        uint16_t old_sig = sample_sig[set][way];
        uint8_t  seen = sample_seen[set][way];
        if (seen == 0) {
            // strong negative feedback
            if (pc_util[old_sig] >= 2) pc_util[old_sig] -= 2;
            else pc_util[old_sig] = 0;
        }
        // Initialize sampler entry for the incoming fill
        sample_sig[set][way] = static_cast<uint16_t>(pc_idx);
        sample_seen[set][way] = 0;
    }

    // Classify PC utility
    uint8_t util = pc_util[pc_idx]; // 0..7
    bool cold_pc  = (util <= 1);
    bool warm_pc  = (util >= 2 && util <= 3);
    bool hot_pc   = (util >= 4);

    // Decide insertion parameters
    meta[set][way].pf = (type == PREFETCH) ? 1 : 0;

    // WRITEBACKs must not bypass or be stream-clamped
    if (type == WRITEBACK) {
        meta[set][way].stream = 0;
        meta[set][way].mhits = 0;
        meta[set][way].ttl = 2;
        lru_insert_pos(set, way, 4); // reasonable insertion
        return;
    }

    // Streaming or cold PCs or prefetch => tail insert, no promotion path
    if (stream_pred || cold_pc || (type == PREFETCH)) {
        meta[set][way].stream = stream_pred ? 1 : 0;
        meta[set][way].mhits = 0;
        meta[set][way].ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
        return;
    }

    // Warm PCs: cautious insert near tail
    if (warm_pc) {
        meta[set][way].stream = 0;
        meta[set][way].mhits = 0;
        meta[set][way].ttl = 1;
        lru_insert_pos(set, way, LLC_WAYS - 2);
        return;
    }

    // Hot PCs: admit mid-MRU with modest TTL, but still require multi-hit for future promotion
    if (hot_pc) {
        meta[set][way].stream = 0;
        meta[set][way].mhits = 0;
        meta[set][way].ttl = 2;
        lru_insert_pos(set, way, 4);
        return;
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