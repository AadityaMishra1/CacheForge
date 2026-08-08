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

// ---------------- Tunables (kept stable, tightened thresholds) ----------------
#define PC_IDX_SIZE 2048          // PC-indexed tables (power of two)
#define STREAM_RUN_THR 2          // >=2 consecutive +1 strides => streamy
#define SCAN_DEBT_MAX 3           // per-set stream pressure cap (0..3)
#define DECAY_PERIOD 2048         // periodic decay cadence (power of two)
#define PIN_STRENGTH 2            // pin for N=2 touches after 2nd demand hit

// ---------------- Per-line metadata (13 bits conceptual) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ehc;     // 0..7 expected hits remaining
    uint8_t mhits;   // 0..3 demand hit count
    uint8_t pf;      // 0/1 prefetched
    uint8_t stream;  // 0/1 stream-clamped (never promote)
    uint8_t pin;     // 0..3 short-term protection after 2nd hit
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC-based predictors ----------------
static uint8_t  pc_ehc[PC_IDX_SIZE];       // 3-bit expected-hit counter per PC
static uint16_t pc_last_line[PC_IDX_SIZE]; // low 16 bits of last line number
static uint8_t  pc_run[PC_IDX_SIZE];       // +1 stride run length (0..3)
static uint8_t  pc_live[PC_IDX_SIZE];      // 2-bit TTL: alloc without reuse -> penalize

// ---------------- Per-set stream pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- Time ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t pc_index(uint64_t pc) {
    // simple entropy mix; PC_IDX_SIZE is power of two
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
    // Shift positions to make space at 'pos'
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru <= pos) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = pos;
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return (set_scan_debt[set] >= 2);

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
        // light bleed on non-stream activity
        if ((event_ctr & 0x3F) == 0 && set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
    return pc_stream || (set_scan_debt[set] >= 2);
}

// Victim scoring (higher is better victim)
static inline uint32_t score_line(const LineMeta &m) {
    if (m.pin > 0) return 0; // pinned lines strongly protected
    uint32_t s = 0;
    s += m.lru;                              // prefer older
    s += (m.ehc == 0) ? 16u : 0u;            // predicted dead
    s += (m.mhits == 0) ? 6u : ((m.mhits == 1) ? 2u : 0u);
    s += m.pf ? 10u : 0u;                    // prefetch quarantine
    s += m.stream ? 12u : 0u;                // stream-clamped
    return s;
}

// periodic decay for stream pressure and pc_live TTL (global cadence)
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    for (uint32_t s = 0; s < LLC_SETS; s += 64) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i += 64) {
        if (pc_live[i] > 0) pc_live[i]--;
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
            meta[s][w].pin = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_ehc[i] = 1;           // start slightly cold
        pc_last_line[i] = 0;
        pc_run[i] = 0;
        pc_live[i] = 0;
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

    uint32_t idx = pc_index(PC);
    uint64_t ln = line_number(paddr);

    // Maintain stream sentinel every access
    bool stream_now = update_stream_sentinel(set, idx, ln, type);

    if (hit) {
        // On hit: gated promotion and pinning
        if (is_demand(type)) {
            // update PC-live: reuse observed
            pc_live[idx] = 0;

            // decrement expected-hits remaining on demand consumption
            if (meta[set][way].ehc > 0) meta[set][way].ehc--;

            // track demand hits
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;

            bool promotable = (meta[set][way].pf == 0) && (meta[set][way].stream == 0);
            if (promotable) {
                if (meta[set][way].mhits == 2) {
                    // second demand hit: promote and pin, train PC as warm
                    if (pc_ehc[idx] < 7) pc_ehc[idx]++;
                    meta[set][way].pin = PIN_STRENGTH;
                    lru_make_mru(set, way);
                } else if (meta[set][way].pin > 0) {
                    // keep MRU while pin lasts
                    meta[set][way].pin--;
                    lru_make_mru(set, way);
                } else {
                    // first demand hit: no promotion
                    // leave LRU position unchanged
                }
            } else {
                // prefetch/stream lines never promote; allow pin decay if any
                if (meta[set][way].pin > 0) meta[set][way].pin--;
            }
        } else {
            // Non-demand hits (e.g., prefetch hit path): never promote
            if (meta[set][way].pin > 0) meta[set][way].pin--;
        }
        return;
    }

    // Miss path: allocate with soft-bypass (tail insert + clamp) for cold/stream
    bool cold_pc = (pc_ehc[idx] <= 1);
    bool clamp_stream = stream_now;

    // PC-live dead-block demotion: repeated allocs without reuse penalize PC
    if (is_demandish(type)) {
        if (pc_live[idx] > 0 && pc_ehc[idx] > 0) pc_ehc[idx]--;
        pc_live[idx] = 2; // arm TTL; cleared on future demand reuse
    }

    // Never bypass on WRITEBACK; otherwise soft-bypass via tail-insert + clamp tag
    bool soft_bypass = (type != WRITEBACK) && (cold_pc || clamp_stream);

    // Initialize metadata for the new line
    meta[set][way].pf     = (type == PREFETCH) ? 1 : 0;
    meta[set][way].stream = soft_bypass ? 1 : 0;
    meta[set][way].mhits  = 0;
    meta[set][way].pin    = 0;
    meta[set][way].ehc    = soft_bypass ? 0 : (pc_ehc[idx] & 0x7);

    // Insertion position: PF/Stream/Writeback => LRU; otherwise near-LRU
    uint8_t ins_pos = (meta[set][way].pf || meta[set][way].stream || (type == WRITEBACK))
                      ? (LLC_WAYS - 1)
                      : (LLC_WAYS - 2);

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