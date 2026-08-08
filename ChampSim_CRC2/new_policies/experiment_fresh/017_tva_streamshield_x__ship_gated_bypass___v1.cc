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
#define PC_IDX_SIZE 512           // PC-indexed stream sentinel size (power of two)
#define REG_HOT_SIZE 512          // Region hotness (page-grain) table size
#define LAST_LINE_BITS 10         // Bits of line number to compare for +1 stride
#define DECAY_PERIOD 4096         // Periodic decay cadence (power of two)
#define STREAM_THR 2              // >=2 consecutive +1 strides => stream
#define SCAN_DEBT_MAX 3           // Per-set scan pressure cap
#define HOT_THR 2                 // Region hotness threshold (0..3) to relax clamping

// SHiP predictor (2-bit) with compact per-line signature
#define SHIP_SIG_BITS 6
#define SHIP_ENTRIES (1u << SHIP_SIG_BITS)

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); }   // 64B lines
static inline uint64_t page_number(uint64_t paddr) { return (paddr >> 12); }  // 4KB pages

static inline uint32_t pc_index(uint64_t pc) {
    // simple mix; PC_IDX_SIZE must be power-of-two
    uint64_t x = pc ^ (pc >> 9) ^ (pc << 7);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t reg_index(uint64_t paddr) {
    uint64_t pg = page_number(paddr);
    uint64_t mix = pg ^ (pg >> 5) ^ (pg << 13);
    return static_cast<uint32_t>(mix) & (REG_HOT_SIZE - 1);
}
static inline uint8_t ship_sig(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 15) ^ (pc << 3);
    return static_cast<uint8_t>(x) & (SHIP_ENTRIES - 1);
}

// ---------------- Per-line metadata (bit-packed conceptually to 13 bits) ----------------
struct LineMeta {
    uint8_t rrpv;   // 2 bits: 0..3 (SRRIP)
    uint8_t mhits;  // 2 bits: demand hit count (multi-hit gating), saturates at 3
    uint8_t pf;     // 1 bit : filled by prefetch
    uint8_t stream; // 1 bit : stream-clamped (no promotion)
    uint8_t sig;    // 6 bits: SHiP signature
    uint8_t valid;  // 1 bit : line validity for eviction update
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set scan pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed stream sentinel ----------------
static uint16_t pc_last_line[PC_IDX_SIZE]; // store LAST_LINE_BITS of line number
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3 (+1 stride run-length)

// ---------------- Region hotness (2-bit) ----------------
static uint8_t region_hot[REG_HOT_SIZE];   // 0..3 saturation

// ---------------- SHiP-style dead-block predictor (2-bit per entry) ----------------
static uint8_t ship_ctr[SHIP_ENTRIES];     // 0..3; higher => more likely live

// ---------------- Time/decay ----------------
static uint64_t event_ctr = 0;
static uint16_t decay_ptr_pc = 0;
static uint16_t decay_ptr_reg = 0;
static uint16_t decay_ptr_set = 0;

// ---------------- SRRIP victim selection helpers ----------------
static inline int pick_rrip_victim(uint32_t set, const BLOCK* current_set) {
    // Prefer: RRPV==3 with mhits==0 and (stream || pf); then any mhits==0; then any
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (!current_set[w].valid) return static_cast<int>(w);

    for (int iter = 0; iter < 8; iter++) {
        // Pass 1: stream/pf + cold
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3 && meta[set][w].mhits == 0 && (meta[set][w].stream || meta[set][w].pf))
                return static_cast<int>(w);
        }
        // Pass 2: cold
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3 && meta[set][w].mhits == 0)
                return static_cast<int>(w);
        }
        // Pass 3: any with RRPV==3
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3)
                return static_cast<int>(w);
        }
        // Age all RRPVs (saturating) and retry
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // Fallback (should not reach)
    return 0;
}

// ---------------- Periodic decay ----------------
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of PC run-length and clear last_line occasionally (soft reset)
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t idx = (decay_ptr_pc + i) & (PC_IDX_SIZE - 1);
        if (pc_run[idx] > 0) pc_run[idx]--;
        if ((i & 3) == 0) pc_last_line[idx] = 0;
    }
    decay_ptr_pc = (decay_ptr_pc + 13) & (PC_IDX_SIZE - 1);

    // Decay a small stripe of region hotness
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr_reg + (i * 5)) & (REG_HOT_SIZE - 1);
        if (region_hot[idx] > 0) region_hot[idx]--;
    }
    decay_ptr_reg = (decay_ptr_reg + 21) & (REG_HOT_SIZE - 1);

    // Bleed set scan debt gradually
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t s = (decay_ptr_set + (i * 7)) & (LLC_SETS - 1);
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    decay_ptr_set = (decay_ptr_set + 33) & (LLC_SETS - 1);
}

// ---------------- Streaming sentinel update ----------------
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (!is_demandish(type)) return false;

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

    // Update per-set scan pressure
    if (pc_run[pc_idx] >= STREAM_THR) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    }

    return (pc_run[pc_idx] >= STREAM_THR) || (set_scan_debt[set] >= 2);
}

// ---------------- CRC2 interface ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv   = 3;
            meta[s][w].mhits  = 0;
            meta[s][w].pf     = 0;
            meta[s][w].stream = 0;
            meta[s][w].sig    = 0;
            meta[s][w].valid  = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    for (uint32_t i = 0; i < REG_HOT_SIZE; i++) region_hot[i] = 0;
    for (uint32_t i = 0; i < SHIP_ENTRIES; i++) ship_ctr[i] = 1; // neutral
    event_ctr = 0;
    decay_ptr_pc = decay_ptr_reg = decay_ptr_set = 0;
}

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
    int v = pick_rrip_victim(set, current_set);
    if (v < 0 || v >= static_cast<int>(LLC_WAYS)) v = 0;
    return static_cast<uint32_t>(v);
}

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
    uint64_t ln = line_number(paddr);
    bool streamish = update_stream_sentinel(set, pc_idx, ln, type);
    uint32_t reg_idx = reg_index(paddr);
    uint8_t hot = region_hot[reg_idx];

    // On hit: gated promotion and SHiP reinforcement
    if (hit) {
        // Demand hits strengthen hotness and ship; stream-clamped lines do not promote
        if (is_demand(type)) {
            if (region_hot[reg_idx] < 3) region_hot[reg_idx]++;
            if (meta[set][way].mhits == 0) {
                meta[set][way].mhits = 1;
                // First demand re-reference => alive
                uint8_t sig = meta[set][way].sig;
                if (ship_ctr[sig] < 3) ship_ctr[sig]++;
                // Gentle nudge toward MRU but keep room for confirmation
                if (!meta[set][way].stream) {
                    if (meta[set][way].rrpv > 1) meta[set][way].rrpv = 1;
                }
            } else {
                // Confirmed reuse: promote to MRU unless stream-clamped
                if (!meta[set][way].stream) meta[set][way].rrpv = 0;
            }
        }
        // Prefetch hits do not trigger immediate promotion (handled by demand path)
        return;
    }

    // Miss path (fill): update SHiP on eviction of old line in this way
    LineMeta old = meta[set][way];
    if (old.valid) {
        if (old.mhits == 0 && old.pf == 0) {
            // Dead on arrival -> penalize signature
            if (ship_ctr[old.sig] > 0) ship_ctr[old.sig]--;
        } else {
            // Beneficial -> reward signature
            if (ship_ctr[old.sig] < 3) ship_ctr[old.sig]++;
        }
    }

    // Decide insertion using streaming clamp, SHiP gating, and region hotness
    uint8_t sig = ship_sig(PC);
    bool predicted_hot = (ship_ctr[sig] >= 2);
    bool hot_override = (hot >= HOT_THR);
    bool clamp_stream = streamish && (type != WRITEBACK) && !hot_override;

    // Initialize new line metadata
    meta[set][way].valid = 1;
    meta[set][way].pf = (type == PREFETCH) ? 1 : 0;
    meta[set][way].stream = clamp_stream ? 1 : 0;
    meta[set][way].mhits = 0;
    meta[set][way].sig = sig;

    // Insertion position (SRRIP RRPV)
    if (type == WRITEBACK) {
        // Never bypass writebacks: moderate priority
        meta[set][way].rrpv = 2;
    } else if (clamp_stream) {
        // Approximate bypass: tail insert, no promotion
        meta[set][way].rrpv = 3;
    } else if (meta[set][way].pf) {
        // Prefetch: always tail, confirm before promotion on later demand hits
        meta[set][way].rrpv = 3;
    } else if (predicted_hot || hot_override) {
        // Friendly PC or hot region => near-MRU
        meta[set][way].rrpv = 1;
    } else {
        // Cold/unknown => conservative
        meta[set][way].rrpv = 3; // more aggressive tail to avoid pollution
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}