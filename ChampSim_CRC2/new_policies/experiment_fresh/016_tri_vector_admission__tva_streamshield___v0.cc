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
#define PC_IDX_SIZE 2048          // PC-indexed tables (power of two)
#define REG_HOT_SIZE 1024         // Region hotness table (page-grain)
#define LAST_LINE_BITS 12         // Low bits of line number for stride compare
#define DECAY_PERIOD 4096         // Periodic decay cadence
#define STREAM_THR 2              // >=2 consecutive +1 strides => stream
#define SCAN_DEBT_MAX 3           // Per-set scan pressure cap

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B line
static inline uint64_t page_number(uint64_t paddr) { return (paddr >> 12); } // 4KB page

static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix; PC_IDX_SIZE must be power of two
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t reg_index(uint64_t paddr) {
    uint64_t pg = page_number(paddr);
    uint64_t mix = pg ^ (pg >> 7) ^ (pg << 11);
    return static_cast<uint32_t>(mix) & (REG_HOT_SIZE - 1);
}

// ---------------- Per-line metadata (packed conceptually to 10 bits) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ttl;     // 0..3 survival priority
    uint8_t mhits;   // 0..3 demand hit count (multi-hit gating)
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped (no promotion)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set scan pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed stream sentinel ----------------
static uint16_t pc_last_line[PC_IDX_SIZE]; // LAST_LINE_BITS of line number
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3 (+1 stride run)

// ---------------- PC bigram reuse cue ----------------
static uint8_t pc_prev_hit[PC_IDX_SIZE];   // 0/1: last access by this PC hit in LLC?
static uint8_t pc_score[PC_IDX_SIZE];      // 2-bit goodness (inc on multi-hit)

// ---------------- Region hotness filter ----------------
static uint8_t region_hot[REG_HOT_SIZE];   // 2-bit hotness per page bucket

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint16_t decay_ptr_pc = 0;
static uint16_t decay_ptr_reg = 0;
static uint16_t decay_ptr_set = 0;

// ---------------- LRU helpers ----------------
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

// ---------------- Periodic decay ----------------
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of PC tables
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr_pc + i) & (PC_IDX_SIZE - 1);
        if (pc_score[idx] > 0) pc_score[idx]--;
        // Occasionally clear prev_hit to force re-validation
        if ((i & 3) == 0 && pc_prev_hit[idx] > 0) pc_prev_hit[idx]--;
    }
    decay_ptr_pc = (decay_ptr_pc + 17) & (PC_IDX_SIZE - 1);

    // Decay a small stripe of region hotness
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr_reg + (i * 3)) & (REG_HOT_SIZE - 1);
        if (region_hot[idx] > 0) region_hot[idx]--;
    }
    decay_ptr_reg = (decay_ptr_reg + 9) & (REG_HOT_SIZE - 1);

    // Bleed set scan debt gradually
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t s = (decay_ptr_set + (i * 7)) & (LLC_SETS - 1);
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    decay_ptr_set = (decay_ptr_set + 33) & (LLC_SETS - 1);
}

// ---------------- Streaming sentinel update ----------------
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

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

    // Update set scan pressure
    if (pc_run[pc_idx] >= STREAM_THR) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    } else {
        if (set_scan_debt[set] > 0 && ((event_ctr & 0x1F) == 0)) set_scan_debt[set]--;
    }

    return (pc_run[pc_idx] >= STREAM_THR) || (set_scan_debt[set] >= 2);
}

// ---------------- Victim scoring ----------------
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                           // older is better victim
    s += (3 - m.ttl) * 8;                 // low TTL preferred
    s += (m.mhits == 0) ? 8 : ((m.mhits == 1) ? 2 : 0); // single/no-hit preferred
    s += m.pf ? 5 : 0;                    // prefetch quarantined
    s += m.stream ? 7 : 0;                // stream-clamped preferred
    return s;
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
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_last_line[i] = 0;
        pc_run[i] = 0;
        pc_prev_hit[i] = 0;
        pc_score[i] = 1; // slightly conservative start
    }
    for (uint32_t i = 0; i < REG_HOT_SIZE; i++) region_hot[i] = 0;

    event_ctr = 0;
    decay_ptr_pc = decay_ptr_reg = decay_ptr_set = 0;
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

    // Choose highest-scoring victim
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        // Tie-break on larger LRU to stabilize victim choice
        if ((sc > best_score) || (sc == best_score && meta[set][w].lru > meta[set][best_way].lru)) {
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
    uint32_t reg_idx = reg_index(paddr);
    uint64_t line = line_number(paddr);

    // Update streaming predictor and set pressure
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    if (hit) {
        // Accessed an existing line
        if (type != WRITEBACK) {
            // Update PC and region signals on any hit by this PC
            pc_prev_hit[pc_idx] = 1;
            if (region_hot[reg_idx] < 3) region_hot[reg_idx]++;

            // Multi-hit gating: never promote on first demand hit or on prefetch hits
            if (is_demand(type)) {
                if (meta[set][way].mhits < 3) meta[set][way].mhits++;
                if (!meta[set][way].stream) {
                    if (meta[set][way].mhits >= 2) {
                        // Promote after second demand hit
                        lru_make_mru(set, way);
                        if (meta[set][way].ttl < 3) meta[set][way].ttl++;
                        if (pc_score[pc_idx] < 3) pc_score[pc_idx]++; // strengthen PC goodness
                    } // else: no promotion on first demand hit
                }
            } else if (type == PREFETCH) {
                // No promotion; remain quarantined
            }
        } else { // WRITEBACK hit (rare path): be conservative
            if (meta[set][way].ttl < 2) meta[set][way].ttl++;
            // Do not aggressively promote writebacks
        }
        return;
    }

    // Miss handling (insertion at 'way')
    // Form Tri-Vector Admission decision
    bool streaming = stream_pred; // strong clamp
    bool pc_hot = (pc_prev_hit[pc_idx] != 0) || (pc_score[pc_idx] >= 2);
    bool reg_hot = (region_hot[reg_idx] >= 2);

    // Default insert: tail with minimal TTL
    uint8_t ins_pos = LLC_WAYS - 1;
    uint8_t ins_ttl = 0;
    uint8_t ins_stream = streaming ? 1 : 0;
    uint8_t ins_pf = (type == PREFETCH) ? 1 : 0;

    // If streaming predicted, clamp: tail insert, no promotion ever
    if (!streaming) {
        // Non-streaming: combine PC and region cues
        bool warm = pc_hot || reg_hot;
        if (warm && type != PREFETCH) {
            // Favor moderate insertion and survival
            ins_pos = (LLC_WAYS > 3) ? 2 : 1; // near-MRU but not top
            ins_ttl = 2;
        }
    }

    // Initialize new line metadata
    meta[set][way].pf = ins_pf;
    meta[set][way].stream = ins_stream;
    meta[set][way].mhits = 0;       // must earn promotion
    meta[set][way].ttl = ins_ttl;
    lru_insert_pos(set, way, ins_pos);

    // Update PC/region on miss
    if (type != WRITEBACK) {
        pc_prev_hit[pc_idx] = 0; // record that last access by this PC missed
        // Leave pc_score to decay; don't punish immediately to avoid oscillation
        if (region_hot[reg_idx] > 0 && streaming) {
            // If region is hot but access is streaming, lightly cool it
            region_hot[reg_idx]--;
        }
    } else {
        // Writeback insert: ensure not bypassed; moderate survival
        meta[set][way].ttl = (meta[set][way].ttl < 1) ? 1 : meta[set][way].ttl;
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