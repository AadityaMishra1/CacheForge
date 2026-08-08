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
#define PC_SIG_ENTRIES 1024      // unified PC⊕page signature table entries (10-bit index)
#define PILOT_SETS 64            // number of sampled pilot sets
#define PILOT_TAG_BITS 16        // 16-bit tag signatures in pilot
#define RUN_THR 2                // >=2 consecutive +1 strides => streaming
#define TTL_MAX 3
#define DECAY_PERIOD 4096        // light decay cadence

// ---------------- Per-line metadata (packed conceptually: 11 bits/line) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ttl;     // 0..3 small survival budget
    uint8_t mhits;   // 0..3 demand hit count (multi-hit gating)
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped (never promote)
    uint8_t cold;    // 0/1 predicted cold at fill (PC utility)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Unified PC⊕page sentinel + utility (16 bits/entry) ----------------
// Index key: hash(PC xor page) -> entry; fields:
// - last_line_low12: low 12 bits of last line number (64B lines)
// - run2: 2-bit +1 stride runlength
// - util2: 2-bit utility (0=cold..3=hot)
static uint16_t pcsig_last_line[PC_SIG_ENTRIES]; // 12-bit stored
static uint8_t  pcsig_run[PC_SIG_ENTRIES];       // 0..3
static uint8_t  pcsig_util[PC_SIG_ENTRIES];      // 0..3

// ---------------- Pilot sampler (64 sampled sets × 16 entries) ----------------
// Each entry keeps a small tag signature, its PC-signature index, valid flag, and a "used" hit flag.
static uint16_t pilot_tag[PILOT_SETS][LLC_WAYS];     // 16-bit line signature
static uint16_t pilot_pcidx[PILOT_SETS][LLC_WAYS];   // 10-bit pc-sig index (stored in 16-bit)
static uint8_t  pilot_valid[PILOT_SETS][LLC_WAYS];   // 0/1
static uint8_t  pilot_used[PILOT_SETS][LLC_WAYS];    // 0/1 (set on pilot hit)

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline uint64_t page_number(uint64_t paddr) { return (paddr >> 12); }

static inline uint32_t pc_sig_index(uint64_t PC, uint64_t paddr) {
    uint64_t page = page_number(paddr);
    uint64_t mix = (PC ^ (page << 1) ^ (page >> 1));
    mix ^= (mix >> 13);
    mix *= 0x9e3779b97f4a7c15ULL;
    mix ^= (mix >> 17);
    return static_cast<uint32_t>(mix) & (PC_SIG_ENTRIES - 1);
}
static inline uint16_t line_sig16(uint64_t line) {
    uint64_t x = line * 0x9e3779b97f4a7c15ULL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    return static_cast<uint16_t>(x & ((1u << PILOT_TAG_BITS) - 1));
}
static inline bool is_pilot_set(uint32_t set) {
    // 2048 sets / 64 = 32: choose every 32nd set
    return ((set % (LLC_SETS / PILOT_SETS)) == 0);
}
static inline uint32_t pilot_set_index(uint32_t set) {
    return (set / (LLC_SETS / PILOT_SETS)) & (PILOT_SETS - 1);
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

// Periodic light decay for runlengths and utility smoothing
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Bleed a few pcsig_run entries and nudge utility slightly toward cold
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr + (i * 31)) & (PC_SIG_ENTRIES - 1);
        if (pcsig_run[idx] > 0) pcsig_run[idx]--;
        if (pcsig_util[idx] > 0) pcsig_util[idx]--;
    }
    decay_ptr = (decay_ptr + 1) & (PC_SIG_ENTRIES - 1);
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t pcsi, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;

    uint16_t cur = static_cast<uint16_t>(line & 0xFFF); // low 12 bits
    uint16_t prev = pcsig_last_line[pcsi];
    uint16_t delta = static_cast<uint16_t>((cur - prev) & 0x0FFF);

    if (delta == 1) {
        if (pcsig_run[pcsi] < 3) pcsig_run[pcsi]++;
    } else if (cur != prev) {
        if (pcsig_run[pcsi] > 0) pcsig_run[pcsi]--;
    }
    pcsig_last_line[pcsi] = cur;

    return (pcsig_run[pcsi] >= RUN_THR);
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                                  // older is better victim
    s += (3 - m.ttl) * 6;                        // low TTL preferred
    s += (m.mhits == 0) ? 5 : ((m.mhits == 1) ? 2 : 0); // zero/single-hit preferred
    s += m.pf ? 4 : 0;                           // prefetch quarantined
    s += m.stream ? 7 : 0;                       // stream-clamped preferred
    s += m.cold ? 3 : 0;                         // predicted-cold preferred
    return s;
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
            meta[s][w].cold = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIG_ENTRIES; i++) {
        pcsig_last_line[i] = 0;
        pcsig_run[i] = 0;
        pcsig_util[i] = 1; // start slightly cold
    }
    for (uint32_t p = 0; p < PILOT_SETS; p++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            pilot_tag[p][w] = 0;
            pilot_pcidx[p][w] = 0;
            pilot_valid[p][w] = 0;
            pilot_used[p][w]  = 0;
        }
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
    // Prefer any invalid way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Score-based victim selection
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

    uint64_t line = line_number(paddr);
    uint32_t pcsi = pc_sig_index(PC, paddr);
    bool demand = is_demand(type);
    bool demandish = is_demandish(type);

    // Pilot sampler: probe for pilot hit (mark used)
    if (is_pilot_set(set) && (type != WRITEBACK)) {
        uint32_t ps = pilot_set_index(set);
        uint16_t sig = line_sig16(line);
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (pilot_valid[ps][w] && pilot_tag[ps][w] == sig) {
                pilot_used[ps][w] = 1; // mark reuse in pilot
                break;
            }
        }
        // On demand fills, insert into pilot with deterministic hashed victim
        if (!hit && demand) {
            uint32_t vict = (static_cast<uint32_t>(sig) ^ pcsi) & (LLC_WAYS - 1);
            if (pilot_valid[ps][vict]) {
                // Train utility from evicted pilot entry
                uint16_t old_idx = pilot_pcidx[ps][vict];
                if (pilot_used[ps][vict]) {
                    if (pcsig_util[old_idx] < 3) pcsig_util[old_idx]++;
                } else {
                    if (pcsig_util[old_idx] > 0) pcsig_util[old_idx]--;
                }
            }
            pilot_tag[ps][vict]   = sig;
            pilot_pcidx[ps][vict] = static_cast<uint16_t>(pcsi);
            pilot_valid[ps][vict] = 1;
            pilot_used[ps][vict]  = 0;
        }
    }

    // Streaming sentinel update
    bool stream_pred = update_stream_sentinel(pcsi, line, type);

    // Handle hit
    if (hit) {
        LineMeta &m = meta[set][way];
        if (type == WRITEBACK) {
            // treat as quiet writeback: keep at tail-ish
            lru_insert_pos(set, way, LLC_WAYS - 2);
            if (m.ttl > 0) m.ttl--; // gently age
            return;
        }

        if (m.stream) {
            // stream-clamped: never promote, keep evictable
            m.ttl = 0;
            m.mhits = 0;
            lru_insert_pos(set, way, LLC_WAYS - 2);
            return;
        }

        if (m.pf && demand) {
            // first demand after prefetch does not unlock promotion
            if (m.mhits == 0) {
                m.mhits = 1;
                lru_insert_pos(set, way, 4); // slight bump, not MRU
                m.cold = 0;
                return;
            }
        }

        if (demand) {
            if (m.mhits == 0) {
                // first demand hit: record but do not fully promote
                m.mhits = 1;
                m.cold = 0;
                lru_insert_pos(set, way, 3);
            } else {
                // confirmed reuse: promote and extend TTL
                if (m.ttl < TTL_MAX) m.ttl++;
                if (m.mhits < 3) m.mhits++;
                lru_make_mru(set, way);
            }
        } else {
            // prefetch hit or others: do minimal motion
            lru_insert_pos(set, way, 6);
        }
        return;
    }

    // Miss/Fill path (hit == 0): set insertion state
    LineMeta &mnew = meta[set][way];

    // Base defaults
    mnew.pf = (type == PREFETCH) ? 1 : 0;
    mnew.mhits = 0;
    mnew.stream = 0;
    mnew.cold = 0;
    mnew.ttl = 1;

    // Determine admission using PC utility and stream prediction (soft-bypass)
    uint8_t util = pcsig_util[pcsi];

    if (type != WRITEBACK && (stream_pred || util <= 1)) {
        // Soft-bypass: tail insert, never promote
        mnew.stream = stream_pred ? 1 : 0;
        mnew.cold = (util <= 1) ? 1 : 0;
        mnew.ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
        return;
    }

    // Admitted: adaptive insertion and gating
    if (mnew.pf) {
        // quarantine prefetches at tail
        mnew.ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
        return;
    } else {
        // demand fill: position based on utility
        if (util >= 3) {
            mnew.ttl = 2;
            lru_insert_pos(set, way, 4); // near-MRU
        } else {
            mnew.ttl = 1;
            lru_insert_pos(set, way, 8); // mid
        }
        mnew.cold = 0;
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