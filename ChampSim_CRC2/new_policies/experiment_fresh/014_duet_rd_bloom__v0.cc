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
#define PC_IDX_SIZE 2048              // per-PC tables (power of two)
#define SAMP_PERIOD 32                // sample 1/32 sets => 64 sampled sets
#define SAMP_SETS (LLC_SETS / SAMP_PERIOD)
#define PC_SIG_BITS 11                // log2(PC_IDX_SIZE)
#define SAMP_TAG_BITS 16              // store low bits of line number
#define BLOOM_BITS 16384              // 2KB bloom
#define BLOOM_BYTES (BLOOM_BITS/8)
#define BLOOM_CLEAR_STRIDE 16         // bytes cleared per decay tick
#define DECAY_PERIOD 4096             // bloom decay cadence
#define STREAM_RUN_THR 2              // >=2 consecutive +1 strides => stream

// ---------------- Per-line metadata (8 bits packed conceptually) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t mhits;   // 0..3 demand hit count
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped: never promote
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-PC tables ----------------
// pc_useful: 2-bit saturating counter (0..3): trained by sampler (evict-used => inc, evict-unused => dec)
static uint8_t pc_useful[PC_IDX_SIZE];     // 2 bits each (stored in byte)
// stream sentinel: last line low bits and +1 stride run count (0..3 but we use 0..3; threshold 2)
static uint16_t pc_last_line[PC_IDX_SIZE]; // store low 14..16 bits of line#
static uint8_t  pc_run[PC_IDX_SIZE];       // 0..3

// ---------------- Dead-address Bloom (PC⊕line) ----------------
static uint8_t bloom[BLOOM_BYTES];
static uint32_t bloom_decay_ptr = 0;

// ---------------- Sampler (shadow tag store for sampled sets) ----------------
struct SampEntry {
    uint16_t tag;      // low bits of line number
    uint16_t pc_sig;   // PC signature (PC_IDX)
    uint8_t  lru;      // 0..15
    uint8_t  used;     // 0/1: seen a demand hit before eviction
    uint8_t  valid;    // 0/1
};
static SampEntry sampler[SAMP_SETS][LLC_WAYS];

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

static inline bool is_sampled_set(uint32_t set) {
    return ((set % SAMP_PERIOD) == 0);
}
static inline uint32_t samp_index(uint32_t set) {
    return (set / SAMP_PERIOD);
}

// LRU helpers for main LLC metadata
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

// LRU helpers for sampler
static inline void samp_make_mru(uint32_t sidx, uint32_t way) {
    uint8_t old = sampler[sidx][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (sampler[sidx][w].lru < old) {
            uint8_t np = sampler[sidx][w].lru + 1;
            sampler[sidx][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    sampler[sidx][way].lru = 0;
}
static inline uint32_t samp_lru_victim(uint32_t sidx) {
    uint32_t vic = 0;
    uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == 0 || sampler[sidx][w].lru > best) {
            best = sampler[sidx][w].lru;
            vic = w;
        }
    }
    return vic;
}

// 32-bit mix for Bloom hashing
static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline void bloom_set_bits(uint32_t h) {
    uint32_t idx = h & (BLOOM_BITS - 1);
    bloom[idx >> 3] |= (1u << (idx & 7));
}
static inline bool bloom_test_bits(uint32_t h) {
    uint32_t idx = h & (BLOOM_BITS - 1);
    return (bloom[idx >> 3] & (1u << (idx & 7))) != 0;
}
static inline void bloom_add(uint32_t key) {
    uint32_t h1 = mix32(key);
    uint32_t h2 = mix32(key ^ 0x5bd1e995U);
    bloom_set_bits(h1);
    bloom_set_bits(h2);
}
static inline bool bloom_query(uint32_t key) {
    uint32_t h1 = mix32(key);
    uint32_t h2 = mix32(key ^ 0x5bd1e995U);
    return bloom_test_bits(h1) && bloom_test_bits(h2);
}

// Periodic soft decay for Bloom (clears a stripe)
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    for (uint32_t i = 0; i < BLOOM_CLEAR_STRIDE; i++) {
        uint32_t idx = (bloom_decay_ptr + i) & (BLOOM_BYTES - 1);
        bloom[idx] = 0;
    }
    bloom_decay_ptr = (bloom_decay_ptr + BLOOM_CLEAR_STRIDE) & (BLOOM_BYTES - 1);
}

// Update stream sentinel; returns if this access looks like a stream
static inline bool update_stream(uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;
    uint16_t cur = static_cast<uint16_t>(line & ((1u << SAMP_TAG_BITS) - 1u));
    uint16_t prev = pc_last_line[pc_idx];
    int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(prev);
    if (stride == 1) {
        if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
    } else if (cur != prev) {
        if (pc_run[pc_idx] > 0) pc_run[pc_idx]--;
    }
    pc_last_line[pc_idx] = cur;
    return (pc_run[pc_idx] >= STREAM_RUN_THR);
}

// Sampler access and training
static inline void sampler_access(uint32_t set, uint64_t line, uint32_t pc_idx, uint32_t type) {
    if (!is_sampled_set(set)) return;
    uint32_t sidx = samp_index(set);
    uint16_t tag = static_cast<uint16_t>(line & ((1u << SAMP_TAG_BITS) - 1u));

    // lookup
    int32_t hit_way = -1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (sampler[sidx][w].valid && sampler[sidx][w].tag == tag) {
            hit_way = static_cast<int32_t>(w);
            break;
        }
    }

    if (hit_way >= 0) {
        if (is_demand(type)) sampler[sidx][hit_way].used = 1;
        samp_make_mru(sidx, static_cast<uint32_t>(hit_way));
        return;
    }

    // miss in sampler: insert, evict LRU
    uint32_t vic = samp_lru_victim(sidx);
    if (sampler[sidx][vic].valid) {
        uint16_t old_pc = sampler[sidx][vic].pc_sig & ((1u << PC_SIG_BITS) - 1u);
        if (sampler[sidx][vic].used) {
            if (pc_useful[old_pc] < 3) pc_useful[old_pc]++;
        } else {
            if (pc_useful[old_pc] > 0) pc_useful[old_pc]--;
            // add dead sample to Bloom: key = old_pc ⊕ tag
            uint32_t key = (static_cast<uint32_t>(old_pc) << 16) ^ static_cast<uint32_t>(sampler[sidx][vic].tag);
            bloom_add(key);
        }
    }

    // insert new sample entry
    sampler[sidx][vic].valid = 1;
    sampler[sidx][vic].tag = tag;
    sampler[sidx][vic].pc_sig = static_cast<uint16_t>(pc_idx & ((1u << PC_SIG_BITS) - 1u));
    sampler[sidx][vic].used = 0;
    // make MRU in sampler
    uint8_t old = sampler[sidx][vic].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == vic) continue;
        if (sampler[sidx][w].lru <= old) {
            uint8_t np = sampler[sidx][w].lru + 1;
            sampler[sidx][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    sampler[sidx][vic].lru = 0;
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                         // prefer older
    s += (m.mhits == 0) ? 8 : 0;        // prefer single/no-hit
    s += m.pf ? 6 : 0;                  // prefer prefetch
    s += m.stream ? 10 : 0;             // prefer stream-clamped
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = static_cast<uint8_t>(w);
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_useful[i] = 1;       // slightly cold to start
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    for (uint32_t s = 0; s < SAMP_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sampler[s][w].valid = 0;
            sampler[s][w].tag = 0;
            sampler[s][w].pc_sig = 0;
            sampler[s][w].used = 0;
            sampler[s][w].lru = static_cast<uint8_t>(w);
        }
    }
    for (uint32_t i = 0; i < BLOOM_BYTES; i++) bloom[i] = 0;
    bloom_decay_ptr = 0;
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

    uint64_t line = line_number(paddr);
    uint32_t pc_idx = pc_index(PC);

    // Update sampler for training (demand/prefetch both do lookups; only demand sets 'used')
    sampler_access(set, line, pc_idx, type);

    // Update stream sentinel
    bool is_stream = update_stream(pc_idx, line, type);

    if (hit) {
        // On hits: do not change policy for writebacks
        if (type == WRITEBACK) return;

        // First, unquarantine prefetched lines on first demand hit
        if (is_demand(type) && meta[set][way].pf) {
            meta[set][way].pf = 0;
        }

        // Multi-hit gating and stream clamp
        if (is_demand(type)) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;
        }

        if (meta[set][way].stream) {
            // never promote stream-clamped lines
            return;
        }

        // Multi-hit promotion: no promotion on first demand hit
        if (is_demand(type)) {
            if (meta[set][way].mhits <= 1) {
                // gentle nudge: move toward middle
                uint8_t target = (LLC_WAYS >= 4) ? (LLC_WAYS / 2) : (LLC_WAYS - 2);
                lru_insert_pos(set, way, target);
            } else {
                lru_make_mru(set, way);
            }
        } else {
            // prefetch hit: do not promote
        }
        return;
    }

    // Miss: a new line was inserted in 'way' (no bypass on WRITEBACK)
    bool bloom_dead = false;
    if (type != WRITEBACK) {
        uint32_t key = (pc_idx << 16) ^ static_cast<uint32_t>(line & ((1u << SAMP_TAG_BITS) - 1u));
        bloom_dead = bloom_query(key);
    }
    bool rd_cold = (pc_useful[pc_idx] <= 1);
    bool cold = (type != WRITEBACK) && (is_stream || bloom_dead || rd_cold);

    // Reset per-line metadata
    meta[set][way].mhits = 0;
    meta[set][way].pf = (type == PREFETCH) ? 1 : 0;
    meta[set][way].stream = is_stream ? 1 : 0;

    // Choose insertion position
    if (type == WRITEBACK) {
        // writebacks: safe tail insertion
        lru_insert_pos(set, way, LLC_WAYS - 1);
    } else if (cold) {
        // cold/stream/Bloom => tail insert, stream clamp keeps it from promoting
        lru_insert_pos(set, way, LLC_WAYS - 1);
    } else {
        // hot demand: near-MRU insert
        uint8_t pos = (LLC_WAYS >= 4) ? 2 : 0;
        lru_insert_pos(set, way, pos);
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