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
#define PC_SIG_BITS 11                // log2(PC_IDX_SIZE)
#define SAMP_PERIOD 32                // sample 1/32 sets => 64 sampled sets
#define SAMP_SETS (LLC_SETS / SAMP_PERIOD)
#define SAMP_TAG_BITS 16              // store low bits of line number
#define BLOOM_BITS 16384              // 2KB bloom
#define BLOOM_BYTES (BLOOM_BITS/8)
#define DECAY_PERIOD 4096             // bloom decay cadence (power of two)
#define BLOOM_CLEAR_STRIDE 16         // bytes cleared per decay tick
#define STREAM_RUN_THR 2              // >=2 consecutive matching ±1 strides

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline bool is_sampled_set(uint32_t set) { return ((set % SAMP_PERIOD) == 0); }
static inline uint32_t samp_index(uint32_t set) { return (set / SAMP_PERIOD); }

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t mhits;   // 0..3 demand hit count (saturating)
    uint8_t pf;      // 0/1 filled by prefetch
    uint8_t stream;  // 0/1 stream-clamped: never promote
    uint8_t protect; // 0 probation, 1 protected queue
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-PC tables ----------------
// SHiP-lite: 2-bit usefulness counter (0..3): trained by sampler (used=>inc, unused=>dec)
static uint8_t pc_useful[PC_IDX_SIZE];     // store 2-bit values in bytes (report minimal)
// Stream sentinel state per PC
static uint16_t pc_last_line[PC_IDX_SIZE]; // low bits of line number
static int8_t  pc_last_delta[PC_IDX_SIZE]; // last stride clipped to {-1,0,+1}
static uint8_t pc_run[PC_IDX_SIZE];        // 0..3 run length of consistent ±1 strides

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

// ---------------- LRU helpers for main LLC ----------------
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

// ---------------- LRU helpers for sampler ----------------
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

// ---------------- Bloom hashing ----------------
static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline void bloom_set_bit(uint32_t bit) {
    uint32_t idx = bit & (BLOOM_BITS - 1);
    bloom[idx >> 3] |= (1u << (idx & 7));
}
static inline bool bloom_test_bit(uint32_t bit) {
    uint32_t idx = bit & (BLOOM_BITS - 1);
    return (bloom[idx >> 3] & (1u << (idx & 7))) != 0;
}
static inline void bloom_add(uint32_t key) {
    uint32_t h1 = mix32(key);
    uint32_t h2 = mix32(key ^ 0x5bd1e995U);
    bloom_set_bit(h1);
    bloom_set_bit(h2);
}
static inline bool bloom_query(uint32_t key) {
    uint32_t h1 = mix32(key);
    uint32_t h2 = mix32(key ^ 0x5bd1e995U);
    return bloom_test_bit(h1) && bloom_test_bit(h2);
}
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    for (uint32_t i = 0; i < BLOOM_CLEAR_STRIDE; i++) {
        uint32_t idx = (bloom_decay_ptr + i) & (BLOOM_BYTES - 1);
        bloom[idx] = 0;
    }
    bloom_decay_ptr = (bloom_decay_ptr + BLOOM_CLEAR_STRIDE) & (BLOOM_BYTES - 1);
}

// ---------------- Stream sentinel ----------------
static inline bool update_stream_and_predict(uint32_t pc_idx, uint64_t line, uint32_t type) {
    // Track only LOAD/RFO/PREFETCH; ignore WRITEBACK for stream decisions
    int8_t last_d = pc_last_delta[pc_idx];
    uint16_t last_l = pc_last_line[pc_idx];
    int64_t d64 = static_cast<int64_t>(line) - static_cast<int64_t>(last_l);
    int8_t d = 0;
    if (d64 > 0) d = (d64 >= 1) ? (d64 == 1 ? 1 : 0) : 0;
    if (d64 < 0) d = (d64 <= -1) ? (d64 == -1 ? -1 : 0) : d;

    if (d == 1 || d == -1) {
        if (pc_run[pc_idx] == 0 || d == last_d) {
            if (pc_run[pc_idx] < 3) pc_run[pc_idx]++;
        } else {
            pc_run[pc_idx] = 1;
        }
        pc_last_delta[pc_idx] = d;
    } else {
        pc_run[pc_idx] = 0;
        pc_last_delta[pc_idx] = 0;
    }
    pc_last_line[pc_idx] = static_cast<uint16_t>(line & 0xFFFFULL);
    // Predict stream if confidence reached and not a writeback
    bool predict_stream = (pc_run[pc_idx] >= STREAM_RUN_THR) && (type != WRITEBACK);
    return predict_stream;
}

// ---------------- SHiP training via sampler ----------------
static inline void ship_train_on_eviction(const SampEntry& e) {
    uint32_t idx = e.pc_sig & (PC_IDX_SIZE - 1);
    // Update per-PC usefulness
    uint8_t u = pc_useful[idx] & 0x3;
    if (e.used) {
        if (u < 3) u++;
    } else {
        if (u > 0) u--;
        // Unused => remember as dead in Bloom using compact PC⊕tag key
        uint32_t key = (static_cast<uint32_t>(idx) << 16) ^ static_cast<uint32_t>(e.tag);
        bloom_add(key);
    }
    pc_useful[idx] = u;
}

// ---------------- API functions ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].protect = 0;
        }
    }

    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        pc_useful[i] = 1; // weak-cold start
        pc_last_line[i] = 0;
        pc_last_delta[i] = 0;
        pc_run[i] = 0;
    }

    for (uint32_t i = 0; i < BLOOM_BYTES; i++) bloom[i] = 0;
    bloom_decay_ptr = 0;
    event_ctr = 0;

    for (uint32_t s = 0; s < SAMP_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sampler[s][w].tag = 0;
            sampler[s][w].pc_sig = 0;
            sampler[s][w].lru = w;
            sampler[s][w].used = 0;
            sampler[s][w].valid = 0;
        }
    }
}

// Victim selection prefers probation (protect=0) and within it stream/prefetch/single-hit
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return first invalid if present
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // First pick among probation (protect=0)
    int best_lru = -1;
    int best_bias = -1;
    uint32_t best_way = LLC_WAYS;

    auto bias_fn = [&](uint32_t w)->int {
        int b = 0;
        if (meta[set][w].stream) b += 2;
        if (meta[set][w].pf) b += 2;
        if (meta[set][w].mhits == 0) b += 1;
        return b;
    };

    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].protect) continue;
        int l = static_cast<int>(meta[set][w].lru);
        int b = bias_fn(w);
        if (l > best_lru || (l == best_lru && b > best_bias)) {
            best_lru = l;
            best_bias = b;
            best_way = w;
        }
    }
    if (best_way != LLC_WAYS) return best_way;

    // Otherwise pick among all, again favoring stream/pf/single-hit on ties
    best_lru = -1; best_bias = -1; best_way = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        int l = static_cast<int>(meta[set][w].lru);
        int b = bias_fn(w);
        if (w == 0 || l > best_lru || (l == best_lru && b > best_bias)) {
            best_lru = l;
            best_bias = b;
            best_way = w;
        }
    }
    return best_way;
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

    uint64_t ln = line_number(paddr);
    uint32_t pc_idx = pc_index(PC);

    // Sampler: observe hits to sampled sets (demand only)
    if (is_sampled_set(set)) {
        uint32_t sidx = samp_index(set);
        uint16_t stag = static_cast<uint16_t>(ln & ((1u << SAMP_TAG_BITS) - 1));
        // Mark used on demand hit lookups
        if (is_demand(type)) {
            for (uint32_t w2 = 0; w2 < LLC_WAYS; w2++) {
                if (sampler[sidx][w2].valid && sampler[sidx][w2].tag == stag) {
                    sampler[sidx][w2].used = 1;
                    samp_make_mru(sidx, w2);
                    break;
                }
            }
        }
    }

    // On hit
    if (hit) {
        // Prefetch hits never promote
        if (type == PREFETCH) return;

        // Stream-clamped lines never promote
        if (meta[set][way].stream) return;

        // Demand hit promotion with multi-hit threshold
        if (is_demand(type)) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;
            meta[set][way].pf = 0; // consumed by demand
            if (meta[set][way].mhits == 1) {
                // First hit: gentle touch, move to mid
                lru_insert_pos(set, way, 7);
            } else {
                // Second or later hit: protect and MRU
                meta[set][way].protect = 1;
                lru_make_mru(set, way);
            }
        }
        return;
    }

    // Miss path: determine insertion policy
    bool predict_stream = update_stream_and_predict(pc_idx, ln, type);
    // Dead-address Bloom query (PC⊕line)
    uint32_t bloom_key = static_cast<uint32_t>((ln ^ (ln >> 17) ^ static_cast<uint32_t>(PC) ^ static_cast<uint32_t>(PC >> 32)) & 0xFFFFFFFFu);
    bool bloom_dead = bloom_query(bloom_key);
    bool hot_pc = ((pc_useful[pc_idx] & 0x3) >= 2);
    bool is_pf = (type == PREFETCH);

    uint8_t ins_pos = 15;   // default tail
    uint8_t ins_stream = 0;

    if (type == WRITEBACK) {
        // Never bypass on writeback; keep it probation and low priority
        ins_pos = 15;
        ins_stream = 0;
    } else if (is_pf) {
        ins_pos = 15;              // keep prefetches cold
        ins_stream = predict_stream ? 1 : 0;
    } else if (predict_stream || bloom_dead) {
        ins_pos = 15;              // clamp scans and dead PCs
        ins_stream = predict_stream ? 1 : 0;
    } else if (hot_pc) {
        ins_pos = 3;               // hot PC: near-MRU, but still probation
        ins_stream = 0;
    } else {
        ins_pos = 14;              // cold (non-stream) demand: deep probation
        ins_stream = 0;
    }

    // Install metadata
    meta[set][way].mhits = 0;
    meta[set][way].pf = is_pf ? 1 : 0;
    meta[set][way].stream = ins_stream;
    meta[set][way].protect = 0; // all admissions start in probation
    lru_insert_pos(set, way, ins_pos);

    // Sampler: on fills for sampled sets, insert and train on eviction
    if (is_sampled_set(set) && is_demandish(type)) {
        uint32_t sidx = samp_index(set);
        uint32_t wv = LLC_WAYS; // find invalid
        for (uint32_t w2 = 0; w2 < LLC_WAYS; w2++) {
            if (!sampler[sidx][w2].valid) { wv = w2; break; }
        }
        if (wv == LLC_WAYS) {
            wv = samp_lru_victim(sidx);
            if (sampler[sidx][wv].valid) {
                ship_train_on_eviction(sampler[sidx][wv]);
            }
        }
        sampler[sidx][wv].tag = static_cast<uint16_t>(ln & ((1u << SAMP_TAG_BITS) - 1));
        sampler[sidx][wv].pc_sig = static_cast<uint16_t>(pc_idx & ((1u << PC_SIG_BITS) - 1));
        sampler[sidx][wv].used = 0;
        sampler[sidx][wv].valid = 1;
        samp_make_mru(sidx, wv);
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}