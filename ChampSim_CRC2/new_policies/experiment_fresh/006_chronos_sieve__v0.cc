#include <vector>
#include <cstdint>
#include <cstring>
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
#define PC_IDX_SIZE 2048          // PC hash entries (2K)
#define STREAM_RUN_THR 2          // >=2 consecutive +1 strides => streaming
#define SCAN_DEBT_MAX 3
#define EMA_SHIFT 2               // EMA alpha = 1/4
#define EPOCH_BITS 6              // 6-bit modular epoch (0..63)
#define DECAY_PERIOD 8192         // for Bloom generation rotation
#define SAMPLE_RATE 64            // 1-of-64 leader sets for PC doorkeeper training

// ---------------- Per-line metadata ----------------
// Stored compactly as arrays (info-theoretic: 13 bits/line)
// - lru   : 4 bits (0=MRU .. 15=LRU)
// - mhits : 2 bits (0..3 demand hits; gate promotion until >=2)
// - stream: 1 bit (stream-clamped; never promote)
// - due6  : 6 bits (deadline epoch modulo 64)
static uint8_t lru_pos[LLC_SETS][LLC_WAYS];
static uint8_t mhits[LLC_SETS][LLC_WAYS];
static uint8_t stream_flag[LLC_SETS][LLC_WAYS];
static uint8_t due6[LLC_SETS][LLC_WAYS];

// ---------------- Per-set stream pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed predictors ----------------
// Packed as separate arrays (info-theoretic: 26 bits/PC entry):
// - gap6     : 6-bit EMA of inter-access gap (proxy for reuse distance)
// - last_ep6 : 6-bit last access epoch
// - last_ln  : 12-bit last line number (low bits) for stride detection
// - run2     : 2-bit run length of +1 strides (0..3)
static uint8_t  PC_gap6[PC_IDX_SIZE];        // 0..63
static uint8_t  PC_last_ep6[PC_IDX_SIZE];    // 0..63
static uint16_t PC_last_ln12[PC_IDX_SIZE];   // 0..4095 (line# low 12 bits)
static uint8_t  PC_run2[PC_IDX_SIZE];        // 0..3

// ---------------- Doorkeeper (negative Bloom) ----------------
// Two-generation Bloom filters to decay cold-PC marks over time.
// Total bits: 2 * 2048 = 4096 bits.
static uint64_t bloom_gen[2][2048/64];
static uint8_t bloom_cur_gen = 0;

// ---------------- Sampled-set PC capture for doorkeeper training ----------------
static inline bool is_sampled(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
static inline uint32_t sample_set_idx(uint32_t set) { return set / SAMPLE_RATE; }
static uint16_t sample_pcidx[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // 11-bit effective
static uint8_t  sample_valid[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // 0/1

// ---------------- Global epoch ----------------
static uint64_t global_events = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline bool is_demandish(uint32_t type) { return (type == LOAD) || (type == RFO) || (type == PREFETCH); }
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 17) ^ (pc >> 33);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); }
static inline uint8_t epoch6_now() { return static_cast<uint8_t>(global_events & ((1u << EPOCH_BITS) - 1)); }
static inline uint8_t sat_u8(uint8_t v, uint8_t maxv) { return (v < maxv) ? (uint8_t)(v + 1) : maxv; }

// LRU primitives
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = lru_pos[set][way];
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (lru_pos[set][w] < old) {
            uint8_t np = lru_pos[set][w] + 1;
            lru_pos[set][w] = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    lru_pos[set][way] = 0;
}
static inline void lru_insert_at(uint32_t set, uint32_t way, uint8_t pos) {
    if (pos >= LLC_WAYS) pos = LLC_WAYS - 1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (lru_pos[set][w] <= pos) {
            uint8_t np = lru_pos[set][w] + 1;
            lru_pos[set][w] = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    lru_pos[set][way] = pos;
}

// Bloom helpers
static inline uint32_t bloom_h1(uint64_t pc) {
    uint64_t x = pc * 0x9e3779b97f4a7c15ULL;
    x ^= (x >> 33);
    return static_cast<uint32_t>(x) & (2048 - 1);
}
static inline uint32_t bloom_h2(uint64_t pc) {
    uint64_t x = (pc ^ (pc >> 17)) * 0xc6a4a7935bd1e995ULL;
    x ^= (x >> 33);
    return static_cast<uint32_t>(x) & (2048 - 1);
}
static inline void bloom_set(uint64_t pc) {
    uint32_t i1 = bloom_h1(pc), i2 = bloom_h2(pc);
    bloom_gen[bloom_cur_gen][i1 >> 6] |= (1ULL << (i1 & 63));
    bloom_gen[bloom_cur_gen][i2 >> 6] |= (1ULL << (i2 & 63));
}
static inline bool bloom_query(uint64_t pc) {
    uint32_t i1 = bloom_h1(pc), i2 = bloom_h2(pc);
    bool g0 = (bloom_gen[0][i1 >> 6] >> (i1 & 63)) & 1ULL;
    g0 = g0 && ((bloom_gen[0][i2 >> 6] >> (i2 & 63)) & 1ULL);
    bool g1 = (bloom_gen[1][i1 >> 6] >> (i1 & 63)) & 1ULL;
    g1 = g1 && ((bloom_gen[1][i2 >> 6] >> (i2 & 63)) & 1ULL);
    return g0 || g1;
}

// Stream sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (!is_demandish(type)) return false;
    uint16_t cur = static_cast<uint16_t>(line & 0xFFF); // 12-bit low line#
    int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(PC_last_ln12[pc_idx]);
    if (stride == 1) {
        if (PC_run2[pc_idx] < 3) PC_run2[pc_idx]++;
    } else {
        if (PC_run2[pc_idx] > 0) PC_run2[pc_idx]--;
    }
    PC_last_ln12[pc_idx] = cur;

    if (PC_run2[pc_idx] >= STREAM_RUN_THR) {
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++;
    } else {
        // occasional bleed
        if ( (global_events & 0x3F) == 0 && set_scan_debt[set] > 0 ) set_scan_debt[set]--;
    }

    return (PC_run2[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
}

// EMA training on inter-access gap (PC-level)
static inline void train_pc_gap(uint32_t pc_idx) {
    uint8_t now = epoch6_now();
    uint8_t last = PC_last_ep6[pc_idx];
    uint8_t delta = static_cast<uint8_t>((now - last) & ((1u << EPOCH_BITS) - 1));
    // use only if delta is in forward window (avoid wrap ambiguity)
    if (delta < 32 && delta > 0) {
        uint8_t g = PC_gap6[pc_idx];
        int16_t diff = static_cast<int16_t>(delta) - static_cast<int16_t>(g);
        g = static_cast<uint8_t>(g + (diff >> EMA_SHIFT));
        if (g == 0) g = 1; // avoid zero gap
        PC_gap6[pc_idx] = g;
    }
    PC_last_ep6[pc_idx] = now;
}

// Periodic maintenance: Bloom generation rotation and mild scan-debt bleed
static inline void periodic_maintenance() {
    if ((global_events % DECAY_PERIOD) == 0) {
        // rotate Bloom generation
        bloom_cur_gen ^= 1;
        std::memset(bloom_gen[bloom_cur_gen], 0, sizeof(bloom_gen[0]));
    }
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            lru_pos[s][w] = static_cast<uint8_t>(w);
            mhits[s][w] = 0;
            stream_flag[s][w] = 0;
            due6[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_gap6[i] = 8; // conservative initial gap
        PC_last_ep6[i] = 0;
        PC_last_ln12[i] = 0;
        PC_run2[i] = 0;
    }
    std::memset(bloom_gen, 0, sizeof(bloom_gen));
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_pcidx[ss][w] = 0;
            sample_valid[ss][w] = 0;
        }
    }
    global_events = 0;
}

// ---------------- Victim Selection ----------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Deadline-aware scoring
    uint8_t now = epoch6_now();
    uint32_t best_way = 0;
    int32_t best_score = -1;

    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint8_t due = due6[set][w] & 63;
        uint8_t overdue = static_cast<uint8_t>((now - due) & 63);
        bool expired = (overdue < 32);                 // due is behind within 32 ticks
        uint8_t forward = static_cast<uint8_t>((due - now) & 63);

        int32_t score = 0;
        // Primary: expired/overdue first, more overdue => higher score
        if (expired) score += 64 + overdue;
        else         score += (31 - (forward & 31));   // sooner deadline => higher

        // Secondary features
        score += (int32_t)lru_pos[set][w];             // older recency preferred
        if (mhits[set][w] == 0) score += 8;            // single/no-hit lines
        if (stream_flag[set][w]) score += 12;          // stream-clamped preferred

        if (score > best_score) {
            best_score = score;
            best_way = w;
        }
    }
    return best_way;
}

// ---------------- State Update ----------------
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

    // Advance epoch and periodic maintenance
    global_events++;
    periodic_maintenance();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);

    // Train PC inter-access EMA and update stream sentinel
    if (type != WRITEBACK) {
        train_pc_gap(pc_idx);
    }
    bool is_stream = update_stream_sentinel(set, pc_idx, line, type);

    // Hit path
    if (hit) {
        // Count only demand hits for gating; never promote on first hit
        if (is_demand(type)) {
            if (mhits[set][way] < 3) mhits[set][way]++;
            if (!stream_flag[set][way] && mhits[set][way] >= 2) {
                lru_make_mru(set, way);
            }
        }
        // Prefetch hits do not promote (implicit by gating)
        return;
    }

    // Miss/fill path:
    // Sampled-set training for doorkeeper (record victim PC on dead-on-evict)
    if (is_sampled(set)) {
        uint32_t ss = sample_set_idx(set);
        // If evicting a valid line that died with zero demand hits and not stream-clamped,
        // mark its PC as cold via doorkeeper (if we captured it).
        if (sample_valid[ss][way]) {
            if (mhits[set][way] == 0 && stream_flag[set][way] == 0) {
                // We don't know the evicted PC exactly in general, but for sampled sets we stored it:
                uint32_t ev_pc_idx = sample_pcidx[ss][way];
                bloom_set(static_cast<uint64_t>(ev_pc_idx));
            }
            sample_valid[ss][way] = 0;
        }
        // Capture the incoming PC for this line
        sample_pcidx[ss][way] = static_cast<uint16_t>(pc_idx & 0x7FF);
        sample_valid[ss][way] = 1;
    }

    // Decide insertion characteristics
    uint8_t now = epoch6_now();

    // Doorkeeper cold PC?
    bool pc_cold = false;
    if (type != WRITEBACK) {
        // Using hashed PC index as key in the Bloom
        pc_cold = bloom_query(static_cast<uint64_t>(pc_idx));
    }

    // Predicted gap (min 1, max 31 to stay within forward window)
    uint8_t pred_gap = PC_gap6[pc_idx] & 63;
    if (pred_gap == 0) pred_gap = 1;
    if (pred_gap > 31) pred_gap = 31;

    // Streaming lines: soft-bypass via tail insertion, no promotion, immediate expiry
    if (is_stream && type != WRITEBACK) {
        stream_flag[set][way] = 1;
        mhits[set][way] = 0;
        due6[set][way] = now;                 // expire immediately (overdue soon)
        lru_insert_at(set, way, LLC_WAYS - 1);
        return;
    }

    // Doorkeeper-cold PCs: tail insert, immediate expiry
    if (pc_cold && type != WRITEBACK) {
        stream_flag[set][way] = 0;
        mhits[set][way] = 0;
        due6[set][way] = now;                 // expire immediately
        lru_insert_at(set, way, LLC_WAYS - 1);
        return;
    }

    // Normal insertion: deadline = now + predicted gap
    stream_flag[set][way] = 0;
    mhits[set][way] = 0;
    due6[set][way] = static_cast<uint8_t>((now + pred_gap) & 63);

    // Adaptive insertion depth: moderate tail to reduce pollution
    // - writebacks: slightly older insertion (still tail)
    // - prefetch: tail
    // - demands: near-tail
    if (type == WRITEBACK || type == PREFETCH) {
        lru_insert_at(set, way, LLC_WAYS - 1);
    } else {
        lru_insert_at(set, way, LLC_WAYS - 4); // demand near-tail
    }
}

// ---------------- Stats ----------------
void PrintStats() { }
void PrintStats_Heartbeat() { }