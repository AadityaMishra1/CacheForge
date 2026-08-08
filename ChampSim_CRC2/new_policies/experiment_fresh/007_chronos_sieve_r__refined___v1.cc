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
#define SCAN_DEBT_MAX 3           // per-set stream pressure cap
#define EMA_SHIFT 2               // EMA alpha = 1/4
#define EPOCH_BITS 6              // 6-bit modular epoch (0..63)
#define SAMPLE_RATE 64            // 1-of-64 leader sets for SHiP training

// ---------------- Per-line metadata (13 bits/line) ----------------
// - lru   : 4 bits (0=MRU .. 15=LRU)
// - mhits : 2 bits (0..3 demand hits; promotion only when >=2)
// - stream: 1 bit (stream-clamped; never promote)
// - due6  : 6 bits (deadline epoch modulo 64)
static uint8_t lru_pos[LLC_SETS][LLC_WAYS];
static uint8_t mhits[LLC_SETS][LLC_WAYS];
static uint8_t stream_flag[LLC_SETS][LLC_WAYS];
static uint8_t due6[LLC_SETS][LLC_WAYS];

// ---------------- Per-set stream pressure ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3 (we count 2 bits in storage)

// ---------------- PC-indexed predictors ----------------
// - gap6     : 6-bit EMA of inter-access gap
// - last_ep6 : 6-bit last access epoch
// - last_ln  : 12-bit last line number (low bits) for stride detection
// - run2     : 2-bit run length of +1 strides (0..3)
// - shct2    : 2-bit SHiP-style dead-block counter per PC (0=cold .. 3=hot)
static uint8_t  PC_gap6[PC_IDX_SIZE];        // 0..63
static uint8_t  PC_last_ep6[PC_IDX_SIZE];    // 0..63
static uint16_t PC_last_ln12[PC_IDX_SIZE];   // 0..4095
static uint8_t  PC_run2[PC_IDX_SIZE];        // 0..3
static uint8_t  SHCT2[PC_IDX_SIZE];          // 2-bit counter

// ---------------- Sampled-set SHiP training ----------------
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
static inline uint8_t sat_inc_u2(uint8_t v) { return (v < 3) ? (uint8_t)(v + 1) : 3; }
static inline uint8_t sat_dec_u2(uint8_t v) { return (v > 0) ? (uint8_t)(v - 1) : 0; }

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

// Stream sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (!is_demandish(type)) return false;
    uint16_t cur = static_cast<uint16_t>(line & 0xFFF); // 12-bit low line#
    int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(PC_last_ln12[pc_idx]);
    if (stride == 1) {
        if (PC_run2[pc_idx] < 3) PC_run2[pc_idx]++;
        if (set_scan_debt[set] < SCAN_DEBT_MAX) set_scan_debt[set]++; // accumulate scan pressure
    } else {
        if (PC_run2[pc_idx] > 0) PC_run2[pc_idx]--;
        // bleed pressure slowly
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
    PC_last_ln12[pc_idx] = cur;

    // streaming when PC has >=2 run and set has any pressure
    return (PC_run2[pc_idx] >= STREAM_RUN_THR) && (set_scan_debt[set] > 0);
}

// EMA update and deadline proposal
static inline uint8_t update_pc_gap_and_get_ttl(uint32_t pc_idx, uint8_t now) {
    uint8_t last = PC_last_ep6[pc_idx];
    uint8_t gap = static_cast<uint8_t>((now - last) & ((1u << EPOCH_BITS) - 1));
    // EMA: gap_ema += (gap - gap_ema)/4
    uint8_t ema = PC_gap6[pc_idx];
    int16_t diff = static_cast<int16_t>(gap) - static_cast<int16_t>(ema);
    ema = static_cast<uint8_t>(ema + (diff >> EMA_SHIFT));
    PC_gap6[pc_idx] = ema;
    PC_last_ep6[pc_idx] = now;
    // Clip TTL: at least 2 when reused; elsewhere caller may override
    uint8_t ttl = ema;
    if (ttl < 2) ttl = 2;
    return ttl;
}

void InitReplacementState() {
    std::memset(lru_pos, 0, sizeof(lru_pos));
    std::memset(mhits, 0, sizeof(mhits));
    std::memset(stream_flag, 0, sizeof(stream_flag));
    std::memset(due6, 0, sizeof(due6));
    std::memset(set_scan_debt, 0, sizeof(set_scan_debt));
    std::memset(PC_gap6, 0, sizeof(PC_gap6));
    std::memset(PC_last_ep6, 0, sizeof(PC_last_ep6));
    std::memset(PC_last_ln12, 0, sizeof(PC_last_ln12));
    std::memset(PC_run2, 0, sizeof(PC_run2));
    std::memset(SHCT2, 0, sizeof(SHCT2));
    std::memset(sample_pcidx, 0, sizeof(sample_pcidx));
    std::memset(sample_valid, 0, sizeof(sample_valid));
    // Initialize LRU stack positions
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            lru_pos[s][w] = w;
            mhits[s][w] = 0;
            stream_flag[s][w] = 0;
            due6[s][w] = 0;
        }
    }
    // Neutral SHCT start (1) to avoid over-aggressive bypass
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT2[i] = 1;
    }
    global_events = 0;
}

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

    uint8_t now = epoch6_now();

    // First prefer expired lines (especially streaming), else choose farthest-deadline
    int best = -1;
    bool looking_expired = true;

    // Try expired group first
    for (int pass = 0; pass < 2; pass++) {
        looking_expired = (pass == 0);
        best = -1;
        uint8_t best_stream = 0;
        uint8_t best_mhits_lt2 = 0;
        uint8_t best_lru = 0;
        uint8_t best_dist = 0;

        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            uint8_t due = due6[set][w];
            uint8_t dist = static_cast<uint8_t>((due - now) & ((1u << EPOCH_BITS) - 1));
            bool expired = (dist >= (1u << (EPOCH_BITS - 1))); // >=32 => in the past
            if (looking_expired != expired) continue;

            uint8_t sflag = stream_flag[set][w] ? 1 : 0;
            uint8_t mhlt2 = (mhits[set][w] < 2) ? 1 : 0;
            uint8_t lruv = lru_pos[set][w];

            // Priority: stream > mhits<2 > (for expired: more overdue dist; for non-expired: farther dist) > LRU
            bool better = false;
            if (best == -1) better = true;
            else if (sflag != best_stream) better = (sflag > best_stream);
            else if (mhlt2 != best_mhits_lt2) better = (mhlt2 > best_mhits_lt2);
            else if (dist != best_dist) better = (dist > best_dist);
            else if (lruv != best_lru) better = (lruv > best_lru);

            if (better) {
                best = w;
                best_stream = sflag;
                best_mhits_lt2 = mhlt2;
                best_lru = lruv;
                best_dist = dist;
            }
        }
        if (best != -1) return static_cast<uint32_t>(best);
    }

    // Fallback (should not happen)
    return 0;
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

    // Advance epoch on any access to keep deadlines moving
    global_events++;

    uint8_t now = epoch6_now();
    uint64_t line = line_number(paddr);
    uint32_t pidx = pc_index(PC);

    // Update per-PC EMA and stride/run; compute TTL proposal
    uint8_t ttl = update_pc_gap_and_get_ttl(pidx, now);
    bool stream_pred = update_stream_sentinel(set, pidx, line, type);

    // SHiP prediction (dead/cold PCs): 0..1 => cold, 2..3 => hot
    bool shct_cold = (SHCT2[pidx] <= 1);

    if (hit) {
        // On hit: demand hits increment multi-hit; prefetch/writeback do not promote
        if (is_demand(type)) {
            if (mhits[set][way] < 3) mhits[set][way]++;
        }
        // Stream-clamped lines never promote; otherwise promote only on >=2 demand hits
        if (!stream_flag[set][way] && is_demand(type) && (mhits[set][way] >= 2)) {
            lru_make_mru(set, way);
        }
        // Refresh deadline for reused lines
        if (is_demand(type)) {
            uint8_t new_due = static_cast<uint8_t>((now + ttl) & ((1u << EPOCH_BITS) - 1));
            due6[set][way] = new_due;
        }
        return;
    }

    // Miss path: train SHiP on eviction for sampled sets
    if (is_sampled(set)) {
        uint32_t sid = sample_set_idx(set);
        if (sample_valid[sid][way]) {
            uint16_t ev_pcidx = sample_pcidx[sid][way];
            // Consider a line reused if it saw any demand hit before eviction
            bool reused = (mhits[set][way] > 0);
            if (reused) SHCT2[ev_pcidx] = sat_inc_u2(SHCT2[ev_pcidx]);
            else        SHCT2[ev_pcidx] = sat_dec_u2(SHCT2[ev_pcidx]);
        }
        // Record new PC index for this line
        sample_pcidx[sid][way] = static_cast<uint16_t>(pidx & (PC_IDX_SIZE - 1));
        sample_valid[sid][way] = 1;
    }

    // Decide insertion policy
    bool is_stream_fill = stream_pred && (type != WRITEBACK); // never bypass WB
    bool is_cold_fill = (type != WRITEBACK) && shct_cold && !is_stream_fill;

    uint8_t ins_pos = LLC_WAYS - 1; // default tail
    uint8_t new_due = now;          // default immediate expiry

    if (type == PREFETCH) {
        // Always lowest priority, no promotion
        ins_pos = LLC_WAYS - 1;
        new_due = static_cast<uint8_t>((now + 1) & ((1u << EPOCH_BITS) - 1));
        stream_flag[set][way] = 1; // clamp prefetches to avoid accidental promotion
    } else if (is_stream_fill) {
        // Stream soft-bypass: tail insert, never promote, immediate expiry
        ins_pos = LLC_WAYS - 1;
        new_due = static_cast<uint8_t>((now - 1) & ((1u << EPOCH_BITS) - 1)); // expired
        stream_flag[set][way] = 1;
    } else if (is_cold_fill) {
        // SHiP-predicted dead: tail insert, immediate expiry
        ins_pos = LLC_WAYS - 1;
        new_due = static_cast<uint8_t>((now - 1) & ((1u << EPOCH_BITS) - 1)); // expired
        stream_flag[set][way] = 0;
    } else {
        // Warm/Hot demand/WB: adaptive insertion, protect with EMA deadline
        uint8_t strength = SHCT2[pidx];
        if (type == WRITEBACK) {
            // Never bypass WB; insert moderately
            ins_pos = (LLC_WAYS > 8) ? 8 : (LLC_WAYS - 2);
        } else {
            // Demand: stronger PCs get nearer MRU
            ins_pos = (strength >= 3) ? 4 : 8;
        }
        new_due = static_cast<uint8_t>((now + ttl) & ((1u << EPOCH_BITS) - 1));
        stream_flag[set][way] = 0;
    }

    // Reset per-line state and insert
    mhits[set][way] = 0;
    due6[set][way] = new_due;
    lru_insert_at(set, way, ins_pos);
}

void PrintStats() {}
void PrintStats_Heartbeat() {}