#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS            = 6;     // 64-entry PC signatures (per-line PC sig)
static constexpr uint32_t STREAM_IDX_BITS    = 9;     // 512-entry stream table
static constexpr uint8_t  LINE_LOW_BITS      = 10;    // track low 10 bits of line number for stride
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;     // >=2 => near-bypass for streams
static constexpr uint32_t BLOOM_BITS         = 4096;  // 2-hash Bloom filter (total bits per hash)
static constexpr uint32_t EPOCH_PERIOD       = 2048;  // fills between epoch toggles (lazy heat aging)

// RRPV insertion depths
static constexpr uint8_t  RRPV_DISTANT       = 3;
static constexpr uint8_t  RRPV_COLD_INSERT   = 3;     // stream/prefetch/WB/bloom-cold insertion
static constexpr uint8_t  RRPV_WARM_INSERT   = 2;     // default demand insertion

// --------------- Per-line state ---------------
// 2-bit RRPV: 0 (MRU) .. 3 (LRU)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];
// 2-bit HEAT (demand hits only), 0..3 with lazy decay
static uint8_t HEAT[LLC_SETS][LLC_WAYS];
// 1-bit epoch tag for lazy aging
static uint8_t EPOCH_TAG[LLC_SETS][LLC_WAYS];
// STATUS: 0=normal, 1=stream-guard, 2=sentinel (prefetch/WB), 3=cold-guard (Bloom)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];
// Per-line PC signature (PC_BITS)
static uint8_t PC_SIG[LLC_SETS][LLC_WAYS];

// --------------- Stream Doorkeeper (PC-indexed) ---------------
static constexpr uint32_t STREAM_SIZE = (1u << STREAM_IDX_BITS);
static uint16_t ST_LAST_LINE[STREAM_SIZE]; // low LINE_LOW_BITS of line number
static uint8_t  ST_CONF[STREAM_SIZE];      // 2-bit conf (0..3)
static uint8_t  ST_LAST_ABS1[STREAM_SIZE]; // 0/1: last step had |delta|==1

// --------------- DOA Bloom (2-hash) ---------------
static constexpr uint32_t BLOOM_QWORDS = (BLOOM_BITS + 63u) / 64u;
static uint64_t BLOOM0[BLOOM_QWORDS];
static uint64_t BLOOM1[BLOOM_QWORDS];

// --------------- Global epoch and counters ---------------
static uint8_t  GLOBAL_EPOCH = 0;   // 1-bit logical epoch
static uint32_t fill_counter = 0;   // counts fills to toggle epoch

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // simple xor-folding hash -> PC_BITS
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23) ^ (pc >> 37);
    return static_cast<uint32_t>(x) & ((1u << PC_BITS) - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 21);
    return static_cast<uint32_t>(x) & (STREAM_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u1(uint8_t &x) { if (x > 0) x--; }

static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_lowN(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(prev));
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) sat_inc_u2(ST_CONF[idx]); // consecutive ±1
        else { if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; ST_LAST_ABS1[idx] = 1; }
    } else {
        sat_dec_u1(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Bloom hashing on PC⊕region signature
static inline uint32_t region_sig(uint64_t paddr) {
    // region: 4KB page index low 12 bits (simple, small but effective)
    return static_cast<uint32_t>((paddr >> 12) & 0xFFFu);
}
static inline uint32_t bloom_h1(uint32_t key) {
    uint32_t x = key * 0x9E3779B1u; // golden ratio 32-bit
    x ^= (x >> 15);
    return x & (BLOOM_BITS - 1);
}
static inline uint32_t bloom_h2(uint32_t key) {
    uint32_t x = (key ^ 0x7f4a7c15u) * 0x85ebca6bu;
    x ^= (x >> 13);
    return x & (BLOOM_BITS - 1);
}
static inline void bloom_set(uint32_t key) {
    uint32_t i0 = bloom_h1(key);
    uint32_t i1 = bloom_h2(key);
    BLOOM0[i0 >> 6] |= (1ull << (i0 & 63));
    BLOOM1[i1 >> 6] |= (1ull << (i1 & 63));
}
static inline bool bloom_test(uint32_t key) {
    uint32_t i0 = bloom_h1(key);
    uint32_t i1 = bloom_h2(key);
    bool b0 = (BLOOM0[i0 >> 6] >> (i0 & 63)) & 1ull;
    bool b1 = (BLOOM1[i1 >> 6] >> (i1 & 63)) & 1ull;
    return b0 && b1;
}

static inline uint8_t effective_heat(uint32_t set, uint32_t way) {
    uint8_t h = HEAT[set][way];
    if (EPOCH_TAG[set][way] != GLOBAL_EPOCH) {
        return (h > 0) ? static_cast<uint8_t>(h - 1) : 0;
    }
    return h;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]      = RRPV_DISTANT;
            HEAT[s][w]      = 0;
            EPOCH_TAG[s][w] = GLOBAL_EPOCH;
            STATUS[s][w]    = 0; // normal
            PC_SIG[s][w]    = 0;
        }
    }
    for (uint32_t i = 0; i < STREAM_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    for (uint32_t i = 0; i < BLOOM_QWORDS; i++) {
        BLOOM0[i] = 0;
        BLOOM1[i] = 0;
    }
    GLOBAL_EPOCH = 0;
    fill_counter = 0;
}

// Prefer evicting sentinels, then lowest effective heat, tie-broken by largest RRPV
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool sa = (STATUS[set][a] != 0);
    bool sb = (STATUS[set][b] != 0);
    if (sa != sb) return sa; // prefer evicting sentinel-guarded lines

    uint8_t ha = effective_heat(set, a);
    uint8_t hb = effective_heat(set, b);
    if (ha != hb) return ha < hb;

    // tie-breaker by RRPV (larger is more evictable)
    return RRPV[set][a] > RRPV[set][b];
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Choose most evictable way by composite priority
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);

    // Demand access updates stream detector (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        uint8_t st = STATUS[set][way];

        // Never promote or train sentinels (prefetch/WB) or stream/cold-guarded lines
        if (st != 0) return;

        // Lazy aging touch
        if (EPOCH_TAG[set][way] != GLOBAL_EPOCH) {
            if (HEAT[set][way] > 0) HEAT[set][way]--;
            EPOCH_TAG[set][way] = GLOBAL_EPOCH;
        }

        if (demand) {
            uint8_t prev = HEAT[set][way];
            if (prev < 3) HEAT[set][way] = static_cast<uint8_t>(prev + 1);
            // Multi-hit promotion gating: promote only on 2nd+ demand hit
            if (prev >= 1) {
                RRPV[set][way] = 0; // MRU
            }
        }
        return;
    }

    // MISS (fill) path: train on evicted line BEFORE overwriting metadata
    {
        uint8_t old_st   = STATUS[set][way];
        uint8_t old_heat = HEAT[set][way];
        uint8_t old_sig  = PC_SIG[set][way];

        // Train DOA Bloom: only if previous line was demand-inserted (normal) and dead (0 demand hits)
        if (old_st == 0 && old_heat == 0) {
            uint32_t key = (static_cast<uint32_t>(old_sig) << LINE_LOW_BITS) ^ region_sig(victim_addr);
            bloom_set(key);
        }
    }

    // Toggle epoch periodically to age heat (and softly reset Bloom)
    fill_counter++;
    if (fill_counter >= EPOCH_PERIOD) {
        fill_counter = 0;
        GLOBAL_EPOCH ^= 1u;
        // Clear Bloom to bound false positives and adapt phases
        for (uint32_t i = 0; i < BLOOM_QWORDS; i++) {
            BLOOM0[i] = 0;
            BLOOM1[i] = 0;
        }
    }

    // Insert new line metadata
    uint8_t new_status = 0;
    uint8_t new_rrpv   = RRPV_WARM_INSERT; // default for normal demand
    uint8_t new_heat   = 0;
    uint8_t new_pc_sig = static_cast<uint8_t>(pc_index(PC));

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert tail, sentinel, no promotion/training
        new_status = 2; // sentinel
        new_rrpv   = RRPV_COLD_INSERT;
    } else if (type == ACCESS_PREFETCH) {
        // Prefetch inserts at tail with sentinel
        new_status = 2; // sentinel
        new_rrpv   = RRPV_COLD_INSERT;
    } else { // demand (LOAD/RFO)
        // DOA Bloom prediction on current PC⊕region
        uint32_t key = (static_cast<uint32_t>(new_pc_sig) << LINE_LOW_BITS) ^ region_sig(paddr);
        bool bloom_cold = bloom_test(key);
        bool stream_guard = (stream_conf >= STREAM_CONF_BYPASS);

        if (stream_guard) {
            new_status = 1; // stream-guard: near-bypass
            new_rrpv   = RRPV_COLD_INSERT;
        } else if (bloom_cold) {
            new_status = 3; // cold-guard: near-bypass
            new_rrpv   = RRPV_COLD_INSERT;
        } else {
            new_status = 0; // normal
            new_rrpv   = RRPV_WARM_INSERT;
        }
    }

    // Commit insertion state
    STATUS[set][way]    = new_status;
    RRPV[set][way]      = new_rrpv;
    HEAT[set][way]      = new_heat;
    EPOCH_TAG[set][way] = GLOBAL_EPOCH;
    PC_SIG[set][way]    = new_pc_sig;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}