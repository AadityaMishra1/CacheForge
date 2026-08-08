#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// CRC2 access type tags
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
// StreamSkip (per-PC stride detector)
static constexpr uint32_t ST_IDX_BITS = 10;   // 1024-entry per-PC detector
static constexpr uint32_t ST_SIZE     = (1u << ST_IDX_BITS);
static constexpr uint8_t  ST_LINE_LOW = 12;   // track low 12 bits of line number
static constexpr uint8_t  ST_CONF_BYPASS = 2; // >=2 => stream-guard insertion

// Dead-PC counting Bloom (two 2-bit counter tables)
static constexpr uint32_t BLOOM_IDX_BITS = 10;   // 1024 counters per hash
static constexpr uint32_t BLOOM_SIZE     = (1u << BLOOM_IDX_BITS);
static constexpr uint8_t  BLOOM_DEAD_TH  = 3;    // sum(c1+c2) >= 3 => predicted-dead

// Insertion and aging
static constexpr uint8_t AGE_MAX  = 3;  // 0=MRU .. 3=LRU-ish
static constexpr uint8_t AGE_WARM = 1;  // warm insertion depth
static constexpr uint32_t EPOCH_PERIOD = 1024; // fills between epoch toggles (lazy aging)

// ---------------- Per-line state (packed logically) ----------------
// AGE: 2 bits (0..3)
static uint8_t AGE[LLC_SETS][LLC_WAYS];
// FREQ: 2-bit demand hit count (0..3), multi-hit gating and victim signal
static uint8_t FREQ[LLC_SETS][LLC_WAYS];
// EPOCH_TAG: 1-bit epoch tag for lazy frequency aging
static uint8_t EPOCH_TAG[LLC_SETS][LLC_WAYS];
// STATUS: 2 bits: 0=normal, 1=stream-guard (no promote/train), 2=sentinel (prefetch/WB)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];
// PC_SIG8: 8-bit per-line PC signature captured at fill (for Bloom training on evict/2+ hits)
static uint8_t PC_SIG8[LLC_SETS][LLC_WAYS];

// ---------------- Global predictors ----------------
// StreamSkip per-PC detector
static uint16_t ST_LAST_LINE[ST_SIZE]; // low 12 bits of line number
static uint8_t  ST_CONF[ST_SIZE];      // 2-bit (0..3)
static uint8_t  ST_LAST_ABS1[ST_SIZE]; // 0/1: last step had |delta|==1

// Dead-PC counting Bloom (two hash tables of 2-bit counters)
static uint8_t BLOOM1[BLOOM_SIZE]; // 0..3
static uint8_t BLOOM2[BLOOM_SIZE]; // 0..3

// Global epoch for lazy aging
static uint8_t  GLOBAL_EPOCH = 0;      // 1-bit logical epoch
static uint32_t fills_since_epoch = 0; // to toggle epoch periodically

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline uint64_t line_number(uint64_t paddr) { return paddr >> 6; }

// PC hash to 8-bit signature (kept small to save per-line bits)
static inline uint8_t pc_sig8(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 17) ^ (pc >> 27);
    x ^= (x >> 16);
    return static_cast<uint8_t>(x & 0xFFu);
}

// Two Bloom indices derived from an 8-bit signature (deterministic mixing)
static inline void bloom_indices(uint8_t sig8, uint16_t &i1, uint16_t &i2) {
    uint16_t x = static_cast<uint16_t>(sig8);
    uint16_t m1 = static_cast<uint16_t>((x ^ (x << 3) ^ (x << 7) ^ 0xA5u) & (BLOOM_SIZE - 1));
    uint16_t m2 = static_cast<uint16_t>(((x * 131u) ^ (x << 5) ^ 0x5Bu) & (BLOOM_SIZE - 1));
    i1 = m1; i2 = m2;
}
static inline uint8_t bloom_query(uint8_t sig8) {
    uint16_t i1, i2; bloom_indices(sig8, i1, i2);
    return static_cast<uint8_t>(BLOOM1[i1] + BLOOM2[i2]);
}
static inline void bloom_inc_train(uint8_t sig8) {
    uint16_t i1, i2; bloom_indices(sig8, i1, i2);
    sat_inc(BLOOM1[i1], 3);
    sat_inc(BLOOM2[i2], 3);
}
static inline void bloom_dec_train(uint8_t sig8) {
    uint16_t i1, i2; bloom_indices(sig8, i1, i2);
    sat_dec(BLOOM1[i1]);
    sat_dec(BLOOM2[i2]);
}

// StreamSkip: update stream detector on demand access and return confidence
static inline uint32_t st_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12) ^ (pc >> 18);
    return static_cast<uint32_t>(x) & (ST_SIZE - 1);
}
static inline uint8_t line_low12(uint64_t paddr) {
    return static_cast<uint8_t>((paddr >> 6) & ((1u << ST_LINE_LOW) - 1));
}
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = st_index(pc);
    uint16_t prev = ST_LAST_LINE[idx];
    uint16_t curr = line_low12(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(prev));
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) sat_inc(ST_CONF[idx], 3);
        else { if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; ST_LAST_ABS1[idx] = 1; }
    } else {
        sat_dec(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Effective frequency with lazy aging (decay by 1 if epoch tag is stale)
static inline uint8_t effective_freq(uint8_t f, uint8_t tag) {
    if (tag != GLOBAL_EPOCH) return (f > 0) ? static_cast<uint8_t>(f - 1) : 0;
    return f;
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]       = AGE_MAX;
            FREQ[s][w]      = 0;
            EPOCH_TAG[s][w] = GLOBAL_EPOCH;
            STATUS[s][w]    = 0;
            PC_SIG8[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < ST_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    for (uint32_t i = 0; i < BLOOM_SIZE; i++) {
        BLOOM1[i] = 0;
        BLOOM2[i] = 0;
    }
    GLOBAL_EPOCH = 0;
    fills_since_epoch = 0;
}

// ---------------- Victim selection ----------------
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // 1) Prefer invalid way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // 2) Choose by (status_prefer, effective_freq, age desc)
    uint32_t best = 0;
    auto status_rank = [&](uint8_t st)->uint8_t {
        return (st == 2) ? 0 : (st == 1 ? 1 : 2); // evict prefetch/WB first, then stream, then normal
    };
    uint8_t best_sr   = status_rank(STATUS[set][0]);
    uint8_t best_eff  = effective_freq(FREQ[set][0], EPOCH_TAG[set][0]);
    uint8_t best_age  = AGE[set][0];

    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        uint8_t sr  = status_rank(STATUS[set][w]);
        uint8_t eff = effective_freq(FREQ[set][w], EPOCH_TAG[set][w]);
        uint8_t age = AGE[set][w];

        bool better = false;
        if (sr < best_sr) better = true;
        else if (sr == best_sr) {
            if (eff < best_eff) better = true;
            else if (eff == best_eff && age > best_age) better = true;
        }
        if (better) {
            best_sr  = sr;
            best_eff = eff;
            best_age = age;
            best     = w;
        }
    }

    // 3) Dead-on-evict training for victim (only if valid and normal status and 0 demand hits)
    if (current_set[best].valid) {
        if (STATUS[set][best] == 0 && FREQ[set][best] == 0) {
            bloom_inc_train(PC_SIG8[set][best]);
        }
    }
    return best;
}

// ---------------- State update ----------------
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);

    // Update stream detector on all demand accesses (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        uint8_t st = STATUS[set][way];

        // Never promote or train stream-guarded or sentinel lines
        if (st == 1 || st == 2) {
            return;
        }

        // Lazy frequency aging on touch
        if (EPOCH_TAG[set][way] != GLOBAL_EPOCH) {
            if (FREQ[set][way] > 0) FREQ[set][way]--;
            EPOCH_TAG[set][way] = GLOBAL_EPOCH;
        }

        // Demand hit: multi-hit gating and survivor training
        if (demand) {
            uint8_t prev = FREQ[set][way];
            if (prev < 3) FREQ[set][way] = static_cast<uint8_t>(prev + 1);

            // On second demand hit: promote to MRU and train Bloom negative (survivor)
            if (prev == 1) {
                AGE[set][way] = 0; // MRU
                bloom_dec_train(PC_SIG8[set][way]);
            }
            // First demand hit (prev==0): do not promote
        }
        return;
    }

    // MISS/FILL path: initialize line state based on predictors
    // Toggle epoch periodically (on fills)
    fills_since_epoch++;
    if (fills_since_epoch >= EPOCH_PERIOD) {
        GLOBAL_EPOCH ^= 1u;
        fills_since_epoch = 0;
    }

    // Common init
    AGE[set][way]       = AGE_MAX;
    FREQ[set][way]      = 0;
    EPOCH_TAG[set][way] = GLOBAL_EPOCH;
    STATUS[set][way]    = 0; // default normal
    PC_SIG8[set][way]   = pc_sig8(PC);

    // Writebacks: low-priority sentinel, no training
    if (type == ACCESS_WRITEBACK) {
        STATUS[set][way] = 2; // sentinel
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Prefetches: always insert at tail as sentinel, no training
    if (type == ACCESS_PREFETCH) {
        STATUS[set][way] = 2; // sentinel
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Demand fills: StreamSkip then Dead-PC Bloom decide insertion depth
    if (demand) {
        // Stream-guard on strong ±1-line confidence
        if (stream_conf >= ST_CONF_BYPASS) {
            STATUS[set][way] = 1;   // stream-guard (no promote/train)
            AGE[set][way]    = AGE_MAX; // near-bypass
            return;
        }

        // Dead-PC Bloom prediction
        uint8_t dead_score = bloom_query(PC_SIG8[set][way]);
        if (dead_score >= BLOOM_DEAD_TH) {
            // Predicted cold: old insertion
            AGE[set][way] = AGE_MAX;
        } else {
            // Predicted warm: shallow insertion
            AGE[set][way] = AGE_WARM;
        }
        // STATUS remains normal (eligible for multi-hit promotion/training)
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // Intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // Intentionally blank
}