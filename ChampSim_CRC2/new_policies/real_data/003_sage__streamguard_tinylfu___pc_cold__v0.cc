#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// Tunables
static constexpr uint32_t EPOCH_PERIOD = 1024;  // fills between epoch toggles
static constexpr uint8_t  PC_BITS      = 6;     // 64-entry PC tables
static constexpr uint8_t  PC_SIZE      = (1u << PC_BITS);
static constexpr uint8_t  STREAM_CONF_BYPASS = 2; // >=2 => stream-guard
static constexpr uint8_t  PC_COLD_TH   = 1;     // <=1 => predicted cold
static constexpr uint8_t  RRPV_DISTANT = 3;
static constexpr uint8_t  RRPV_WARM    = 1;

// Per-line state (packed logically; stored in bytes)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];      // 2 bits used (0..3)
static uint8_t FREQ[LLC_SETS][LLC_WAYS];      // 2-bit TinyLFU count (0..3), demand hits only
static uint8_t EPOCH_TAG[LLC_SETS][LLC_WAYS]; // 1-bit epoch tag
// STATUS: 0=normal, 1=stream-guard (no promote/train), 2=sentinel (prefetch/WB)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];    // 2 bits used (0..2)
// Per-line PC signature for training PC-Cold on dead-on-evict
static uint8_t PC_SIG[LLC_SETS][LLC_WAYS];    // 6-bit index (0..63)

// Global PC-based cold predictor (2-bit saturating counters)
static uint8_t PC_CT[PC_SIZE];                // 0..3

// StreamGuard: per-PC stride detector (low8 line number + conf + last_abs1)
static uint8_t ST_LAST_LINE[PC_SIZE];         // low 8 bits of line number
static uint8_t ST_CONF[PC_SIZE];              // 2-bit confidence (0..3)
static uint8_t ST_LAST_ABS1[PC_SIZE];         // 0/1: last step had |delta|==1

// Global epoch and counter
static uint8_t  GLOBAL_EPOCH = 0;             // 1-bit logical
static uint32_t epoch_counter = 0;

// Helpers
static inline void sat_inc2(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec1(uint8_t &x) { if (x > 0) x--; }

static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12) ^ (pc >> 18);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint8_t line_low8(uint64_t paddr) {
    return static_cast<uint8_t>((paddr >> 6) & 0xFFu);
}
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint8_t curr = line_low8(paddr);
    int8_t delta = static_cast<int8_t>(curr - ST_LAST_LINE[idx]);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) sat_inc2(ST_CONF[idx], 3);
        else { if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; ST_LAST_ABS1[idx] = 1; }
    } else {
        sat_dec1(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}
static inline uint8_t effective_freq(uint8_t f, uint8_t tag) {
    // Lazy aging: if epoch tag is stale, decay by 1
    if (tag != GLOBAL_EPOCH) return (f > 0) ? static_cast<uint8_t>(f - 1) : 0;
    return f;
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]      = RRPV_DISTANT;
            FREQ[s][w]      = 0;
            EPOCH_TAG[s][w] = GLOBAL_EPOCH;
            STATUS[s][w]    = 0; // normal
            PC_SIG[s][w]    = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_CT[i]        = 1; // slightly cold start
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    GLOBAL_EPOCH = 0;
    epoch_counter = 0;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Choose lowest effective frequency; tie-break by largest RRPV
    uint32_t best_way = 0;
    uint8_t best_eff = 255;
    uint8_t best_rrpv = 0;

    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint8_t eff = effective_freq(FREQ[set][w], EPOCH_TAG[set][w]);
        uint8_t rrpv = RRPV[set][w];
        if (eff < best_eff || (eff == best_eff && rrpv > best_rrpv)) {
            best_eff = eff;
            best_rrpv = rrpv;
            best_way = w;
        }
    }
    return best_way;
}

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
    uint8_t stream_conf = 0;
    if (demand) {
        // Update stream detector on all demand accesses
        stream_conf = update_stream_conf(PC, paddr);
    }

    // HIT path
    if (hit) {
        uint8_t st = STATUS[set][way];

        // Never promote or train sentinels (prefetch/WB) or stream-guarded lines
        if (st == 2 || st == 1) {
            return;
        }

        // Lazy aging on touch
        if (EPOCH_TAG[set][way] != GLOBAL_EPOCH) {
            if (FREQ[set][way] > 0) FREQ[set][way]--;
            EPOCH_TAG[set][way] = GLOBAL_EPOCH;
        }

        // Multi-hit promotion gating: only 2nd+ demand hit promotes
        // Count demand hits into FREQ (saturating 0..3)
        if (demand) {
            uint8_t prev = FREQ[set][way];
            if (prev < 3) FREQ[set][way] = static_cast<uint8_t>(prev + 1);
            // On first hit: train PC-Cold toward warm, but do not promote
            if (prev == 0) {
                uint32_t idx = pc_index(PC);
                sat_inc2(PC_CT[idx], 3);
            } else {
                // Second or later hit: strong promotion to MRU
                RRPV[set][way] = 0;
            }
        }
        return;
    }

    // MISS path: train on evicted occupant (dead-on-evict), excluding sentinels/streams
    if (victim_addr != 0) {
        if (STATUS[set][way] == 0) {
            // Lazy compute if stale epoch -> effective zero recognized as dead
            uint8_t eff = effective_freq(FREQ[set][way], EPOCH_TAG[set][way]);
            if (eff == 0) {
                uint32_t ps = PC_SIG[set][way] & (PC_SIZE - 1);
                sat_dec1(PC_CT[ps]);
            }
        }
    }

    // Decide insertion for the incoming line
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert at tail as sentinel
        RRPV[set][way]      = RRPV_DISTANT;
        FREQ[set][way]      = 0;
        EPOCH_TAG[set][way] = GLOBAL_EPOCH;
        STATUS[set][way]    = 2; // sentinel
        PC_SIG[set][way]    = 0;
    } else if (type == ACCESS_PREFETCH) {
        // Prefetches are low priority, no promotion, no training
        RRPV[set][way]      = RRPV_DISTANT;
        FREQ[set][way]      = 0;
        EPOCH_TAG[set][way] = GLOBAL_EPOCH;
        STATUS[set][way]    = 2; // sentinel
        PC_SIG[set][way]    = 0;
    } else {
        // Demand insertion: StreamGuard first, then PC-Cold
        uint32_t idx = pc_index(PC);
        bool is_stream = (stream_conf >= STREAM_CONF_BYPASS);

        if (is_stream) {
            // Approximate bypass: insert cold and forbid promotion/training
            RRPV[set][way]      = RRPV_DISTANT;
            FREQ[set][way]      = 0;
            EPOCH_TAG[set][way] = GLOBAL_EPOCH;
            STATUS[set][way]    = 1; // stream-guard
            PC_SIG[set][way]    = static_cast<uint8_t>(idx);
        } else {
            bool pc_cold = (PC_CT[idx] <= PC_COLD_TH);
            RRPV[set][way]      = pc_cold ? RRPV_DISTANT : RRPV_WARM;
            FREQ[set][way]      = 0;                   // hits-only frequency
            EPOCH_TAG[set][way] = GLOBAL_EPOCH;
            STATUS[set][way]    = 0;                   // normal
            PC_SIG[set][way]    = static_cast<uint8_t>(idx);
        }
    }

    // Advance epoch on each fill (any type), with small storage-friendly period
    epoch_counter++;
    if (epoch_counter >= EPOCH_PERIOD) {
        GLOBAL_EPOCH ^= 1u;
        epoch_counter = 0;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}