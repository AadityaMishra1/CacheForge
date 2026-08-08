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

// -------------------- Tunables --------------------
// Sampled sets (every Nth set) for PC friendliness training
static constexpr uint32_t SAMPLE_STRIDE = 32;           // 2048/32 = 64 sampled sets
// PC friendliness table (2-bit saturated)
static constexpr uint32_t FRIEND_BITS   = 11;           // 2048 entries
static constexpr uint32_t FRIEND_SIZE   = (1u << FRIEND_BITS);
// Per-PC stream detector (±1-line stride)
static constexpr uint32_t STREAM_TABLE_SIZE = 1024;     // small, hashed by PC
// Stream thresholds: base 3; allow earlier (2) if PC is strongly cold
static constexpr uint8_t  STREAM_CONF_BASE  = 3;
static constexpr uint8_t  STREAM_CONF_EARLY = 2;
// Insertion depths
static constexpr uint8_t INS_FRIENDLY = 0;              // MRU for friendly PCs
static constexpr uint8_t INS_NEUTRAL  = 2;              // mid-depth
static constexpr uint8_t INS_COLD     = 3;              // tail
// Decay cadence for PC friendliness (phase adaptation) - faster
static constexpr uint32_t DECAY_PERIOD_MISSES = 16;     // one entry decays every 16 demand misses

// -------------------- Per-line state --------------------
// SRRIP re-reference prediction value (logical 2-bit 0..3 stored in 8-bit)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line reuse state: 0 = no reuse yet, 1 = one hit, 2 = 2+ hits
static uint8_t LINE_STATE[LLC_SETS][LLC_WAYS];

// Per-line "no-train/no-promote" tag for stream/writeback lines
static uint8_t LINE_NT[LLC_SETS][LLC_WAYS]; // 0/1

// -------------------- Sampler state (sampled sets only) --------------------
// For sampled sets, keep PC-index and a "saw hit" bit to train friendliness on eviction
static uint16_t SAMPLER_PCIDX[LLC_SETS][LLC_WAYS]; // 11-bit stored in 16
static uint8_t  SAMPLER_HIT[LLC_SETS][LLC_WAYS];   // 0/1

// -------------------- Global predictors --------------------
// PC friendliness table: 2-bit saturating counters (0=cold .. 3=hot)
static uint8_t FRIEND[FRIEND_SIZE];

// Per-PC ±1-line stream detector
static uint16_t ST_LAST_LINE[STREAM_TABLE_SIZE]; // low16 of line number
static uint8_t  ST_CONF[STREAM_TABLE_SIZE];      // 2-bit confidence 0..3
static uint8_t  ST_LAST_ABS1[STREAM_TABLE_SIZE]; // last step was |delta|==1

// Decay machinery
static uint32_t decay_ptr = 0;
static uint32_t demand_miss_ctr = 0;

// -------------------- Helpers --------------------
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x)               { if (x > 0) x--; }

static inline bool is_sampled_set(uint32_t set) { return (set % SAMPLE_STRIDE) == 0; }

static inline uint32_t friend_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (FRIEND_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 6) ^ (pc >> 12);
    return static_cast<uint32_t>(x) & (STREAM_TABLE_SIZE - 1);
}
static inline uint16_t line_low16(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & 0xFFFFu);
}

// Update per-PC stream detector, return confidence after this access (demand only)
static inline uint8_t update_pc_stream(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_low16(paddr);
    uint16_t prev = ST_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(curr - prev);
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            sat_inc(ST_CONF[idx], 3); // consecutive ±1
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1; // first indication
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        sat_dec(ST_CONF[idx]);
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

static inline uint8_t classify_friend(uint64_t pc) {
    uint8_t c = FRIEND[friend_index(pc)];
    // 0=cold, 1=weak-cold, 2=weak-hot, 3=hot
    return c;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]       = 3;
            LINE_STATE[s][w] = 0;
            LINE_NT[s][w]    = 0;
            SAMPLER_PCIDX[s][w] = 0;
            SAMPLER_HIT[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < FRIEND_SIZE; i++) {
        FRIEND[i] = 1; // neutral-cold start (avoid overly aggressive early bypass)
    }
    for (uint32_t i = 0; i < STREAM_TABLE_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS1[i]  = 0;
    }
    decay_ptr = 0;
    demand_miss_ctr = 0;
}

// Find victim in the set (SRRIP with safe aging)
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // First, prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: look for RRPV==3; if none, age and retry
    for (int round = 0; round < 8; round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) {
                // Train PC friendliness on sampled sets (before replacement)
                if (is_sampled_set(set)) {
                    // Skip training for stream/writeback tagged lines
                    if (LINE_NT[set][w] == 0) {
                        uint32_t fidx = SAMPLER_PCIDX[set][w] & (FRIEND_SIZE - 1);
                        if (SAMPLER_HIT[set][w]) sat_inc(FRIEND[fidx], 3);
                        else                      sat_dec(FRIEND[fidx]);
                    }
                    // Clear sampler hit bit for safety; PCIDX will be set on insertion
                    SAMPLER_HIT[set][w] = 0;
                }
                return w;
            }
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
    return 0; // safe fallback (should not happen)
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
    const bool is_demand = (type == ACCESS_LOAD) || (type == ACCESS_RFO);

    // Update stream detector (demand only)
    uint8_t stream_conf = 0;
    if (is_demand) {
        stream_conf = update_pc_stream(PC, paddr);
    }

    // HIT path
    if (hit) {
        // Never promote/train for stream/writeback-tagged lines
        if (LINE_NT[set][way]) {
            return;
        }

        if (is_demand) {
            // Gentle first-hit: promote to near-MRU (1) on first demand hit
            if (LINE_STATE[set][way] == 0) {
                LINE_STATE[set][way] = 1;
                RRPV[set][way] = (RRPV[set][way] > 1) ? 1 : RRPV[set][way];
            } else {
                // Second+ demand hits: promote to MRU
                LINE_STATE[set][way] = 2;
                RRPV[set][way] = 0;
            }
            // Record hit for sampled sets
            if (is_sampled_set(set)) {
                SAMPLER_HIT[set][way] = 1;
            }
        }
        // Prefetch hits (type==ACCESS_PREFETCH) do not promote
        return;
    }

    // MISS/INSERT path
    if (is_demand) {
        // Demand miss drives decay
        demand_miss_ctr++;
        if ((demand_miss_ctr % DECAY_PERIOD_MISSES) == 0) {
            uint32_t idx = decay_ptr & (FRIEND_SIZE - 1);
            sat_dec(FRIEND[idx]);
            decay_ptr++;
        }
    }

    // Decide insertion behavior
    uint8_t ins_rrpv = INS_COLD;
    uint8_t fclass = classify_friend(PC);
    bool strong_cold = (fclass == 0);
    uint8_t stream_thr = strong_cold ? STREAM_CONF_EARLY : STREAM_CONF_BASE;
    bool stream_mark = false;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; keep at tail and never promote/train
        ins_rrpv = 3;
        stream_mark = true; // reuse LINE_NT to suppress training/promotion
    } else if (is_demand) {
        // Guarded stream shield
        if (stream_conf >= stream_thr) {
            // Aggressively shield: insert at tail and never promote/train
            ins_rrpv = 3;
            stream_mark = true;
        } else {
            // Use PC friendliness to choose insertion depth
            if (fclass >= 2)      ins_rrpv = INS_FRIENDLY; // MRU for friendly
            else if (fclass == 1) ins_rrpv = INS_NEUTRAL;
            else                  ins_rrpv = INS_COLD;
        }
    } else if (type == ACCESS_PREFETCH) {
        // Prefetch inserts at tail; later demand hits can promote
        ins_rrpv = 3;
        stream_mark = false;
    }

    // Initialize line metadata on insertion
    RRPV[set][way]       = ins_rrpv;
    LINE_STATE[set][way] = 0;
    LINE_NT[set][way]    = stream_mark ? 1 : 0;

    // Initialize sampler tracking only for demand allocations in sampled sets
    if (is_sampled_set(set)) {
        if (is_demand && !stream_mark) {
            SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(friend_index(PC));
            SAMPLER_HIT[set][way]   = 0;
        } else {
            // For non-demand or stream-tagged allocations, avoid training
            SAMPLER_PCIDX[set][way] = 0;
            SAMPLER_HIT[set][way]   = 0;
        }
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