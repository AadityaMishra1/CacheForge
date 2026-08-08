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
// Stream thresholds: base 3; allow earlier (2) if PC is cold (FRIEND<=1)
static constexpr uint8_t  STREAM_CONF_BASE  = 3;
static constexpr uint8_t  STREAM_CONF_COLD  = 2;
// Insertion depths
static constexpr uint8_t INS_FRIENDLY = 1;              // near-MRU for friendly PCs
static constexpr uint8_t INS_TAIL     = 3;              // tail
// Decay cadence for PC friendliness (phase adaptation) - slower (stable)
static constexpr uint32_t DECAY_PERIOD_MISSES = 64;     // one entry decays every 64 demand misses

// -------------------- Per-line state --------------------
// SRRIP re-reference prediction value (logical 0..3 stored in 8-bit)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];

// Per-line reuse state: 0 = no reuse yet, 1 = one demand hit seen, 2 = 2+ hits
static uint8_t LINE_STATE[LLC_SETS][LLC_WAYS];

// Per-line "no-train/no-promote" tag for stream/writeback lines
static uint8_t LINE_NT[LLC_SETS][LLC_WAYS]; // 0/1

// -------------------- Sampler state (sampled sets only) --------------------
// For sampled sets, keep PC-index and a "saw demand hit" bit to train friendliness on eviction
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

// Update per-PC stream detector, return confidence after this demand access
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

static inline uint8_t read_pc_stream_conf(uint64_t pc) {
    return ST_CONF[stream_index(pc)];
}

static inline uint8_t classify_friend(uint64_t pc) {
    return FRIEND[friend_index(pc)]; // 0..3
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
        FRIEND[i] = 1; // neutral-cold start
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
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: look for RRPV==3; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == 3) {
                // Train PC friendliness on sampled sets (on eviction)
                if (is_sampled_set(set)) {
                    if (LINE_NT[set][w] == 0) {
                        uint32_t fidx = SAMPLER_PCIDX[set][w] & (FRIEND_SIZE - 1);
                        if (SAMPLER_HIT[set][w]) sat_inc(FRIEND[fidx], 3);
                        else                      sat_dec(FRIEND[fidx]);
                    }
                    // clear sampler slot (fresh line will set it)
                    SAMPLER_HIT[set][w] = 0;
                    SAMPLER_PCIDX[set][w] = 0;
                }
                return w;
            }
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < 3) RRPV[set][w]++;
        }
    }
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
    const bool is_load   = (type == ACCESS_LOAD);
    const bool is_rfo    = (type == ACCESS_RFO);
    const bool is_pref   = (type == ACCESS_PREFETCH);
    const bool is_wb     = (type == ACCESS_WRITEBACK);
    const bool is_demand = is_load || is_rfo;

    // Stream detector update on demand accesses
    uint8_t stream_conf = is_demand ? update_pc_stream(PC, paddr) : read_pc_stream_conf(PC);

    // Classify PC friendliness
    uint8_t fclass = classify_friend(PC);
    uint8_t s_thresh = (fclass <= 1) ? STREAM_CONF_COLD : STREAM_CONF_BASE;
    bool stream_confident = (stream_conf >= s_thresh) && !is_wb; // never bypass/writeback

    if (hit) {
        // Hit behavior
        if (is_demand && LINE_NT[set][way] == 0) {
            // strict multi-hit: 1st hit -> RRPV=1, 2nd+ -> MRU
            if (LINE_STATE[set][way] == 0) {
                LINE_STATE[set][way] = 1;
                RRPV[set][way] = 1;
            } else {
                LINE_STATE[set][way] = 2;
                RRPV[set][way] = 0; // MRU
            }
            if (is_sampled_set(set)) {
                SAMPLER_HIT[set][way] = 1; // demand reuse observed
            }
        }
        // Prefetch and writeback hits never promote and never set sampler hit
        return;
    }

    // Miss / insertion path
    if (is_wb) {
        // Never bypass on writeback: tail insert, no-train/no-promote
        RRPV[set][way]       = INS_TAIL;
        LINE_NT[set][way]    = 1;
        LINE_STATE[set][way] = 0;
    } else if (stream_confident) {
        // Confident stream: approximate bypass with tail insert + no-train/no-promote
        RRPV[set][way]       = INS_TAIL;
        LINE_NT[set][way]    = 1;
        LINE_STATE[set][way] = 0;
    } else {
        // Non-stream insertion policy
        if (is_pref) {
            // Prefetch always inserts at tail, may promote later on demand hits
            RRPV[set][way]       = INS_TAIL;
            LINE_NT[set][way]    = 0;     // allow later training/promotion on demand
            LINE_STATE[set][way] = 0;
        } else {
            // Demand miss: SHiP-guided insertion
            if (fclass >= 2) {
                RRPV[set][way] = INS_FRIENDLY; // near MRU
            } else {
                RRPV[set][way] = INS_TAIL;     // tail for cold/unknown
            }
            LINE_NT[set][way]    = 0;
            LINE_STATE[set][way] = 0;
        }
    }

    // Initialize sampler book-keeping for sampled sets
    if (is_sampled_set(set)) {
        SAMPLER_PCIDX[set][way] = static_cast<uint16_t>(friend_index(PC));
        SAMPLER_HIT[set][way]   = 0;
    }

    // Decay friendliness slowly on demand misses
    if (is_demand) {
        demand_miss_ctr++;
        if ((demand_miss_ctr % DECAY_PERIOD_MISSES) == 0) {
            // Decay one entry in a round-robin manner
            sat_dec(FRIEND[decay_ptr]);
            decay_ptr = (decay_ptr + 1) & (FRIEND_SIZE - 1);
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