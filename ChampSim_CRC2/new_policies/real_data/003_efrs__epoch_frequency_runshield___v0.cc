#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access tags (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables (knobs) ----------------
// PC tables
static constexpr uint8_t  PC_BITS          = 8;     // 256-entry per-PC tables
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);

// Epoch-decayed per-PC frequency (3-bit saturating 0..7)
static constexpr uint8_t  FREQ_INIT        = 2;
static constexpr uint8_t  FREQ_WARM_TH     = 3;     // >=3 => warm insertion

// RunShield stream detector (±1/±2 forward run)
static constexpr uint8_t  LINE_LOW_BITS    = 12;    // low bits of line# to compare
static constexpr uint8_t  STREAM_CONF_TH   = 2;     // >=2 => stream quarantine

// Insertion depth (age 0 young/MRU .. AGE_MAX old/LRU)
static constexpr uint8_t  AGE_MAX          = 3;
static constexpr uint8_t  AGE_YOUNG        = 1;

// Multi-hit promotion (only on 2nd+ demand hit)
static constexpr uint8_t  HITS_TO_PROMOTE  = 2;

// Epoch settings for decay
static constexpr uint8_t  EPOCH_MOD        = 4;     // 2-bit epoch id
static constexpr uint32_t EPOCH_TICKS      = 4096;  // fills per epoch

// ---------------- Per-line state (compact) ----------------
// Note: Arrays are stored as bytes in C++; storage section reports minimal bits.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t SENT   [LLC_SETS][LLC_WAYS];   // 1 bit: 1=quarantine(stream/pref)
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: valid mirror for training

// ---------------- Global predictors ----------------
// Per-PC epoch-decayed frequency and epoch tag
static uint8_t PC_FREQ      [PC_SIZE];        // 3-bit 0..7
static uint8_t PC_LAST_EPOCH[PC_SIZE];        // 2-bit epoch tag

// RunShield per-PC state
static uint16_t RS_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  RS_CONF     [PC_SIZE];        // 2-bit (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit: last |delta| in {1,2}
static uint8_t  RS_LAST_FWD [PC_SIZE];        // 1-bit: last step forward

// Global epoch bookkeeping
static uint8_t  GLOBAL_EPOCH = 0;             // 2-bit
static uint64_t fills_since_epoch = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Lightweight hash for entropy
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline bool is_pref(uint32_t type) { return type == ACCESS_PREFETCH; }
static inline bool is_wb(uint32_t type) { return type == ACCESS_WRITEBACK; }
static inline void sat_inc(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
static inline void maybe_bump_epoch_on_fill() {
    fills_since_epoch++;
    if ((fills_since_epoch % EPOCH_TICKS) == 0) {
        GLOBAL_EPOCH = static_cast<uint8_t>((GLOBAL_EPOCH + 1) % EPOCH_MOD);
    }
}
static inline void decay_freq_on_access(uint32_t pcidx) {
    if (PC_LAST_EPOCH[pcidx] != GLOBAL_EPOCH) {
        // One-step decay on first touch in a new epoch
        PC_FREQ[pcidx] >>= 1;
        PC_LAST_EPOCH[pcidx] = GLOBAL_EPOCH;
    }
}
// Update RunShield; returns updated confidence
static inline uint8_t update_runshield(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc(RS_CONF[idx], 3);      // two consecutive forward ±1/±2 steps
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RS_CONF[idx]);             // decay on irregular/backward/large steps
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}
// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting quarantined (stream/prefetch) lines
    bool sa = (SENT[set][a] != 0);
    bool sb = (SENT[set][b] != 0);
    if (sa != sb) return sa;

    // 2) Prefer lines from colder PCs (lower frequency)
    uint8_t fa = PC_FREQ[ PCSIG[set][a] ];
    uint8_t fb = PC_FREQ[ PCSIG[set][b] ];
    if (fa != fb) return (fa < fb);

    // 3) Prefer fewer observed demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
    return AGE[set][a] > AGE[set][b];
}

// ---------------- ChampSim hooks ----------------

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            SENT[s][w]   = 0;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_FREQ[i]       = FREQ_INIT;
        PC_LAST_EPOCH[i] = 0;
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
    }
    GLOBAL_EPOCH = 0;
    fills_since_epoch = 0;
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
    // Composite priority selection
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
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);
    const bool pref   = is_pref(type);
    const bool wb     = is_wb(type);

    // Update RunShield on demand accesses (hit or miss)
    uint8_t rs_conf = 0;
    if (demand) {
        rs_conf = update_runshield(PC, paddr);
    }

    uint32_t pcidx = pc_index(PC);
    decay_freq_on_access(pcidx); // apply epoch decay on first touch this epoch

    if (hit) {
        // Demand hits: multi-hit gating; no promotion for quarantined lines
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;
            if ((HITCNT[set][way] >= HITS_TO_PROMOTE) && (SENT[set][way] == 0)) {
                age_all(set);
                AGE[set][way] = 0; // promote to MRU
            }
        }
        // Prefetch hits or writebacks: never promote
        return;
    }

    // Miss: train victim before installing the new line
    if (LVALID[set][way]) {
        uint8_t v_pcsig = PCSIG[set][way];
        decay_freq_on_access(v_pcsig); // decay on use
        if (HITCNT[set][way] >= 2) sat_inc(PC_FREQ[v_pcsig], 7);
        else                       sat_dec(PC_FREQ[v_pcsig]);
    }

    // Set insertion defaults
    uint8_t new_age   = AGE_MAX;
    uint8_t new_sent  = 0;
    uint8_t new_hits  = 0;

    // Insertion policy
    if (pref) {
        // Prefetches always quarantined at tail
        new_age  = AGE_MAX;
        new_sent = 1;
    } else if (wb) {
        // Never bypass on writeback; insert old, normal
        new_age  = AGE_MAX;
        new_sent = 0;
    } else { // demand fill
        bool is_stream = (rs_conf >= STREAM_CONF_TH);
        if (is_stream) {
            // Near-bypass: quarantine at tail, never promote
            new_age  = AGE_MAX;
            new_sent = 1;
        } else {
            // PC-frequency guided insertion
            if (PC_FREQ[pcidx] >= FREQ_WARM_TH) {
                new_age  = AGE_YOUNG;
                new_sent = 0;
            } else {
                new_age  = AGE_MAX;
                new_sent = 0;
            }
        }
    }

    // Age set and install
    age_all(set);
    AGE[set][way]    = new_age;
    HITCNT[set][way] = new_hits;
    SENT[set][way]   = new_sent;
    PCSIG[set][way]  = static_cast<uint8_t>(pcidx);
    LVALID[set][way] = 1;

    // Epoch maintenance on fills
    maybe_bump_epoch_on_fill();
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}