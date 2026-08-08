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
static constexpr uint8_t  PC8_BITS        = 8;     // RunStride+ table: 256 PCs
static constexpr uint8_t  PC6_BITS        = 6;     // Per-line stored PC signature
static constexpr uint32_t PC8_SIZE        = (1u << PC8_BITS);

static constexpr uint8_t  LINE_LOW_BITS   = 12;    // lines: compare low 12 bits of line#
static constexpr uint8_t  STREAM_TH       = 2;     // >=2 => stream near-bypass
static constexpr uint8_t  HITS_TO_PROMOTE = 2;     // second demand hit promotes

// Age: 0=young/MRU .. 3=old/LRU
static constexpr uint8_t  AGE_MAX         = 3;
static constexpr uint8_t  AGE_WARM        = 1;     // insertion for reuse-BF PCs
static constexpr uint8_t  AGE_UNKNOWN     = 2;     // neutral insertion

// Bloom filters (tiny, power-of-two size)
static constexpr uint32_t BF_BITS         = 4096;  // 4Kb per BF
static constexpr uint32_t BF_BYTES        = (BF_BITS >> 3);
static constexpr uint32_t BF_MASK         = (BF_BITS - 1);

// ---------------- Per-line state (bit-packed conceptually) ----------------
static uint8_t AGE      [LLC_SETS][LLC_WAYS];  // 2b: 0..3
static uint8_t HITCNT   [LLC_SETS][LLC_WAYS];  // 2b: demand hits 0,1,2(=2+),3(=2+)
static uint8_t STATUS   [LLC_SETS][LLC_WAYS];  // 2b: 0=normal,1=stream,2=quarantine(prefetch)
static uint8_t PRED_COLD[LLC_SETS][LLC_WAYS];  // 1b: 1=dead-pred at fill
static uint8_t PCSIG6   [LLC_SETS][LLC_WAYS];  // 6b: stored PC signature
static uint8_t LVALID   [LLC_SETS][LLC_WAYS];  // 1b: track prior validity for eviction training

// ---------------- Global predictors ----------------
// RunStride+ per-PC
static uint16_t RS_LAST_LINE[PC8_SIZE];   // store low LINE_LOW_BITS of line#
static uint8_t  RS_CONF     [PC8_SIZE];   // 2b: 0..3
static uint8_t  RS_LAST_ABS12[PC8_SIZE];  // 1b
static uint8_t  RS_LAST_FWD  [PC8_SIZE];  // 1b

// Bloom filters
static uint8_t BF_DEAD [BF_BYTES];  // recent dead creators
static uint8_t BF_REUSE[BF_BYTES];  // recent reusers

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc_u8(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
static inline uint32_t pc8_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC8_SIZE - 1);
}
static inline uint8_t pc6_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 11) ^ (pc << 7);
    return static_cast<uint8_t>(x & ((1u << PC6_BITS) - 1));
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline uint32_t page_tag(uint64_t paddr) {
    uint64_t x = (paddr >> 12);
    x ^= (x >> 17);
    x *= 0x9e3779b97f4a7c15ull;
    x ^= (x >> 32);
    return static_cast<uint32_t>(x);
}
static inline uint32_t bf_hash1(uint32_t key) {
    uint32_t x = key * 0x7feb352dU;
    x ^= (x >> 15);
    return x & BF_MASK;
}
static inline uint32_t bf_hash2(uint32_t key) {
    uint32_t x = key ^ 0x9e3779b9U;
    x ^= (x << 13);
    x ^= (x >> 17);
    x ^= (x << 5);
    return x & BF_MASK;
}
static inline void bf_set(uint8_t* bf, uint32_t key) {
    uint32_t i1 = bf_hash1(key);
    uint32_t i2 = bf_hash2(key);
    bf[i1 >> 3] |= (1u << (i1 & 7));
    bf[i2 >> 3] |= (1u << (i2 & 7));
}
static inline bool bf_test(const uint8_t* bf, uint32_t key) {
    uint32_t i1 = bf_hash1(key);
    uint32_t i2 = bf_hash2(key);
    return ((bf[i1 >> 3] >> (i1 & 7)) & 1u) && ((bf[i2 >> 3] >> (i2 & 7)) & 1u);
}
// Update RunStride+ on demand access; returns updated confidence
static inline uint8_t update_runstride(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc8_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc_u8(RS_CONF[idx], 3);    // two consecutive forward ±1/±2 steps
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        if (RS_CONF[idx] > 0) RS_CONF[idx]--; // decay on break/back/irregular
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// Eviction comparator: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict sentinels first (prefetch quarantine or stream)
    bool sa = (STATUS[set][a] != 0);
    bool sb = (STATUS[set][b] != 0);
    if (sa != sb) return sa;

    // 2) Evict predicted-cold lines
    bool ca = (PRED_COLD[set][a] != 0);
    bool cb = (PRED_COLD[set][b] != 0);
    if (ca != cb) return ca;

    // 3) Fewer observed demand hits first
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Older age is more evictable
    if (AGE[set][a] != AGE[set][b]) return (AGE[set][a] > AGE[set][b]);

    // 5) Tie-breaker: higher way index
    return (a > b);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]       = AGE_MAX;
            HITCNT[s][w]    = 0;
            STATUS[s][w]    = 0;
            PRED_COLD[s][w] = 0;
            PCSIG6[s][w]    = 0;
            LVALID[s][w]    = 0;
        }
    }
    for (uint32_t i = 0; i < PC8_SIZE; i++) {
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
    }
    for (uint32_t i = 0; i < BF_BYTES; i++) {
        BF_DEAD[i]  = 0;
        BF_REUSE[i] = 0;
    }
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Composite selection
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
    // Update run detector only on demand accesses
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_runstride(PC, paddr);
    }

    if (hit) {
        if (demand) {
            // Count only demand hits
            sat_inc_u8(HITCNT[set][way], 3);
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Clear sentinel on confirmed reuse; promote to MRU
                if (STATUS[set][way] != 0) STATUS[set][way] = 0;
                age_all(set);
                AGE[set][way] = 0;
            }
        }
        // No promotion on prefetch hits
        return;
    }

    // Miss path: train victim before inserting new line
    if (LVALID[set][way]) {
        uint8_t prev_hits = HITCNT[set][way];
        uint8_t prev_pcs  = PCSIG6[set][way];
        uint32_t key_evict = (static_cast<uint32_t>(prev_pcs) << 26) ^ page_tag(victim_addr);
        if (prev_hits >= 2) {
            bf_set(BF_REUSE, key_evict);
        } else {
            bf_set(BF_DEAD, key_evict);
        }
    }

    // Prepare new insertion
    uint8_t pcs6 = pc6_index(PC);
    uint32_t key_fill = (static_cast<uint32_t>(pcs6) << 26) ^ page_tag(paddr);

    bool in_reuse = bf_test(BF_REUSE, key_fill);
    bool in_dead  = bf_test(BF_DEAD , key_fill);
    bool cold_pred = (in_dead && !in_reuse);

    // Age others to keep recency pressure on fills
    age_all(set);

    // Set per-line metadata
    PCSIG6[set][way]    = pcs6;
    HITCNT[set][way]    = 0;
    PRED_COLD[set][way] = cold_pred ? 1 : 0;
    LVALID[set][way]    = 1;

    if (type == ACCESS_PREFETCH) {
        // Quarantine prefetches at the tail
        STATUS[set][way] = 2;        // ST_QUAR
        AGE[set][way]    = AGE_MAX;  // hard tail
        return;
    }

    // Demand/WB insertion policy
    if (demand && (stream_conf >= STREAM_TH)) {
        // Stream near-bypass: insert at tail with stream sentinel; no early promotion
        STATUS[set][way] = 1;        // ST_STREAM
        AGE[set][way]    = AGE_MAX;  // hard tail
        return;
    }

    STATUS[set][way] = 0;            // normal
    if (in_reuse) {
        AGE[set][way] = AGE_WARM;    // slightly young for likely reusers
    } else if (cold_pred) {
        AGE[set][way] = AGE_MAX;     // tail for predicted-dead
    } else {
        AGE[set][way] = AGE_UNKNOWN; // neutral for unknowns
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