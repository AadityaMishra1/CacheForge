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

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS    = 8;                 // 256-entry per-PC tables
static constexpr uint32_t PC_SIZE    = (1u << PC_BITS);
static constexpr uint8_t  LINE_LOW_BITS = 8;              // for stream detector

// StreamGate
static constexpr uint8_t  STREAM_CONF_BYPASS = 2;         // >=2 => near-bypass/quarantine

// Age (0=young..3=oldest)
static constexpr uint8_t  AGE_MAX   = 3;
static constexpr uint8_t  AGE_YOUNG = 1;                  // for hot fills
static constexpr uint8_t  AGE_TAIL  = 3;                  // tail insertion

// Ticketed promotion
static constexpr uint8_t  HITS_TO_PROMOTE = 2;            // promote on 2nd+ demand hit

// Tiny-Reuse Bloom (two arrays of 2-bit counters)
static constexpr uint32_t BLOOM_BITS  = 11;               // 2048 per array
static constexpr uint32_t BLOOM_SIZE  = (1u << BLOOM_BITS);
static constexpr uint8_t  BLOOM_HOT_TH = 2;               // >=2 => hot
static constexpr uint32_t BLOOM_DECAY_PERIOD = 8192;      // periodic counter decay

// Status: 0=normal, 1=stream(no promote/train until 2 hits), 2=quarantine(prefetch)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state ----------------
static uint8_t AGE[LLC_SETS][LLC_WAYS];       // 2 bits (0..3)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];    // 2 bits (0,1,2+)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];    // 2 bits (0..2)
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];     // 8-bit pc index
static uint8_t BHOT[LLC_SETS][LLC_WAYS];      // 1 bit: predicted hot at fill or confirmed at 2nd hit

// ---------------- StreamGate (PC-indexed) ----------------
static uint8_t  ST_LAST_LINE[PC_SIZE];  // low LINE_LOW_BITS of line number
static uint8_t  ST_CONF[PC_SIZE];       // 2-bit confidence (0..3)
static uint8_t  ST_LAST_ABS12[PC_SIZE]; // 1-bit: previous step was |delta|==1 or 2

// ---------------- Tiny-Reuse Bloom ----------------
static uint8_t BLOOM1[BLOOM_SIZE]; // 2-bit counters (0..3)
static uint8_t BLOOM2[BLOOM_SIZE]; // 2-bit counters (0..3)
static uint64_t BLOOM_TICK = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 17) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint64_t line_addr(uint64_t paddr) {
    return (paddr >> 6); // line granularity
}
static inline uint8_t line_lowN(uint64_t paddr) {
    return static_cast<uint8_t>(line_addr(paddr) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc2(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec2(uint8_t &x) { if (x > 0) x--; }

// Small 64-bit mixer -> 32-bit hash
static inline uint32_t mix32(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<uint32_t>(x ^ (x >> 32));
}
static inline void bloom_idx(uint32_t pc_idx, uint64_t paddr, uint32_t &i1, uint32_t &i2) {
    uint64_t key = (static_cast<uint64_t>(pc_idx) << 48) ^ line_addr(paddr);
    i1 = mix32(key) & (BLOOM_SIZE - 1);
    i2 = mix32(key ^ 0x9e3779b97f4a7c15ULL) & (BLOOM_SIZE - 1);
}
static inline bool bloom_is_hot(uint32_t pc_idx, uint64_t paddr) {
    uint32_t i1, i2; bloom_idx(pc_idx, paddr, i1, i2);
    return (BLOOM1[i1] >= BLOOM_HOT_TH) && (BLOOM2[i2] >= BLOOM_HOT_TH);
}
static inline void bloom_add_hot(uint32_t pc_idx, uint64_t paddr) {
    uint32_t i1, i2; bloom_idx(pc_idx, paddr, i1, i2);
    sat_inc2(BLOOM1[i1], 3);
    sat_inc2(BLOOM2[i2], 3);
}
static inline void bloom_dec_cold(uint32_t pc_idx, uint64_t paddr) {
    uint32_t i1, i2; bloom_idx(pc_idx, paddr, i1, i2);
    sat_dec2(BLOOM1[i1]);
    sat_dec2(BLOOM2[i2]);
}
static inline void bloom_decay_all() {
    for (uint32_t i = 0; i < BLOOM_SIZE; i++) {
        if (BLOOM1[i]) BLOOM1[i]--;
        if (BLOOM2[i]) BLOOM2[i]--;
    }
}

// Update StreamGate confidence and return it (demand only)
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint8_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(ST_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);

    if (abs12) {
        if (ST_LAST_ABS12[idx]) {
            sat_inc2(ST_CONF[idx], 3); // consecutive step confirms stream
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS12[idx] = 1;
        }
    } else {
        sat_dec2(ST_CONF[idx]);        // decay on break
        ST_LAST_ABS12[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

// Age all lines in a set (bounded)
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict stream/quarantine sentinels first
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer Bloom-cold lines (BHOT=0)
    if (BHOT[set][a] != BHOT[set][b]) return BHOT[set][a] < BHOT[set][b];

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    if (HITCNT[set][a] != HITCNT[set][b]) return HITCNT[set][a] < HITCNT[set][b];

    // 4) Tie-break by age: older is more evictable
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
            BHOT[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        ST_LAST_LINE[i]  = 0;
        ST_CONF[i]       = 0;
        ST_LAST_ABS12[i] = 0;
    }
    for (uint32_t i = 0; i < BLOOM_SIZE; i++) {
        BLOOM1[i] = 0;
        BLOOM2[i] = 0;
    }
    BLOOM_TICK = 0;
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

    // Composite priority victim selection
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
    // Periodic Bloom decay
    BLOOM_TICK++;
    if ((BLOOM_TICK & (BLOOM_DECAY_PERIOD - 1)) == 0) {
        bloom_decay_all();
    }

    const bool demand = is_demand(type);

    // Update stream detector on all demand accesses
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    // Maintain recency
    age_all(set);

    if (hit) {
        // Demand hit accounting and promotion gate
        if (demand) {
            uint8_t prev = HITCNT[set][way];
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;

            // On first demand hit: light refresh only (no promotion)
            if (prev == 0) {
                if (AGE[set][way] > 1) AGE[set][way] = 1;
            }

            // On 2nd demand hit: promote + positive Bloom train
            if (prev == (HITS_TO_PROMOTE - 1)) {
                AGE[set][way] = 0;          // MRU
                STATUS[set][way] = ST_NORMAL; // clear stream/quarantine once proven
                BHOT[set][way] = 1;         // confirmed hot
                // Train Bloom once at confirmation
                uint32_t pci = pc_index(PC);
                bloom_add_hot(pci, paddr);
            }
        } else {
            // Prefetch/writeback hits do not promote
            if (AGE[set][way] > 1) AGE[set][way] = 1;
        }
        return;
    }

    // Miss path: train on the evicted line (if any) before overwrite
    if (victim_addr != 0) {
        uint8_t prev_status = STATUS[set][way];
        uint8_t prev_hits   = HITCNT[set][way];
        uint8_t prev_pci    = PCSIG[set][way];

        // Negative training for dead (<2 hits) non-stream/non-prefetch lines
        if (prev_status == ST_NORMAL && prev_hits < HITS_TO_PROMOTE) {
            bloom_dec_cold(prev_pci, victim_addr);
        }
        // Positive training is done at 2nd hit time; no action here.
    }

    // Insert the new line
    uint32_t pci = pc_index(PC);
    PCSIG[set][way]  = static_cast<uint8_t>(pci);
    HITCNT[set][way] = 0;

    if (type == ACCESS_PREFETCH) {
        STATUS[set][way] = ST_QUAR;
        BHOT[set][way]   = 0;
        AGE[set][way]    = AGE_TAIL;
        return;
    }

    if (demand && (stream_conf >= STREAM_CONF_BYPASS)) {
        // Near-bypass for streams: tail insert, no promote/train until 2 hits
        STATUS[set][way] = ST_STREAM;
        BHOT[set][way]   = 0;
        AGE[set][way]    = AGE_TAIL;
        return;
    }

    // Normal demand or writeback: Bloom-directed insertion
    bool hot = bloom_is_hot(pci, paddr);
    STATUS[set][way] = ST_NORMAL;
    BHOT[set][way]   = hot ? 1 : 0;
    AGE[set][way]    = hot ? AGE_YOUNG : AGE_TAIL;
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}