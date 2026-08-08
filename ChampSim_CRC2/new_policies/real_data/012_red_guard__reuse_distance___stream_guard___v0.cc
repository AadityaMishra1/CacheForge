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

// ---------------- Tunables ----------------
static constexpr uint8_t PC_BITS          = 8;    // 256-entry PC tables
static constexpr uint32_t PC_SIZE         = (1u << PC_BITS);

static constexpr uint8_t FRIEND_INIT      = 1;    // 2-bit (0..3)
static constexpr uint8_t FRIEND_WARM_TH   = 2;    // >=2 => warm

static constexpr uint8_t LINE_LOW_BITS    = 12;   // low bits of line# for stride detection
static constexpr uint8_t STREAM_CONF_TH   = 2;    // >=2 => stream near-bypass

static constexpr uint8_t HITS_TO_PROMOTE  = 2;    // promote on 2nd+ demand hit

// Age: 0=MRU/young .. 3=old/LRU
static constexpr uint8_t AGE_MAX          = 3;
static constexpr uint8_t AGE_YOUNG        = 1;

// Sampled-set density: 1 out of 64 sets
static constexpr uint8_t SAMP_LOG2        = 6;

// Reuse-distance "young" threshold (stack position <= this is considered early reuse)
static constexpr uint8_t RD_EARLY_TH      = 8;    // for 16-way sets

// ---------------- Per-line state ----------------
// Note: Arrays use bytes for simplicity; report info-theoretic bits in the storage section.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];  // 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];  // 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];  // 2-bit enum below
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];  // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];  // 0/1 mirror validity (for eviction training)

// Sampled-set LRU stack positions (only meaningful for sampled sets)
static uint8_t SPOS   [LLC_SETS][LLC_WAYS];  // 0=MRU .. 15=LRU

// ---------------- Global predictors ----------------
// Per-PC friendliness: 2-bit counter
static uint8_t FRIEND[PC_SIZE];              // 0..3

// Stream-guard per PC
static uint16_t RG_LAST_LINE[PC_SIZE];       // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];            // 2-bit confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];      // 1-bit
static uint8_t  RG_LAST_FWD[PC_SIZE];        // 1-bit

// ---------------- Status enum ----------------
enum : uint8_t {
    ST_NORMAL = 0,   // normal line
    ST_STREAM = 1,   // stream/quarantined by run detector (near-bypass)
    ST_QUAR   = 2,   // prefetch/writeback quarantine
    ST_COLD   = 3    // cold-PC sentinel
};

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Mix low entropy bits
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec(uint8_t &v) { if (v > 0) v--; }
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMP_LOG2) - 1)) == 0);
}

// Update stream-guard; returns updated confidence
static inline uint8_t update_stream_guard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3);    // two consecutive forward ±1/±2 steps
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);           // decay on irregular/backward/big stride
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Sampled-set LRU maintenance
static inline void slru_on_access(uint32_t set, uint32_t way) {
    if (!is_sampled_set(set)) return;
    uint8_t pos = SPOS[set][way];
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (SPOS[set][w] < pos) SPOS[set][w]++;
    }
    SPOS[set][way] = 0;
}
static inline void slru_on_insert(uint32_t set, uint32_t way) {
    if (!is_sampled_set(set)) return;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (SPOS[set][w] < 15) SPOS[set][w]++;
    }
    SPOS[set][way] = 0;
}

// Age promotion to MRU (recency tiebreaker)
static inline void set_MRU(uint32_t set, uint32_t way) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
    AGE[set][way] = 0;
}

// Eviction comparator: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    auto pri = [&](uint32_t w) -> uint8_t {
        uint8_t st = STATUS[set][w];
        if (st == ST_STREAM || st == ST_QUAR) return 2; // top priority to evict
        if (st == ST_COLD) return 1;
        return 0; // normal
    };
    uint8_t pa = pri(a), pb = pri(b);
    if (pa != pb) return pa > pb;

    uint8_t fa = FRIEND[ PCSIG[set][a] ];
    uint8_t fb = FRIEND[ PCSIG[set][b] ];
    if (fa != fb) return (fa < fb); // evict colder-PC first

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // fewer hits first

    return AGE[set][a] > AGE[set][b]; // older first
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
            SPOS[s][w]   = static_cast<uint8_t>(w); // deterministic init
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        FRIEND[i]        = FRIEND_INIT;
        RG_LAST_LINE[i]  = 0;
        RG_CONF[i]       = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
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

    if (hit) {
        // First, reuse-distance training uses sampled-set stack position BEFORE reorder
        if (demand && is_sampled_set(set)) {
            uint8_t pos = SPOS[set][way];
            if (HITCNT[set][way] == 0 && pos <= RD_EARLY_TH) {
                uint8_t idx = PCSIG[set][way];
                sat_inc(FRIEND[idx], 3);
            }
        }

        // Demand hits: multi-hit gate
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++; // 0->1, 1->2(=2+)
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Clear any sentinel and promote to MRU
                STATUS[set][way] = ST_NORMAL;
                set_MRU(set, way);
            }
            // else: do not promote on first demand hit
        }
        // Prefetch hits: never promote; sentinel remains

        // Maintain sampled-set LRU after the access (for RD observation)
        slru_on_access(set, way);
        return;
    }

    // Miss path (fill). Train on eviction of previous valid line in this way.
    if (LVALID[set][way]) {
        if (is_sampled_set(set)) {
            uint8_t old_pc = PCSIG[set][way];
            // If line failed to achieve 2+ demand hits, treat as not friendly
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) sat_inc(FRIEND[old_pc], 3);
            else sat_dec(FRIEND[old_pc]);
        }
    }

    // Insert new line based on predictors
    const uint32_t pc_idx = pc_index(PC);
    uint8_t ins_status = ST_NORMAL;
    uint8_t ins_age    = AGE_YOUNG;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass WB; quarantine at tail
        ins_status = ST_QUAR;
        ins_age    = AGE_MAX;
    } else if (type == ACCESS_PREFETCH) {
        ins_status = ST_QUAR;   // always quarantined
        ins_age    = AGE_MAX;
    } else if (demand) {
        uint8_t sconf = update_stream_guard(PC, paddr);
        if (sconf >= STREAM_CONF_TH) {
            ins_status = ST_STREAM; // near-bypass for streams
            ins_age    = AGE_MAX;
        } else if (FRIEND[pc_idx] < FRIEND_WARM_TH) {
            ins_status = ST_COLD;   // cold-PC tail insertion
            ins_age    = AGE_MAX;
        } else {
            ins_status = ST_NORMAL; // warm PC
            ins_age    = AGE_YOUNG;
        }
    }

    // Commit insertion metadata
    AGE[set][way]    = ins_age;
    STATUS[set][way] = ins_status;
    HITCNT[set][way] = 0;            // demand hits observed so far
    PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
    LVALID[set][way] = 1;

    // Sampled-set stack update: new line becomes MRU
    slru_on_insert(set, way);
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}