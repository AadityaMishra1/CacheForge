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
static constexpr uint8_t  PC_BITS            = 8;   // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE            = (1u << PC_BITS);

static constexpr uint8_t  AGE_MAX            = 3;   // 2-bit age: 0..3 (0=young/MRU)
static constexpr uint8_t  AGE_YOUNG          = 1;   // insert for warm/small PCs

static constexpr uint8_t  HITS_TO_PROMOTE    = 2;   // promote on 2nd+ demand hit

static constexpr uint8_t  LINE_LOW_BITS      = 12;  // for RunFence: low bits of line#
static constexpr uint8_t  RF_NEAR_TAIL_TH    = 2;   // >=2 => near-bypass (tail insert)
static constexpr uint8_t  RF_BYPASS_TH       = 3;   // strong stream fence (treated as tail insert)

static constexpr uint8_t  FP_MAX             = 3;   // 2-bit footprint accumulator (0..3)
static constexpr uint8_t  FP_NEAR_TAIL_TH    = 2;   // large-ish footprint
static constexpr uint8_t  FP_BYPASS_TH       = 3;   // very large footprint

static constexpr uint8_t  LIFE_WARM_TH       = 2;   // lifetime >=2 => warm
static constexpr uint8_t  LIFE_INIT          = 1;   // start slightly cold

static constexpr uint8_t  SAMP_LOG2          = 6;   // sampled training: 1 in 64 sets

// ---------------- Per-line state (compact) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 0=normal,1=stream,2=quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit pc index

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors ----------------
// Per-PC lifetime (friendliness): 2-bit
static uint8_t LIFE[PC_SIZE];                 // 0..3

// Per-PC footprint estimator: 2-bit acc (0..3), trained on sampled sets
static uint8_t FP_ACC[PC_SIZE];               // 0..3
static uint16_t FP_LAST_SET[PC_SIZE];         // 11-bit last set id (store in 16)

// RunFence: stride ±1/±2 forward-run detector per PC
static uint16_t RF_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  RF_CONF[PC_SIZE];             // 0..3
static uint8_t  RF_LAST_ABS12[PC_SIZE];       // 0/1
static uint8_t  RF_LAST_FWD[PC_SIZE];         // 0/1

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix to reduce aliasing
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
}
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMP_LOG2) - 1)) == 0);
}

// Update RunFence; return new confidence
static inline uint8_t update_runfence(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RF_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RF_LAST_ABS12[idx] && RF_LAST_FWD[idx]) {
            sat_inc(RF_CONF[idx], 3); // two consecutive forward ±1/±2 steps
        } else {
            if (RF_CONF[idx] == 0) RF_CONF[idx] = 1;
            RF_LAST_ABS12[idx] = 1;
            RF_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RF_CONF[idx]);        // decay on irregular/backwards/big stride
        RF_LAST_ABS12[idx] = 0;
        RF_LAST_FWD[idx]   = 0;
    }
    RF_LAST_LINE[idx] = curr;
    return RF_CONF[idx];
}

// Footprint training on sampled sets
static inline void train_footprint(uint32_t set, uint64_t pc) {
    if (!is_sampled_set(set)) return;
    uint32_t idx = pc_index(pc);
    uint16_t s   = static_cast<uint16_t>(set & ((1u << 11) - 1)); // 11 bits for 2048 sets
    if (FP_LAST_SET[idx] != s) {
        sat_inc(FP_ACC[idx], FP_MAX); // distinct-set transition => larger footprint
        FP_LAST_SET[idx] = s;
    } else {
        sat_dec(FP_ACC[idx]);         // repeated same set => shrink footprint slowly
    }
}

// Compare candidates: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    uint8_t la = LIFE[ PCSIG[set][a] ];
    uint8_t lb = LIFE[ PCSIG[set][b] ];
    if (la != lb) return (la < lb);

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

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
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        LIFE[i]         = LIFE_INIT;
        FP_ACC[i]       = 0;
        FP_LAST_SET[i]  = 0;
        RF_LAST_LINE[i] = 0;
        RF_CONF[i]      = 0;
        RF_LAST_ABS12[i]= 0;
        RF_LAST_FWD[i]  = 0;
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

    // Train stream detector and footprint on demand references
    if (demand) {
        update_runfence(PC, paddr);
        train_footprint(set, PC);
    }

    // Global indices and signals
    uint32_t idx = pc_index(PC);
    uint8_t rf   = RF_CONF[idx];
    uint8_t life = LIFE[idx];
    uint8_t fp   = FP_ACC[idx];

    // Age all lines in the set to incorporate recency
    if (demand) age_all(set);

    if (hit) {
        // On hit: update hit counter and gated promotion
        if (demand) {
            if (HITCNT[set][way] < HITS_TO_PROMOTE) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                AGE[set][way]    = 0;           // MRU on 2nd+ demand hit
                STATUS[set][way] = ST_NORMAL;   // shed any sentinel
            }
        }
        // No promotion on prefetch hits or first hits by construction
        return;
    }

    // Miss path: train lifetime on eviction of victim line
    {
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_pc   = PCSIG[set][way];
        if (old_hits >= HITS_TO_PROMOTE) sat_inc(LIFE[old_pc], 3);
        else                             sat_dec(LIFE[old_pc]);
    }

    // Decide insertion policy (never bypass on writeback; prefetch quarantined)
    uint8_t new_status = ST_NORMAL;
    uint8_t new_age    = AGE_YOUNG; // default for warm/small PCs

    bool prefetch = (type == ACCESS_PREFETCH);
    bool wb       = (type == ACCESS_WRITEBACK);

    // Stream fence dominates for demand fills
    if (!wb && !prefetch && rf >= RF_NEAR_TAIL_TH) {
        new_status = ST_STREAM;
        new_age    = AGE_MAX; // near-bypass: insert at hard tail
    } else if (!wb && !prefetch) {
        // Footprint-gated insertion for cold PCs
        if ((fp >= FP_BYPASS_TH && life < LIFE_WARM_TH) ||
            (fp >= FP_NEAR_TAIL_TH && life == 0)) {
            new_status = ST_STREAM;  // treat as streaming-prone
            new_age    = AGE_MAX;    // hard tail
        } else if (life >= LIFE_WARM_TH) {
            new_status = ST_NORMAL;
            new_age    = AGE_YOUNG;  // slightly young for warm PCs
        } else {
            new_status = ST_QUAR;    // cold/uncertain -> quarantine tail
            new_age    = AGE_MAX;
        }
    }

    if (prefetch) {
        new_status = ST_QUAR;        // prefetch quarantined
        new_age    = AGE_MAX;        // tail
    }
    if (wb) {
        // Never bypass writebacks: insert at tail under quarantine to avoid pollution
        new_status = ST_QUAR;
        new_age    = AGE_MAX;
    }

    // Install metadata for the new line
    PCSIG[set][way]  = static_cast<uint8_t>(idx);
    STATUS[set][way] = new_status;
    AGE[set][way]    = new_age;
    HITCNT[set][way] = 0;            // first reference not yet seen
}

// Print end-of-simulation statistics
void PrintStats() {
    // intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // intentionally blank
}