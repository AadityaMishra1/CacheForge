#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS         = 8;                // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE         = (1u << PC_BITS);

static constexpr uint8_t  DUE_INIT        = 1;                // 2-bit counters: 0..3
static constexpr uint8_t  DUE_WARM_TH     = 2;                // >=2 => warm (likely multi-use)

static constexpr uint8_t  LINE_LOW_BITS   = 12;               // track low bits of line# for stride check
static constexpr uint8_t  STREAM_BYPASS_TH= 2;                // >=2 => treat as near-bypass
static constexpr uint8_t  HITS_TO_PROMOTE = 2;                // demand hits needed to promote

static constexpr uint8_t  AGE_MAX         = 3;                // 2-bit age (0 young .. 3 old)
static constexpr uint8_t  AGE_YOUNG       = 1;                // young insertion for warm PCs

// Status: 0=normal, 1=stream (near-bypass), 2=quarantine (prefetch/WB)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state ----------------
static uint8_t  AGE    [LLC_SETS][LLC_WAYS];   // 2 bits
static uint8_t  HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t  STATUS [LLC_SETS][LLC_WAYS];   // 2 bits: ST_*
static uint8_t  PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t  LVALID [LLC_SETS][LLC_WAYS];   // 1 bit mirror of line validity for training-on-evict

// ---------------- Global predictors ----------------
// DUE: per-PC 2-bit dead-use estimator
static uint8_t DUE[PC_SIZE];

// StreamRun64: per-PC ±1/±2 forward-run detector
static uint16_t SR_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t  SR_CONF[PC_SIZE];        // 2-bit confidence (0..3)
static uint8_t  SR_LAST_ABS12[PC_SIZE];  // 1-bit: last step |delta| in {1,2}
static uint8_t  SR_LAST_FWD[PC_SIZE];    // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
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
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
// Update StreamRun64 on demand access and return resulting confidence
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(SR_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (SR_LAST_ABS12[idx] && SR_LAST_FWD[idx]) {
            sat_inc(SR_CONF[idx], 3);       // two consecutive forward steps
        } else {
            if (SR_CONF[idx] == 0) SR_CONF[idx] = 1;
            SR_LAST_ABS12[idx] = 1;
            SR_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(SR_CONF[idx]);              // decay on break/backward/irregular
        SR_LAST_ABS12[idx] = 0;
        SR_LAST_FWD[idx]   = 0;
    }
    SR_LAST_LINE[idx] = curr;
    return SR_CONF[idx];
}

// Compare two candidates: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting sentinels (stream/quarantine)
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // sentinel more evictable

    // 2) Prefer colder PCs by DUE score
    uint8_t due_a = DUE[ PCSIG[set][a] ];
    uint8_t due_b = DUE[ PCSIG[set][b] ];
    if (due_a != due_b) return (due_a < due_b);

    // 3) Prefer lines with fewer demand hits (0 < 1 < 2+)
    uint8_t hc_a = HITCNT[set][a];
    uint8_t hc_b = HITCNT[set][b];
    if (hc_a != hc_b) return (hc_a < hc_b);

    // 4) Tie-break by age (older is more evictable)
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
            LVALID[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        DUE[i]          = DUE_INIT;
        SR_LAST_LINE[i] = 0;
        SR_CONF[i]      = 0;
        SR_LAST_ABS12[i]= 0;
        SR_LAST_FWD[i]  = 0;
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
    // Immediate allocation to any invalid way
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
    const bool is_pref = (type == ACCESS_PREFETCH);
    const bool is_wb   = (type == ACCESS_WRITEBACK);

    uint32_t pc_idx = pc_index(PC);

    // Update stream detector on all demand accesses (hit or miss)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    if (hit) {
        // On hit: multi-hit gating and cautious promotion
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++; // demand-only
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Promote to MRU and clear sentinel
                age_all(set);
                AGE[set][way] = 0;
                STATUS[set][way] = ST_NORMAL;
                // Reinforce DUE on confirmed reuse
                sat_inc(DUE[ PCSIG[set][way] ], 3);
            }
            // Do not promote on first demand hit (gate holds)
        } else {
            // Prefetch or writeback hit: no promotion/training
        }
        return;
    }

    // Miss: we are inserting into 'way' (evict if valid)
    // Train DUE on eviction of prior line in this way (if any)
    if (LVALID[set][way]) {
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_pc   = PCSIG[set][way];
        if (old_hits >= 2) {
            sat_inc(DUE[old_pc], 3);
        } else {
            sat_dec(DUE[old_pc]);
        }
    }

    // Age the set for recency ordering (demand fills refresh)
    if (demand) age_all(set);

    // Decide insertion state
    uint8_t st = ST_NORMAL;
    uint8_t ins_age = AGE_MAX;

    if (is_pref) {
        // Prefetch quarantine at hard tail
        st = ST_QUAR;
        ins_age = AGE_MAX;
    } else if (is_wb) {
        // Never bypass WB; insert old and safe
        st = ST_QUAR;
        ins_age = AGE_MAX;
    } else { // demand fill
        bool due_warm = (DUE[pc_idx] >= DUE_WARM_TH);
        if (stream_conf >= STREAM_BYPASS_TH) {
            st = ST_STREAM;
            ins_age = AGE_MAX; // near-bypass
        } else if (due_warm) {
            st = ST_NORMAL;
            ins_age = AGE_YOUNG; // slightly young for likely reusers
        } else {
            st = ST_NORMAL;
            ins_age = AGE_MAX;   // cold PCs: hard tail
        }
    }

    // Commit new line metadata
    AGE[set][way]    = ins_age;
    STATUS[set][way] = st;
    HITCNT[set][way] = 0;           // demand hits observed so far
    PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
    LVALID[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}