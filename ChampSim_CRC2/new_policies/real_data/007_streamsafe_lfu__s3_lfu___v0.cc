#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables (knobs) ----------------
static constexpr uint8_t  PC_BITS           = 8;      // 256-entry per-PC tables
static constexpr uint32_t PC_SIZE           = (1u << PC_BITS);

// TinyLFU per-PC frequency counters (4-bit)
static constexpr uint8_t  LFU_INIT          = 1;
static constexpr uint8_t  LFU_WARM_TH       = 2;      // >=2 => warm (multi-use inclined)
static constexpr uint32_t DECAY_PERIOD      = 4096;   // events between global half-decay

// Run64 stream detector (±1/±2 forward steps)
static constexpr uint8_t  LINE_LOW_BITS     = 12;     // low bits of line number for stride check
static constexpr uint8_t  STREAM_TH         = 2;      // >=2 => confident stream (near-bypass)

// Insertion ages (0 young/MRU .. 3 old)
static constexpr uint8_t  AGE_MAX           = 3;
static constexpr uint8_t  AGE_YOUNG         = 1;      // for warm PCs (non-stream)
static constexpr uint8_t  HITS_TO_PROMOTE   = 2;      // promote on 2nd+ demand hit

// Line status: normal, stream (near-bypass), quarantine (prefetch/WB)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];  // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];  // 2 bits: 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];  // 2 bits: ST_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];  // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];  // 1-bit: line ever allocated

// ---------------- Global predictors ----------------
// TinyLFU: per-PC 4-bit counters (stored in 8-bit)
static uint8_t PC_LFU[PC_SIZE];

// Run64: per-PC stream detector state
static uint16_t RG_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];        // 2-bit confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];  // 1-bit: last |delta| in {1,2}
static uint8_t  RG_LAST_FWD[PC_SIZE];    // 1-bit: last step was forward

// Epoch event counter for TinyLFU decay
static uint32_t epoch_events = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // simple folded hash to 8 bits
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc_u8(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec_u8(uint8_t &v)               { if (v > 0) v--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update Run64 on demand access; return updated confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc_u8(RG_CONF[idx], 3); // two consecutive forward steps -> strengthen stream
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RG_CONF[idx]);        // decay on break/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare two candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting sentinels first (stream/quarantine)
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer colder PCs by TinyLFU score
    uint8_t lfu_a = PC_LFU[ PCSIG[set][a] ];
    uint8_t lfu_b = PC_LFU[ PCSIG[set][b] ];
    if (lfu_a != lfu_b) return (lfu_a < lfu_b);

    // 3) Prefer fewer observed demand hits
    uint8_t hc_a = HITCNT[set][a];
    uint8_t hc_b = HITCNT[set][b];
    if (hc_a != hc_b) return (hc_a < hc_b);

    // 4) Tie-break by age (older is more evictable)
    return AGE[set][a] > AGE[set][b];
}

// Decay TinyLFU counts every DECAY_PERIOD events (halving)
static inline void try_decay_lfu() {
    epoch_events++;
    if ((epoch_events & (DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_SIZE; i++) {
            PC_LFU[i] >>= 1;
        }
    }
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
        PC_LFU[i]       = LFU_INIT;
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS12[i]= 0;
        RG_LAST_FWD[i]  = 0;
    }
    epoch_events = 0;
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

    // Global aging on any access to this set
    age_all(set);

    // Train Run64 on demand references (hits or misses)
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_runguard(PC, paddr);
    }

    // On hit: obey strict multi-hit gate
    if (hit) {
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Confirmed reuse: clear sentinels, promote to MRU, reinforce LFU
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way]    = 0;
                uint32_t idx = pc_index(PCSIG[set][way]);
                (void)idx; // PCSIG holds idx already
                sat_inc_u8(PC_LFU[ PCSIG[set][way] ], 15);
            }
        }
        // No promotion on first hit or on prefetch hits
        try_decay_lfu();
        return;
    }

    // Miss path: potential eviction training on the victim slot
    if (LVALID[set][way]) {
        // If victim saw ≤1 demand hit, penalize its PC
        if (HITCNT[set][way] <= 1) {
            uint8_t &ctr = PC_LFU[ PCSIG[set][way] ];
            sat_dec_u8(ctr);
        }
    }

    // Decide insertion policy for the incoming line
    const uint32_t pc_idx = pc_index(PC);

    uint8_t ins_status = ST_NORMAL;
    uint8_t ins_age    = AGE_MAX;

    if (type == ACCESS_PREFETCH) {
        // Prefetches are quarantined at the tail
        ins_status = ST_QUAR;
        ins_age    = AGE_MAX;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; keep it easy to evict
        ins_status = ST_QUAR;
        ins_age    = AGE_MAX;
    } else if (demand) {
        // Demand: check stream detector first (near-bypass)
        if (stream_conf >= STREAM_TH) {
            ins_status = ST_STREAM; // aggressively evictable, no early promotion
            ins_age    = AGE_MAX;
        } else {
            // Non-stream: use TinyLFU warmth to guide insertion
            if (PC_LFU[pc_idx] >= LFU_WARM_TH) {
                ins_status = ST_NORMAL;
                ins_age    = AGE_YOUNG;
            } else {
                ins_status = ST_NORMAL;
                ins_age    = AGE_MAX;
            }
        }
    }

    // Install new line metadata
    STATUS[set][way] = ins_status;
    AGE[set][way]    = ins_age;
    HITCNT[set][way] = 0;                 // first touch just occurred; no demand hits yet
    PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
    LVALID[set][way] = 1;

    // Update TinyLFU on positive feedback only (promotion handled on future hits)
    // Light-touch feedback at fill for warm PCs to aid phase pickup
    if (demand && PC_LFU[pc_idx] >= LFU_WARM_TH) {
        sat_inc_u8(PC_LFU[pc_idx], 15);
    }

    try_decay_lfu();
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}