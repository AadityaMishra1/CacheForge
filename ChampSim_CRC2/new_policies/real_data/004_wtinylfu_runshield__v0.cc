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
static constexpr uint8_t  PC_BITS          = 8;      // 256-entry PC-indexed structures
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);

static constexpr uint8_t  AGE_MAX          = 3;      // 0=MRU/young .. 3=LRU/tail
static constexpr uint8_t  AGE_YOUNG        = 1;

static constexpr uint8_t  HITS_TO_PROMOTE  = 2;      // promote on 2nd+ demand hit

static constexpr uint8_t  EPOCH_INIT       = 1;      // 2-bit epoch start
static constexpr uint8_t  EPOCH_WARM_TH    = 2;      // >=2 considered warm

static constexpr uint8_t  STREAM_CONF_TH   = 2;      // RunShield confidence to quarantine
static constexpr uint8_t  LINE_LOW_BITS    = 12;     // granularity for run detection

// TinyLFU Count-Min Sketch parameters
static constexpr uint8_t  CMS_ROWS         = 3;
static constexpr uint16_t CMS_WIDTH        = 256;    // match PC index width
static constexpr uint8_t  CMS_BITS         = 4;      // 0..15 saturating
static constexpr uint8_t  CMS_MAX          = (1u << CMS_BITS) - 1;
static constexpr uint32_t CMS_AGING_PERIOD = 4096;   // halve counts periodically
static constexpr uint8_t  ADMIT_BIAS       = 1;      // favor admission when within 1

// ---------------- Per-line state (compact conceptual bits) ----------------
// Arrays use byte storage; report minimal info bits in storage section.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t SENT   [LLC_SETS][LLC_WAYS];   // 1 bit: quarantine/stream/prefetch sentinel
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: mirror validity for training

// ---------------- Global predictors ----------------
// PC epoch (deadness bias), 2-bit per PC
static uint8_t PC_EPOCH[PC_SIZE];             // 0..3

// RunShield per PC: ±1/±2 forward-run detector
static uint16_t RS_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit
static uint8_t  RS_LAST_FWD[PC_SIZE];         // 1-bit

// TinyLFU Count-Min Sketch: CMS_ROWS x CMS_WIDTH, 4-bit counters
static uint8_t CMS[CMS_ROWS][CMS_WIDTH];      // store 0..15 in each byte (use lower 4 bits)
static uint32_t cms_tick = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Lightweight mixer to decorrelate low bits
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 21);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc_u2(uint8_t &v) { if (v < 3) v++; }
static inline void sat_dec_u2(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
static inline uint32_t cms_hash(uint8_t row, uint32_t key) {
    // Three simple xorshift salts mapped to [0, CMS_WIDTH-1]
    static constexpr uint32_t SALT[CMS_ROWS] = {0x9e3779b9u, 0x7f4a7c15u, 0x94d049bbu};
    uint32_t x = key ^ (SALT[row] >> (row + 1));
    x ^= (x >> 7);
    x ^= (x >> 13);
    return x & (CMS_WIDTH - 1);
}
static inline uint8_t cms_query(uint32_t key) {
    uint8_t m = 0xFF;
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_hash(r, key);
        uint8_t c = CMS[r][idx] & CMS_MAX;
        if (c < m) m = c;
    }
    return (m == 0xFF) ? 0 : m;
}
static inline void cms_add(uint32_t key) {
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_hash(r, key);
        uint8_t c = CMS[r][idx] & CMS_MAX;
        if (c < CMS_MAX) CMS[r][idx] = static_cast<uint8_t>(c + 1);
    }
    cms_tick++;
    if (cms_tick % CMS_AGING_PERIOD == 0) {
        for (uint8_t r = 0; r < CMS_ROWS; r++) {
            for (uint32_t i = 0; i < CMS_WIDTH; i++) {
                CMS[r][i] = static_cast<uint8_t>((CMS[r][i] & CMS_MAX) >> 1);
            }
        }
    }
}
// Update RunShield on demand access; returns updated confidence
static inline uint8_t update_runshield(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc_u2(RS_CONF[idx]);     // two consecutive forward small strides
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u2(RS_CONF[idx]);         // decay on irregular/backwards/large stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Quarantined (stream/prefetch) first
    bool qa = (SENT[set][a] != 0);
    bool qb = (SENT[set][b] != 0);
    if (qa != qb) return qa;

    // 2) Colder PC epoch
    uint8_t ea = PC_EPOCH[ PCSIG[set][a] ];
    uint8_t eb = PC_EPOCH[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb);

    // 3) Fewer demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Older age
    return AGE[set][a] > AGE[set][b];
}

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
        PC_EPOCH[i]       = EPOCH_INIT;
        RS_LAST_LINE[i]   = 0;
        RS_CONF[i]        = 0;
        RS_LAST_ABS12[i]  = 0;
        RS_LAST_FWD[i]    = 0;
    }
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_WIDTH; i++) CMS[r][i] = 0;
    }
    cms_tick = 0;
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
    const bool is_pf  = (type == ACCESS_PREFETCH);
    const bool is_wb  = (type == ACCESS_WRITEBACK);

    // Age recency for the set on each access
    age_all(set);

    // RunShield + TinyLFU update on demand accesses
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_runshield(PC, paddr);
    }

    uint32_t idx_pc = pc_index(PC);

    // On hits: multi-hit promotion (only on 2nd+ demand hit)
    if (hit) {
        if (demand) {
            // Frequency accounting for TinyLFU
            cms_add(idx_pc);

            // Update hit count (cap at 2)
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            // Promote only on reaching HITS_TO_PROMOTE via demand
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                AGE[set][way]  = 0;      // MRU
                SENT[set][way] = 0;      // shed quarantine
                if (PC_EPOCH[idx_pc] < 3) PC_EPOCH[idx_pc]++; // warm the PC epoch
            }
        }
        // Do not promote on prefetch hits or first hit
        return;
    }

    // Miss: train on eviction of the chosen way (if it held a valid line)
    if (LVALID[set][way]) {
        uint8_t vpc = PCSIG[set][way];
        uint8_t vh  = HITCNT[set][way];
        if (vh >= HITS_TO_PROMOTE) {
            if (PC_EPOCH[vpc] < 3) PC_EPOCH[vpc]++;
        } else {
            if (PC_EPOCH[vpc] > 0) PC_EPOCH[vpc]--;
        }
    }

    // Admission decision for the incoming miss
    uint8_t age_ins   = AGE_YOUNG;
    uint8_t sentinel  = 0;

    if (is_wb) {
        // Never bypass on writeback; insert conservatively, no sentinel
        age_ins  = (AGE_MAX > 1) ? (AGE_MAX - 1) : AGE_MAX;
        sentinel = 0;
    } else {
        // Estimate frequencies (TinyLFU)
        uint8_t inc_est = cms_query(idx_pc);
        uint8_t vic_est = 0;
        if (LVALID[set][way]) {
            vic_est = cms_query( PCSIG[set][way] );
        }

        bool deny = false;
        // Quarantine prefetches and confident streams
        if (is_pf) deny = true;
        if (demand && (stream_conf >= STREAM_CONF_TH)) deny = true;

        // Bias by PC epoch deadness
        if (PC_EPOCH[idx_pc] < EPOCH_WARM_TH) deny = true;

        // TinyLFU comparison (give slight bias to admit)
        if (!deny && (inc_est + ADMIT_BIAS <= vic_est)) deny = true;

        if (deny) {
            age_ins  = AGE_MAX;  // hard tail
            sentinel = 1;        // quarantine
        } else {
            age_ins  = AGE_YOUNG;
            sentinel = 0;
        }
    }

    // Install the new line
    PCSIG[set][way]  = static_cast<uint8_t>(idx_pc);
    HITCNT[set][way] = 0;
    SENT[set][way]   = sentinel;
    AGE[set][way]    = age_ins;
    LVALID[set][way] = 1;

    // Account the incoming demand in TinyLFU after using inc_est (window order)
    if (demand) {
        cms_add(idx_pc);
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