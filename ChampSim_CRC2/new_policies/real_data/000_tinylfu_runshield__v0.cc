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
static constexpr uint8_t  PC_BITS            = 8;     // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE            = (1u << PC_BITS);

static constexpr uint8_t  AGE_MAX            = 3;     // 2-bit age: 0 (MRU) .. 3 (LRU)
static constexpr uint8_t  AGE_YOUNG          = 1;     // insertion age for warm PCs

static constexpr uint8_t  HITS_TO_PROMOTE    = 2;     // promote on 2nd+ demand hit

static constexpr uint8_t  STREAM_CONF_TH     = 2;     // RunShield confidence to treat as stream
static constexpr uint8_t  LINE_LOW_BITS      = 12;    // line# low bits for run detection

static constexpr uint8_t  TLFU_INIT          = 1;     // TinyLFU initial score
static constexpr uint8_t  TLFU_WARM_TH       = 4;     // >= threshold => warm PC
static constexpr uint8_t  TLFU_MAX           = 255;   // 8-bit saturating
static constexpr uint8_t  TLFU_SHIFT_LOG2    = 12;    // halve every 2^12 = 4096 accesses

// ---------------- Per-line state ----------------
// Minimal info-theoretic bits (implementation stored as bytes)
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2 bits: 0=normal, 1=stream, 2=quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: mirror validity for eviction training

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors ----------------
// TinyLFU per-PC friendliness (8-bit with periodic right-shift)
static uint8_t TLFU[PC_SIZE];

// RunShield per-PC: forward-run detector (pack conceptually into 16 bits/PC)
static uint16_t RS_LAST_LINE[PC_SIZE];  // store low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];       // 0..3 confidence
static uint8_t  RS_LAST_ABS12[PC_SIZE]; // 0/1: last step |delta| in {1,2}
static uint8_t  RS_LAST_FWD[PC_SIZE];   // 0/1: last step forward

// Global access counter for TinyLFU aging
static uint64_t ACCESS_CTR = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Mix for low-bit entropy; mask to PC_SIZE-1
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
static inline void sat_dec_u8(uint8_t &v) { if (v > 0) v--; }

static inline void age_all(uint32_t set) {
    // Per-access recency aging within the touched set (bounded and fast)
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
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
            sat_inc_u8(RS_CONF[idx], 3);      // two consecutive forward steps → strengthen stream confidence
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RS_CONF[idx]);             // decay on irregular/backward/long stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// TinyLFU periodic halving
static inline void tlfu_maybe_age() {
    ACCESS_CTR++;
    if ((ACCESS_CTR & ((1ull << TLFU_SHIFT_LOG2) - 1)) == 0) {
        for (uint32_t i = 0; i < PC_SIZE; i++) {
            TLFU[i] >>= 1;
        }
    }
}

// Compare candidates: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting quarantined/stream-sentinel lines
    bool sa = (STATUS[set][a] != ST_NORMAL);
    bool sb = (STATUS[set][b] != ST_NORMAL);
    if (sa != sb) return sa;

    // 2) Prefer lines from colder PCs (lower TinyLFU)
    uint8_t fa = TLFU[ PCSIG[set][a] ];
    uint8_t fb = TLFU[ PCSIG[set][b] ];
    if (fa != fb) return (fa < fb);

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
    return AGE[set][a] > AGE[set][b];
}

// ---------------- CRC2 API ----------------
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
        TLFU[i]          = TLFU_INIT;
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
    }
    ACCESS_CTR = 0;
}

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
    // Composite victim selection
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

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

    // Per-set age bump for recency
    age_all(set);

    // Maintain stream detector on demand accesses (hits or misses)
    if (demand) {
        (void)update_runshield(PC, paddr);
    }

    // TinyLFU global periodic aging
    tlfu_maybe_age();

    // On hit
    if (hit) {
        if (demand) {
            // Demand hit: multi-hit gate
            if (HITCNT[set][way] < 2) HITCNT[set][way]++; // 0->1->2(=2+)
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Confirmed reuse: promote and clear any sentinel
                AGE[set][way]   = 0;
                STATUS[set][way]= ST_NORMAL;
                // Reinforce PC friendliness
                uint32_t idx = PCSIG[set][way];
                sat_inc_u8(TLFU[idx], TLFU_MAX);
            } else {
                // First hit only: small rejuvenation, no promotion
                if (AGE[set][way] > 0) AGE[set][way]--;
            }
        } else {
            // Prefetch or writeback hit: never promote; optional gentle rejuvenation omitted
        }
        return;
    }

    // On miss/fill: train on eviction of previous occupant if valid
    if (LVALID[set][way]) {
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_pc   = PCSIG[set][way];
        if (old_hits >= 2) {
            sat_inc_u8(TLFU[old_pc], TLFU_MAX);
        } else {
            sat_dec_u8(TLFU[old_pc]);
        }
    }

    // Decide insertion for the new line
    uint32_t idx = pc_index(PC);
    uint8_t conf = RS_CONF[idx];
    bool streamy = (conf >= STREAM_CONF_TH);

    PCSIG[set][way]  = static_cast<uint8_t>(idx);
    HITCNT[set][way] = 0;
    LVALID[set][way] = 1;

    if (type == ACCESS_PREFETCH) {
        // Prefetch: quarantine at tail; never promote on prefetch hit
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on WB: quarantine at tail to avoid pollution
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Demand fill
    if (streamy) {
        // RunShield near-bypass: tail + stream sentinel; promote only on 2nd+ demand hit
        STATUS[set][way] = ST_STREAM;
        AGE[set][way]    = AGE_MAX;
    } else {
        // TinyLFU-guided adaptive insertion
        bool warm = (TLFU[idx] >= TLFU_WARM_TH);
        STATUS[set][way] = ST_NORMAL;
        AGE[set][way]    = warm ? AGE_YOUNG : AGE_MAX;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}