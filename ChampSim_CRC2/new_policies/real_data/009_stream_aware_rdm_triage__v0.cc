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

// ---------------- Tunables (knobs) ----------------
static constexpr uint8_t  PC_BITS          = 8;                // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);

static constexpr uint8_t  FRIEND_INIT      = 1;                // 2-bit (0..3), start slightly cold
static constexpr uint8_t  FRIEND_WARM_TH   = 2;                // >=2 => warm (likely multi-use)

static constexpr uint8_t  LINE_LOW_BITS    = 12;               // low bits of line# to compare
static constexpr uint8_t  STREAM_CONF_TH   = 2;                // >=2 => treat as stream near-bypass

static constexpr uint8_t  HITS_TO_PROMOTE  = 2;                // promote on 2nd+ demand hit

// Age: 0=young/MRU .. 3=old/LRU
static constexpr uint8_t  AGE_MAX          = 3;
static constexpr uint8_t  AGE_YOUNG        = 1;

// Sampled-set density: 1 out of 64 sets used for training
static constexpr uint8_t  SAMP_LOG2        = 6;                // sample when (set & ((1<<SAMP_LOG2)-1))==0

// ---------------- Per-line state (compact) ----------------
// Note: Arrays use byte storage; report minimal info-theoretic bits in the storage section.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: demand hits 0,1,2(=2+)
static uint8_t SENT   [LLC_SETS][LLC_WAYS];   // 1 bit: 1=quarantine/stream sentinel
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit pc index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: mirror validity for eviction training

// ---------------- Global predictors ----------------
// RDM-lite friendliness per PC (2-bit)
static uint8_t FRIEND[PC_SIZE];               // 0..3

// RunStride per PC: ±1/±2 forward-run detector
static uint16_t RS_LAST_LINE[PC_SIZE];        // store low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit: last step |delta| in {1,2}
static uint8_t  RS_LAST_FWD[PC_SIZE];         // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix for low-bit entropy
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
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMP_LOG2) - 1)) == 0);
}
// Update RunStride on demand access; returns updated confidence
static inline uint8_t update_runstride(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc(RS_CONF[idx], 3);        // two consecutive fwd steps of 1 or 2 lines
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RS_CONF[idx]);               // decay on irregular/backwards/big stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// Compare candidates: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting quarantined/stream-sentinel lines
    bool sa = (SENT[set][a] != 0);
    bool sb = (SENT[set][b] != 0);
    if (sa != sb) return sa;

    // 2) Prefer lines from colder PCs by FRIEND score
    uint8_t fa = FRIEND[ PCSIG[set][a] ];
    uint8_t fb = FRIEND[ PCSIG[set][b] ];
    if (fa != fb) return (fa < fb);

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
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
        FRIEND[i]        = FRIEND_INIT;
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
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
    const bool demand   = is_demand(type);
    const bool prefetch = (type == ACCESS_PREFETCH);
    const bool writebk  = (type == ACCESS_WRITEBACK);

    // Always gently age the set
    age_all(set);

    // Update RunStride only on demand references
    uint8_t stream_conf = 0;
    if (demand) stream_conf = update_runstride(PC, paddr);

    if (hit) {
        // Demand hits contribute to promotion; prefetch hits do not change hitcnt or promotion
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            // Multi-hit promotion gate: promote only on 2nd+ demand hit
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                AGE[set][way]  = 0;     // MRU
                SENT[set][way] = 0;     // shed any sentinel (stream/quarantine)
                // Reinforce friendliness on promotion event (helps short reuse)
                uint8_t idx = PCSIG[set][way];
                sat_inc(FRIEND[idx], 3);
            }
            // On first demand hit, do not promote (line remains aged by age_all)
        }
        // Nothing else to do on hits
        return;
    }

    // Miss: a new line will be filled into (set, way). Train on eviction of the old line if valid and sampled.
    if (LVALID[set][way] && is_sampled_set(set)) {
        uint8_t old_sig  = PCSIG[set][way];
        uint8_t old_hits = HITCNT[set][way];
        if (old_hits >= 2) sat_inc(FRIEND[old_sig], 3);
        else               sat_dec(FRIEND[old_sig]);
    }

    // Prepare insertion metadata for the new line
    uint32_t idx = pc_index(PC);
    bool pc_warm = (FRIEND[idx] >= FRIEND_WARM_TH);
    bool streamy = (demand && (stream_conf >= STREAM_CONF_TH));

    // Insert policy:
    // - Prefetch or streamy demand: hard tail with sentinel (near-bypass), never promote until 2nd demand hit
    // - Writeback: never bypass; insert at tail with sentinel to keep easily evictable
    // - Warm PCs: slightly young
    // - Cold PCs: tail
    PCSIG[set][way]  = static_cast<uint8_t>(idx);
    HITCNT[set][way] = 0;
    LVALID[set][way] = 1;

    if (prefetch || streamy || writebk) {
        AGE[set][way]  = AGE_MAX;   // tail
        SENT[set][way] = 1;         // quarantine/stream sentinel
    } else {
        AGE[set][way]  = pc_warm ? AGE_YOUNG : AGE_MAX;
        SENT[set][way] = 0;
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