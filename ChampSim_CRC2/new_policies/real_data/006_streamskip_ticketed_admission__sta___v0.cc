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
static constexpr uint8_t  PC_BITS         = 8;     // 256-entry per-PC tables (fits 8-bit per-line tag)
static constexpr uint32_t PC_SIZE         = (1u << PC_BITS);

// Ticket predictor (2-bit: 0..3) and threshold
static constexpr uint8_t  TICKET_INIT     = 1;     // slightly cold start
static constexpr uint8_t  TICKET_WARM_TH  = 2;     // >=2 => warm PC (likely multi-use)

// Run detector (per-PC ±1/±2 forward steps)
static constexpr uint8_t  LINE_LOW_BITS   = 12;    // low bits of line# for stride check
static constexpr uint8_t  STREAM_CONF_TH  = 2;     // >=2 => treat as stream near-bypass

// Ages: 0=young(MRU)..3=old(LRU)
static constexpr uint8_t  AGE_MAX         = 3;
static constexpr uint8_t  AGE_YOUNG       = 1;     // insertion depth for warm PCs

// Multi-hit promotion
static constexpr uint8_t  HITS_TO_PROMOTE = 2;     // promote only on 2nd+ demand hit

// Sampled-set density for ticket training
static constexpr uint8_t  SAMP_LOG2       = 6;     // 1 out of 64 sets

// ---------------- Per-line state (compact) ----------------
// Note: Arrays use bytes; see storage section for information-theoretic bits.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2b: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2b: 0,1,2(=2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2b: 0=normal,1=stream,2=quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8b: PC index (PC_BITS=8)
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1b: our view of validity for training

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global per-PC state ----------------
static uint8_t  TICKET[PC_SIZE];          // 2b: 0..3 (ticketed deadness)
static uint16_t RG_LAST_LINE[PC_SIZE];    // low LINE_LOW_BITS of line#
static uint8_t  RG_CONF[PC_SIZE];         // 2b confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];   // 1b: last step |delta| in {1,2}
static uint8_t  RG_LAST_FWD[PC_SIZE];     // 1b: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix for entropy into low bits
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
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
// Update per-PC run detector on a demand access; returns updated confidence
static inline uint8_t update_runconf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3);          // two consecutive forward ±1/±2 steps
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);                 // decay on break/backward/large stride
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict sentinels (stream/quarantine) first
    bool sa = (STATUS[set][a] != ST_NORMAL);
    bool sb = (STATUS[set][b] != ST_NORMAL);
    if (sa != sb) return sa;

    // 2) Evict lines from colder PCs (lower ticket)
    uint8_t ta = TICKET[ PCSIG[set][a] ];
    uint8_t tb = TICKET[ PCSIG[set][b] ];
    if (ta != tb) return (ta < tb);

    // 3) Prefer fewer observed demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older more evictable)
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
        TICKET[i]        = TICKET_INIT;
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

    // Age set gently to maintain a recency gradient
    age_all(set);

    // Demand access updates the run detector
    uint8_t run_conf = 0;
    if (demand) {
        run_conf = update_runconf(PC, paddr);
    }

    if (hit) {
        // On hit: count only demand hits; never promote on first hit or any prefetch hit
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;
            // Promote only when crossing the multi-hit threshold
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                AGE[set][way]    = 0;           // MRU
                STATUS[set][way] = ST_NORMAL;   // shed any sentinel
                // Reinforce the PC ticket on confirmed multi-use
                uint32_t pcidx = PCSIG[set][way];
                sat_inc(TICKET[pcidx], 3);
            }
        }
        // No other action on hits; prefetch hits do not promote
        return;
    }

    // Miss path: train on the victim being replaced (if previously valid) only on sampled sets
    if (LVALID[set][way] && is_sampled_set(set)) {
        uint32_t vpc   = PCSIG[set][way];
        uint8_t  vhits = HITCNT[set][way];
        if (vhits >= 2) sat_inc(TICKET[vpc], 3);
        else            sat_dec(TICKET[vpc]);
    }

    // Insert new line
    uint32_t pcidx = pc_index(PC);
    PCSIG[set][way]  = static_cast<uint8_t>(pcidx);
    HITCNT[set][way] = 0;
    LVALID[set][way] = 1;

    // Never bypass on writeback; quarantine WB/prefetch at hard tail
    if (type == ACCESS_PREFETCH || type == ACCESS_WRITEBACK) {
        AGE[set][way]    = AGE_MAX;
        STATUS[set][way] = ST_QUAR;
        return;
    }

    // Demand fill: use stream near-bypass or ticketed admission
    if (run_conf >= STREAM_CONF_TH) {
        // Stream near-bypass: hard tail + stream sentinel; never promotes until 2nd demand hit
        AGE[set][way]    = AGE_MAX;
        STATUS[set][way] = ST_STREAM;
        return;
    }

    // Ticketed admission: cold PCs insert old; warm PCs slightly young
    if (TICKET[pcidx] >= TICKET_WARM_TH) {
        AGE[set][way]    = AGE_YOUNG;
        STATUS[set][way] = ST_NORMAL;
    } else {
        AGE[set][way]    = AGE_MAX;
        STATUS[set][way] = ST_NORMAL;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}