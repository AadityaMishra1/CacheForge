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
static constexpr uint8_t  PC_BITS          = 8;    // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS    = 12;   // low bits of line# for run detection
static constexpr uint8_t  STREAM_CONF_TH   = 2;    // >=2 => treat as stream near-bypass

static constexpr uint8_t  AGE_MAX          = 3;    // 0..3, 3 is tail
static constexpr uint8_t  AGE_YOUNG        = 1;    // young insertion for warm PCs

static constexpr uint8_t  HITS_TO_PROMOTE  = 2;    // promote on 2nd+ demand hit
static constexpr uint8_t  PS_INIT          = 1;    // PhaseScore init (0..3)
static constexpr uint8_t  PS_WARM_TH       = 2;    // >=2 => warm insertion

static constexpr uint8_t  SAMP_LOG2        = 6;    // sample 1 out of 64 sets

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 }; // per-line status

// ---------------- Per-line state (compact logical bits) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits (0..3)
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits (0,1,2=2+ demand hits)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2 bits (ST_*)
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit mirror validity for eviction training

// ---------------- Global predictors ----------------
// PhaseScore (PC⊕set): friendliness 2-bit, trained only at eviction on sampled sets
static uint8_t PSCORE[PC_SIZE];

// StreamRun per-PC: ±1/±2 forward-run detector
static uint16_t SR_LAST_LINE[PC_SIZE];  // low LINE_LOW_BITS of line#
static uint8_t  SR_CONF[PC_SIZE];       // 2-bit confidence (0..3)
static uint8_t  SR_LAST_ABS12[PC_SIZE]; // 1-bit: last step |delta| in {1,2}
static uint8_t  SR_LAST_FWD[PC_SIZE];   // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mixer for entropy into low bits
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
// Update StreamRun on demand access; returns updated confidence
static inline uint8_t update_streamrun(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(SR_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (SR_LAST_ABS12[idx] && SR_LAST_FWD[idx]) {
            sat_inc(SR_CONF[idx], 3); // two consecutive forward ±1/±2 steps
        } else {
            if (SR_CONF[idx] == 0) SR_CONF[idx] = 1;
            SR_LAST_ABS12[idx] = 1;
            SR_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(SR_CONF[idx]);  // decay on irregular/backwards/large stride
        SR_LAST_ABS12[idx] = 0;
        SR_LAST_FWD[idx]   = 0;
    }
    SR_LAST_LINE[idx] = curr;
    return SR_CONF[idx];
}

// PhaseScore index (PC⊕set)
static inline uint32_t pscore_index_from_pcisg(uint8_t pc_sig, uint32_t set) {
    return static_cast<uint32_t>((pc_sig ^ (set & (PC_SIZE - 1))) & (PC_SIZE - 1));
}
static inline uint32_t pscore_index_from_pc(uint64_t pc, uint32_t set) {
    return pscore_index_from_pcisg(static_cast<uint8_t>(pc_index(pc)), set);
}

// Compare candidates: return true if 'a' is more evictable than 'b'
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting quarantined/stream-sentinel lines
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer lines from colder PhaseScore contexts
    uint8_t ia = pscore_index_from_pcisg(PCSIG[set][a], set);
    uint8_t ib = pscore_index_from_pcisg(PCSIG[set][b], set);
    uint8_t pa = PSCORE[ia];
    uint8_t pb = PSCORE[ib];
    if (pa != pb) return (pa < pb);

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
    return AGE[set][a] > AGE[set][b];
}

// ---------------- API ----------------
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
        PSCORE[i]        = PS_INIT;
        SR_LAST_LINE[i]  = 0;
        SR_CONF[i]       = 0;
        SR_LAST_ABS12[i] = 0;
        SR_LAST_FWD[i]   = 0;
    }
}

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
    // Composite selection
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

    // Demand accesses update StreamRun
    uint8_t sr_conf = 0;
    if (demand) {
        sr_conf = update_streamrun(PC, paddr);
    }

    if (hit) {
        // Demand hits update hit counter; first hit does not promote
        if (demand) {
            if (HITCNT[set][way] < HITS_TO_PROMOTE) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // On 2nd+ demand hit, clear sentinel and promote to MRU
                STATUS[set][way] = ST_NORMAL;
                age_all(set);
                AGE[set][way] = 0;
            }
        }
        // Prefetch hits (type==PREFETCH) do not promote; no recency boost
        return;
    }

    // Miss: train PhaseScore on the victim being replaced if valid and sampled
    if (LVALID[set][way] && is_sampled_set(set)) {
        uint32_t vidx = pscore_index_from_pcisg(PCSIG[set][way], set);
        if (HITCNT[set][way] >= HITS_TO_PROMOTE) sat_inc(PSCORE[vidx], 3);
        else                                     sat_dec(PSCORE[vidx]);
    }

    // Prepare new line metadata
    uint8_t pc_sig = static_cast<uint8_t>(pc_index(PC));
    uint32_t pidx  = pscore_index_from_pcisg(pc_sig, set);

    // Base insertion policy from PhaseScore
    uint8_t ins_age = (PSCORE[pidx] >= PS_WARM_TH) ? AGE_YOUNG : AGE_MAX;
    uint8_t ins_stat = ST_NORMAL;

    // Stream-aware near-bypass (only for demand). Prefetches always quarantined.
    if (type == ACCESS_PREFETCH) {
        ins_age  = AGE_MAX;          // tail
        ins_stat = ST_QUAR;          // quarantine
    } else if (demand && (sr_conf >= STREAM_CONF_TH)) {
        ins_age  = AGE_MAX;          // hard tail
        ins_stat = ST_STREAM;        // stream sentinel (no promote until 2nd demand hit)
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass on WB; insert conservatively at tail but normal status
        ins_age  = AGE_MAX;
        ins_stat = ST_NORMAL;
    }

    // Age set and insert the new line
    age_all(set);
    AGE[set][way]    = ins_age;
    STATUS[set][way] = ins_stat;
    PCSIG[set][way]  = pc_sig;
    HITCNT[set][way] = 0;
    LVALID[set][way] = 1;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}