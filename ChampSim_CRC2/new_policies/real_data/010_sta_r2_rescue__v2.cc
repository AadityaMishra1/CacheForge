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
static constexpr uint8_t  PC_BITS            = 8;     // 256-entry PC tables
static constexpr uint32_t PC_SIZE            = (1u << PC_BITS);

// Ticket predictor (2-bit: 0..3)
static constexpr uint8_t  TICKET_INIT        = 1;
static constexpr uint8_t  TICKET_WARM_TH     = 2;     // >=2 => warm PC

// Run detector (per-PC ±1/±2 forward steps)
static constexpr uint8_t  LINE_LOW_BITS      = 12;    // low bits of line# for stride check
static constexpr uint8_t  STREAM_CONF_TH     = 2;     // >=2 => stream near-bypass

// RRPV-like ages: 0=young(MRU)..3=old(LRU)
static constexpr uint8_t  AGE_MAX            = 3;
static constexpr uint8_t  AGE_YOUNG          = 1;     // insertion depth for warm PCs

// Multi-hit promotion thresholds
static constexpr uint8_t  HITS_WARM_PROMOTE  = 2;     // warm PCs promote on 2nd demand hit
static constexpr uint8_t  HITS_COLD_PROMOTE  = 3;     // cold PCs promote on 3rd demand hit
static constexpr uint8_t  HITS_PREF_PROMOTE  = 2;     // prefetch quarantine needs 2 demand hits

// Sampled-set density for ticket training
static constexpr uint8_t  SAMP_LOG2          = 6;     // 1 out of 64 sets

// ---------------- Per-line state (compact) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2b: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2b: 0..3 (counts demand hits; 3=3+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2b: 0=normal,1=stream,2=quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8b: PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1b: resident-valid for training

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global per-PC state ----------------
static uint8_t  TICKET[PC_SIZE];          // 2b: 0..3 (ticketed deadness)
static uint16_t RG_LAST_LINE[PC_SIZE];    // low LINE_LOW_BITS of line#
static uint8_t  RG_CONF[PC_SIZE];         // 2b confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];   // 1b: last step |delta| in {1,2}
static uint8_t  RG_LAST_FWD[PC_SIZE];     // 1b: last step forward

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
static inline void sat_inc_u2(uint8_t &v) { if (v < 3) v++; }
static inline void sat_dec_u2(uint8_t &v) { if (v > 0) v--; }
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
            sat_inc_u2(RG_CONF[idx]);            // two consecutive forward ±1/±2 steps
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u2(RG_CONF[idx]);                // decay on break/backward/large stride
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool sa = (STATUS[set][a] != ST_NORMAL);
    bool sb = (STATUS[set][b] != ST_NORMAL);
    if (sa != sb) return sa; // evict stream/quarantine sentinels first

    uint8_t ta = TICKET[ PCSIG[set][a] ];
    uint8_t tb = TICKET[ PCSIG[set][b] ];
    if (ta != tb) return (ta < tb); // evict colder-PC lines

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // evict fewer-hit lines

    return AGE[set][a] > AGE[set][b]; // older is more evictable
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
    bool demand = is_demand(type);
    uint32_t pcidx = pc_index(PC);

    // Gentle global demotion to stabilize recency
    age_all(set);

    if (hit) {
        // On hits: strict multi-hit promotion; count only demand hits
        if (demand) {
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;

            // Determine threshold
            uint8_t th = (STATUS[set][way] == ST_QUAR) ? HITS_PREF_PROMOTE
                           : ((TICKET[pcidx] >= TICKET_WARM_TH) ? HITS_WARM_PROMOTE : HITS_COLD_PROMOTE);

            if (HITCNT[set][way] >= th) {
                // Rescue from stream/quarantine and promote
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way] = 0;
                // Reward PC on confirmed reuse
                sat_inc_u2(TICKET[pcidx]);
            } else {
                // First demand hit: do not fully promote, but give slight recency nudge
                if (AGE[set][way] > 0) AGE[set][way]--;
            }
            // Update run detector on every demand access
            update_runconf(PC, paddr);
        } else {
            // Prefetch hit/access: no promotion
            if (AGE[set][way] > 0) AGE[set][way]--; // mild touch
        }
        return;
    }

    // Miss path: train on eviction (previous occupant in 'way') if valid and sampled
    if (LVALID[set][way] && is_sampled_set(set)) {
        uint8_t old_pc = PCSIG[set][way];
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_st = STATUS[set][way];

        if (old_hits == 0) {
            // Single-use (dead); penalize slightly; extra penalty if stream/quarantine
            sat_dec_u2(TICKET[old_pc]);
            if (old_st != ST_NORMAL) sat_dec_u2(TICKET[old_pc]);
        } else if (old_hits >= 2) {
            // Multi-hit; reward
            sat_inc_u2(TICKET[old_pc]);
        }
    }

    // Decide insertion for the new line
    uint8_t ins_status = ST_NORMAL;
    uint8_t ins_age    = AGE_MAX;
    uint8_t warm       = (TICKET[pcidx] >= TICKET_WARM_TH) ? 1 : 0;

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine at tail
        ins_status = ST_QUAR;
        ins_age = AGE_MAX;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass WB; insert based on PC ticket
        ins_status = ST_NORMAL;
        ins_age = warm ? AGE_YOUNG : AGE_MAX;
    } else {
        // Demand (LOAD/RFO)
        uint8_t conf = update_runconf(PC, paddr); // detect forward ±1/±2 runs
        if (conf >= STREAM_CONF_TH) {
            // Stream near-bypass: hard-tail with stream sentinel
            ins_status = ST_STREAM;
            ins_age = AGE_MAX;
        } else {
            // Non-stream: ticketed insertion
            ins_status = ST_NORMAL;
            ins_age = warm ? AGE_YOUNG : AGE_MAX;
        }
    }

    // Install new metadata
    STATUS[set][way] = ins_status;
    AGE[set][way]    = ins_age;
    HITCNT[set][way] = 0;             // demand hits count
    PCSIG[set][way]  = static_cast<uint8_t>(pcidx);
    LVALID[set][way] = 1;
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}