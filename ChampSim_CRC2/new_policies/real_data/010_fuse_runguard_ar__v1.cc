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
// PC-indexed tables
static constexpr uint8_t  PC_BITS        = 8;               // 256-entry tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);

// Expected-Use PC predictor (EUP): 2-bit (0..3)
static constexpr uint8_t  EUP_INIT       = 1;               // slightly cold start
static constexpr uint8_t  EUP_WARM_TH    = 2;               // >=2 => warm

// RunGuard: per-PC ±1/±2 forward-run detector
static constexpr uint8_t  LINE_LOW_BITS  = 12;              // compare low bits of line#
static constexpr uint8_t  RG_BYPASS_TH   = 2;               // >=2 => stream near-bypass

// Insertion depth (age: 0=MRU youngest .. 3=oldest)
static constexpr uint8_t  AGE_MAX        = 3;               // tail
static constexpr uint8_t  AGE_YOUNG      = 1;               // slightly young for warm PCs

// Multi-hit promotion (only on demand hits)
static constexpr uint8_t  HITS_TO_PROMOTE = 2;              // promote on 2nd+ demand hit

// Status: 0=normal, 1=stream-guard, 2=quarantine(prefetch/WB)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state (compact) ----------------
static uint8_t AGE   [LLC_SETS][LLC_WAYS];   // 2 bits (0..3)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];   // 0,1,2(=2+ demand hits)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];   // ST_*
static uint8_t PCSIG [LLC_SETS][LLC_WAYS];   // 8-bit PC index
static uint8_t VALID [LLC_SETS][LLC_WAYS];   // 0/1: line ever valid in our metadata

// ---------------- Global predictors ----------------
// EUP: 2-bit counters per PC
static uint8_t EUP[PC_SIZE];

// RunGuard per-PC state
static uint16_t RG_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];        // 2-bit (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];  // 1-bit
static uint8_t  RG_LAST_FWD[PC_SIZE];    // 1-bit (last step was forward)

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update RunGuard on demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3);      // two consecutive forward steps: strengthen stream
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);             // decay on break/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting sentinels (stream or quarantined)
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer lines from colder PCs (lower EUP)
    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb);

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
            VALID[s][w]  = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]          = EUP_INIT;
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS12[i]= 0;
        RG_LAST_FWD[i]  = 0;
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
    // Demote everyone else slightly every access to favor quick recycling of tails
    age_all(set);

    const bool demand = is_demand(type);
    if (demand) {
        // Update RunGuard on every demand reference (hit or miss)
        update_runguard(PC, paddr);
    }

    uint32_t pc_idx = pc_index(PC);

    if (hit) {
        // On hit: only demand hits contribute to reuse and promotion
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Escape-on-reuse: promote and clear any sentinel
                AGE[set][way] = 0; // MRU
                if (STATUS[set][way] != ST_NORMAL) STATUS[set][way] = ST_NORMAL;
                // Reinforce EUP on proven reuse
                sat_inc(EUP[ PCSIG[set][way] ], 3);
            }
            // First demand hit does not promote; quarantine/stream sentinels remain in place
        }
        // Prefetch-hit or non-demand hit: do not count or promote
        return;
    }

    // Miss fill or writeback fill:
    // Train EUP on the evicted line (old contents of this way), if it was valid
    if (VALID[set][way]) {
        uint8_t old_pc = PCSIG[set][way];
        uint8_t reuse  = HITCNT[set][way]; // demand hits only
        if (reuse >= HITS_TO_PROMOTE) sat_inc(EUP[old_pc], 3);
        else                          sat_dec(EUP[old_pc]);
    }

    // Record new line signature and mark valid
    PCSIG[set][way]  = pc_idx;
    VALID[set][way]  = 1;
    HITCNT[set][way] = 0;

    // Insertion policy
    if (type == ACCESS_PREFETCH) {
        // Quarantine prefetches at the hard tail; never promote on first hit
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    if (type == ACCESS_WRITEBACK) {
        // No bypass allowed on writeback; keep at tail and deprioritize
        STATUS[set][way] = ST_QUAR;
        AGE[set][way]    = AGE_MAX;
        return;
    }

    // Demand fill: combine RunGuard (stream) and EUP (PC warmness)
    uint8_t rgc = RG_CONF[pc_idx];
    bool streamy = (rgc >= RG_BYPASS_TH);

    if (streamy) {
        // Near-bypass: hard tail insert, no promotion until 2nd demand hit
        STATUS[set][way] = ST_STREAM;
        AGE[set][way]    = AGE_MAX;
    } else {
        // PC-guided insertion depth
        bool warm = (EUP[pc_idx] >= EUP_WARM_TH);
        STATUS[set][way] = ST_NORMAL;
        AGE[set][way]    = warm ? AGE_YOUNG : AGE_MAX;
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