#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

// HawkFusion-StreamShield (Mode A + Mode B) with per-set epoch selector
// CRC2 constraints: no bypass return, no WB bypass, safe counters, low storage

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  AGE_MAX          = 3;    // shared 2-bit age (RRIP-like)
static constexpr uint8_t  PC_BITS          = 8;    // shared per-line signature + predictors
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);

// StreamShield (Mode B)
static constexpr uint8_t  EUP_WARM_TH      = 2;    // 2-bit (0..3): >=2 => warm PC
static constexpr uint8_t  RG_BYPASS_TH     = 2;    // RunGuard confidence >=2 => stream near-bypass
static constexpr uint8_t  HITS_TO_PROMOTE  = 2;    // promote to MRU only on 2nd+ demand hit

// Hawkeye-like (Mode A)
static constexpr uint8_t  HK_CTR_MAX       = 31;   // 5-bit counters
static constexpr uint8_t  HK_WARM_TH       = 16;   // >=16 => friendly insert young

// Selector
static constexpr uint8_t  EPOCH_FILLS      = 64;   // per-set epoch length in fills
static constexpr uint8_t  SWITCH_MARGIN    = 1;    // require B > A + margin

// ---------------- Per-line metadata (bit-packed conceptually: 14 bits/line) ----------------
// age: 2b, hitcnt: 2b (0,1,2=2+), status: 2b (0=normal,1=stream,2=quarantine), pcsig: 8b
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

static uint8_t AGE    [LLC_SETS][LLC_WAYS]; // 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS]; // 0..2 (2 means 2+)
static uint8_t STATUS [LLC_SETS][LLC_WAYS]; // ST_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS]; // 8-bit signature

// ---------------- Mode B global predictors ----------------
static uint8_t EUP[PC_SIZE];                // 2-bit expected-use per PC (0..3)

// RunGuard per-PC run detector: last_line(8b), conf(2b), last_abs12(1b), last_fwd(1b)
static uint8_t RG_LAST_LINE[PC_SIZE];
static uint8_t RG_CONF     [PC_SIZE];       // 0..3
static uint8_t RG_LAST_ABS12[PC_SIZE];      // 0/1
static uint8_t RG_LAST_FWD [PC_SIZE];       // 0/1

// ---------------- Mode A Hawkeye-like predictors (shared 8-bit signature) ----------------
static uint8_t HK_DEMAND[PC_SIZE];          // 5-bit (store in 8-bit), 0..31
static uint8_t HK_PREFCH[PC_SIZE];          // 5-bit (store in 8-bit), 0..31

// ---------------- Per-set epoch selector (default Hawkeye) ----------------
static uint8_t SET_MODE   [LLC_SETS];       // 0=Hawkeye-like (A), 1=StreamShield (B)
static uint8_t SET_SCOREA [LLC_SETS];       // delivered hits while mode A active
static uint8_t SET_SCOREB [LLC_SETS];       // delivered hits while mode B active
static uint8_t SET_EPOCH  [LLC_SETS];       // fills seen this epoch
static uint8_t LAST_VICTIM_WAS_INVALID[LLC_SETS]; // flag from GetVictimInSet

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return t == ACCESS_LOAD || t == ACCESS_RFO; }
static inline uint32_t pc_index(uint64_t pc) {
    // 8-bit stable hash
    uint64_t x = pc ^ (pc >> 17) ^ (pc >> 5);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint8_t line_index8(uint64_t paddr) {
    // 8 low bits of line number for RunGuard
    return static_cast<uint8_t>((paddr >> 6) & 0xFFu);
}
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc2(uint8_t &x) { if (x < 2) x++; }

// Update RunGuard on demand access
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    const uint32_t idx = pc_index(pc);
    const uint8_t curr = line_index8(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int16_t>(curr) - static_cast<int16_t>(RG_LAST_LINE[idx]));
    uint16_t ad = static_cast<uint16_t>(delta < 0 ? -delta : delta);
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc_u8(RG_CONF[idx], 3);
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RG_CONF[idx]);
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// StreamShield victim preference
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict quarantined/stream-tagged first

    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // evict colder PC

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // evict fewer-use

    return AGE[set][a] > AGE[set][b]; // tie-break by age
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w] = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w] = 0;
        }
        SET_MODE[s] = 0; // default: Hawkeye-like
        SET_SCOREA[s] = 0;
        SET_SCOREB[s] = 0;
        SET_EPOCH[s]  = 0;
        LAST_VICTIM_WAS_INVALID[s] = 0;
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i] = 1; // slightly cold
        RG_LAST_LINE[i] = 0;
        RG_CONF[i] = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
        HK_DEMAND[i] = 8;  // cool start
        HK_PREFCH[i] = 4;
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
    // Return the first invalid way if any
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            LAST_VICTIM_WAS_INVALID[set] = 1;
            return w;
        }
    }
    LAST_VICTIM_WAS_INVALID[set] = 0;

    if (SET_MODE[set] == 0) {
        // Mode A: Hawkeye-like RRIP (shared 2-bit age)
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (AGE[set][w] > AGE[set][best]) best = w;
        }
        return best;
    } else {
        // Mode B: StreamShield preference
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable(set, w, best)) best = w;
        }
        return best;
    }
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
    if (type == ACCESS_WRITEBACK) return;

    const bool demand = is_demand(type);
    const uint32_t pcs = pc_index(PC);

    // Demand access updates RunGuard
    uint8_t rg_conf = 0;
    if (demand) {
        rg_conf = update_runguard(PC, paddr);
    }

    // Selector accounting on hits (attribute to current active mode)
    if (hit) {
        if (SET_MODE[set] == 0) {
            sat_inc_u8(SET_SCOREA[set], 255);
        } else {
            sat_inc_u8(SET_SCOREB[set], 255);
        }
    }

    // On a hit: update per-line state
    if (hit) {
        // Mode B multi-hit gate
        if (demand) {
            sat_inc2(HITCNT[set][way]);
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Promote on 2nd+ demand hit (except for ST_STREAM lines)
                if (STATUS[set][way] != ST_STREAM) {
                    AGE[set][way] = 0;
                }
                // EUP positive reinforcement
                sat_inc_u8(EUP[ PCSIG[set][way] ], 3);
                // Hawkeye-like reinforcement
                sat_inc_u8(HK_DEMAND[ PCSIG[set][way] ], HK_CTR_MAX);
            }
        }
        // No promotion on first hit or on prefetch hits
        // Mode A: also bring friendly to MRU on hits (recency)
        if (SET_MODE[set] == 0 && demand) {
            AGE[set][way] = 0;
        }
        return;
    }

    // Miss/fill path
    // Train predictors on the evicted line if a valid victim was replaced
    if (!LAST_VICTIM_WAS_INVALID[set]) {
        const uint8_t prior_sig  = PCSIG[set][way];
        const uint8_t prior_hits = HITCNT[set][way];
        const uint8_t prior_stat = STATUS[set][way];

        if (prior_hits >= 2) {
            sat_inc_u8(EUP[prior_sig], 3);
            sat_inc_u8(HK_DEMAND[prior_sig], HK_CTR_MAX);
            if (prior_stat == ST_QUAR) sat_inc_u8(HK_PREFCH[prior_sig], HK_CTR_MAX);
        } else {
            sat_dec_u8(EUP[prior_sig]);
            if (HK_DEMAND[prior_sig] > 0) HK_DEMAND[prior_sig]--;
            if (prior_stat == ST_QUAR && HK_PREFCH[prior_sig] > 0) HK_PREFCH[prior_sig]--;
        }
    }

    // Epoch-based selector update
    if (SET_EPOCH[set] < 0xFF) SET_EPOCH[set]++;
    if (SET_EPOCH[set] >= EPOCH_FILLS) {
        // Default to Hawkeye unless B is clearly ahead
        if (SET_SCOREB[set] > static_cast<uint8_t>(SET_SCOREA[set] + SWITCH_MARGIN))
            SET_MODE[set] = 1;
        else
            SET_MODE[set] = 0;

        SET_EPOCH[set] = 0;
        SET_SCOREA[set] = 0;
        SET_SCOREB[set] = 0;
    }

    // Insert new line
    PCSIG[set][way]  = pcs;
    HITCNT[set][way] = 0;

    if (type == ACCESS_PREFETCH) {
        STATUS[set][way] = ST_QUAR;   // quarantine prefetches
        AGE[set][way]    = AGE_MAX;   // hard tail
        // light prefetch predictor training towards cold on fill
        if (HK_PREFCH[pcs] > 0) HK_PREFCH[pcs]--;
        return;
    }

    // Demand fill
    if (SET_MODE[set] == 1) {
        // Mode B: StreamShield
        if (rg_conf >= RG_BYPASS_TH) {
            // Stream near-bypass: deepest age and never promote
            STATUS[set][way] = ST_STREAM;
            AGE[set][way]    = AGE_MAX;
        } else {
            STATUS[set][way] = ST_NORMAL;
            // PC Expected-Use guided insertion
            AGE[set][way] = (EUP[pcs] >= EUP_WARM_TH) ? 1 : AGE_MAX;
        }
    } else {
        // Mode A: Hawkeye-like
        STATUS[set][way] = ST_NORMAL;
        AGE[set][way] = (HK_DEMAND[pcs] >= HK_WARM_TH) ? 0 : AGE_MAX;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}