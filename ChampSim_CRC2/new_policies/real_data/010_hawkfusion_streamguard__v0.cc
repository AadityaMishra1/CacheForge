#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access kinds (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables (adjust carefully) ----------------
static constexpr uint8_t  PC_BITS        = 7;               // 128-entry PC tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);

static constexpr uint8_t  EUP_BITS       = 2;               // 0..3
static constexpr uint8_t  EUP_INIT       = 1;               // slightly cold
static constexpr uint8_t  EUP_WARM_TH    = 2;               // >=2 => warm

static constexpr uint8_t  LINE_LOW_BITS  = 12;              // for run detector
static constexpr uint8_t  RG_BYPASS_TH   = 2;               // >=2 => stream insert at tail

// Insertion ages (0=MRU/young ... 3=oldest)
static constexpr uint8_t  AGE_MAX        = 3;
static constexpr uint8_t  AGE_YOUNG      = 1;

// Multi-hit promotion (Mode B)
static constexpr uint8_t  HITS_TO_PROMOTE = 2;              // promote only on 2nd+ demand hit

// Selector epoching
static constexpr uint8_t  EPOCH_LEN      = 32;              // fills per-set per-epoch
static constexpr uint8_t  CNT_MAX5       = 31;              // 5-bit counters
static constexpr uint8_t  EXPLORE_MASK   = 0x7;             // 1/8 exploration when not preferring B
static constexpr uint8_t  PREF_MARGIN    = 0;               // tie -> stick with A

// ---------------- Per-line state (compact) ----------------
// 2b age, 2b hitcnt, 2b status, 7b pcsig, 1b modesrc
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

static uint8_t AGE     [LLC_SETS][LLC_WAYS];
static uint8_t HITCNT  [LLC_SETS][LLC_WAYS];   // demand-hit count (sat at 3)
static uint8_t STATUS  [LLC_SETS][LLC_WAYS];   // ST_*
static uint8_t PCSIG   [LLC_SETS][LLC_WAYS];   // 7-bit PC index
static uint8_t MODESRC [LLC_SETS][LLC_WAYS];   // 0=A(Hawkeye-like), 1=B(StreamGuard)

// ---------------- Global predictors (tiny) ----------------
static uint8_t  EUP[PC_SIZE];                  // 2-bit expected-use predictor per PC

// RunGuard per-PC (stream run detector)
static uint16_t RG_LAST_LINE[PC_SIZE];         // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];              // 2-bit conf (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];        // 1-bit: last step was ±1/±2
static uint8_t  RG_LAST_FWD[PC_SIZE];          // 1-bit: last step was forward

// ---------------- Per-set selector state ----------------
static uint8_t SEL_PREF_B [LLC_SETS];          // 0/1 prefer Mode B
static uint8_t SEL_EPOCH  [LLC_SETS];          // 5-bit epoch fill counter (0..31)
static uint8_t A_FILL     [LLC_SETS];          // 5-bit fills attributed to A
static uint8_t A_HIT      [LLC_SETS];          // 5-bit hits attributed to A
static uint8_t B_FILL     [LLC_SETS];          // 5-bit fills attributed to B
static uint8_t B_HIT      [LLC_SETS];          // 5-bit hits attributed to B

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // simple mix then mask
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23) ^ (pc << 7);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }

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
            sat_inc_u8(RG_CONF[idx], 3);
        } else {
            // mild boost to detect runs
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RG_CONF[idx]);    // decay on irregular/backwards/large stride
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    if (a == b) return false;

    // 1) Stream/quarantine sentinels first
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Colder PCs first (lower EUP)
    uint8_t ea = EUP[ PCS[set][a] ]; // placeholder; fixed below
    return false;
}

// Corrected comparator (expanded to avoid placeholder above)
static inline bool more_evictable_ex(uint32_t set, uint32_t a, uint32_t b) {
    if (a == b) return false;

    // 1) Stream/quarantine sentinels first
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Colder PCs first (lower EUP)
    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb);

    // 3) Fewer demand hits first
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Older age first
    if (AGE[set][a] != AGE[set][b]) return (AGE[set][a] > AGE[set][b]);

    // 5) Tie-breaker: higher way index
    return a > b;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]     = AGE_MAX;
            HITCNT[s][w]  = 0;
            STATUS[s][w]  = ST_NORMAL;
            PCSIG[s][w]   = 0;
            MODESRC[s][w] = 0; // start as Mode A
        }
        SEL_PREF_B[s] = 0;
        SEL_EPOCH[s]  = 0;
        A_FILL[s] = A_HIT[s] = 0;
        B_FILL[s] = B_HIT[s] = 0;
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Otherwise, choose most evictable way
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable_ex(set, w, best)) best = w;
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
    // Ignore writebacks (no allocation policy change)
    if (type == ACCESS_WRITEBACK) return;

    const bool demand = is_demand(type);
    const bool prefetch = (type == ACCESS_PREFETCH);
    const uint8_t pcidx = static_cast<uint8_t>(pc_index(PC));

    // RunGuard: update on demand accesses
    if (demand) {
        update_runguard(PC, paddr);
    }

    // Attribution on hits (increment mode's hit counter)
    if (hit) {
        uint8_t &hc = HITCNT[set][way];
        if (demand && hc < 3) hc++;

        // Mode-specific promotion policy
        if (demand) {
            if (MODESRC[set][way]) {
                // Mode B: promote only on 2nd+ demand hit
                if (hc >= HITS_TO_PROMOTE) {
                    AGE[set][way] = 0;
                    if (STATUS[set][way] != ST_NORMAL) STATUS[set][way] = ST_NORMAL; // de-quarantine/de-stream
                    sat_inc_u8(EUP[ PCSIG[set][way] ], 3);
                }
            } else {
                // Mode A: promote on first demand hit (Hawkeye-like recency)
                AGE[set][way] = 0;
                sat_inc_u8(EUP[ PCSIG[set][way] ], 3);
            }
        }
        // Attribute hits to inserting mode
        if (MODESRC[set][way]) {
            sat_inc_u8(B_HIT[set], CNT_MAX5);
        } else {
            sat_inc_u8(A_HIT[set], CNT_MAX5);
        }
        return;
    }

    // Miss and fill path
    // Age everyone a bit to bias towards older victims
    age_all(set);

    // Selector: decide which mode to use for this fill
    bool preferB = (SEL_PREF_B[set] != 0);
    bool exploreB = (!preferB) && ((SEL_EPOCH[set] & EXPLORE_MASK) == 0); // sparse exploration
    bool useB = preferB || exploreB;

    // Increment epoch fill counters and per-mode fill attribution
    if (useB) sat_inc_u8(B_FILL[set], CNT_MAX5);
    else      sat_inc_u8(A_FILL[set], CNT_MAX5);

    // If an epoch completed, evaluate preferences
    sat_inc_u8(SEL_EPOCH[set], EPOCH_LEN); // safe sat_inc; we'll manually wrap
    if (SEL_EPOCH[set] >= EPOCH_LEN) {
        // Compare B's hits-per-fill vs A's hits-per-fill via cross-multiplication; stick to A on ties
        uint32_t a_f = std::max<uint32_t>(1, A_FILL[set]);
        uint32_t b_f = std::max<uint32_t>(1, B_FILL[set]);
        uint32_t a_h = A_HIT[set];
        uint32_t b_h = B_HIT[set];

        uint32_t lhs = b_h * a_f;               // B effectiveness
        uint32_t rhs = a_h * b_f + PREF_MARGIN; // A effectiveness with margin

        SEL_PREF_B[set] = (lhs > rhs) ? 1 : 0;

        // Reset epoch counters
        SEL_EPOCH[set] = 0;
        A_FILL[set] = A_HIT[set] = 0;
        B_FILL[set] = B_HIT[set] = 0;
    }

    // Train EUP on eviction of the old resident (if any) before reinit
    // We don't have validity here; be conservative: train only if line saw at least one demand hit
    uint8_t old_hits = HITCNT[set][way];
    uint8_t old_sig  = PCSIG[set][way];
    if (old_hits >= 2) {
        sat_inc_u8(EUP[old_sig], 3);
    } else if (old_hits == 0 || old_hits == 1) {
        sat_dec_u8(EUP[old_sig]);
    }

    // Decide insertion state for the new line
    STATUS[set][way]  = ST_NORMAL;
    HITCNT[set][way]  = 0;
    PCSIG[set][way]   = pcidx;
    MODESRC[set][way] = useB ? 1 : 0;

    if (useB) {
        // Mode B: StreamGuard-EUP
        bool streamy = (demand && RG_CONF[pcidx] >= RG_BYPASS_TH);
        if (prefetch) {
            STATUS[set][way] = ST_QUAR;  // quarantine prefetches at tail
            AGE[set][way]    = AGE_MAX;
        } else if (streamy) {
            STATUS[set][way] = ST_STREAM; // demand stream -> hard tail
            AGE[set][way]    = AGE_MAX;
        } else {
            // PC coldness guided insertion
            AGE[set][way] = (EUP[pcidx] >= EUP_WARM_TH) ? AGE_YOUNG : AGE_MAX;
        }
    } else {
        // Mode A: Hawkeye-like (friendly PCs insert young, cold insert old)
        if (prefetch) {
            STATUS[set][way] = ST_QUAR;
            AGE[set][way]    = AGE_MAX;
        } else {
            AGE[set][way] = (EUP[pcidx] >= EUP_WARM_TH) ? AGE_YOUNG : AGE_MAX;
        }
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}