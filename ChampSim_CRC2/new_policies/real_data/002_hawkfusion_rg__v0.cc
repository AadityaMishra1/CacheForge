#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cmath>

// HawkFusion-RG: Mode A (Hawkeye-like RRIP+PC-friendliness) + Mode B (RunGuard+PC-dead+Multi-hit)
// Safety: never return LLC_WAYS (no true bypass), WRITEBACK ignored, prefetches quarantined at tail in Mode B.
// Selector: leader sets (64 sampled; half A-leaders, half B-leaders) update a global saturating chooser.
// Non-leader sets default to A unless B shows clear advantage.

// ------------------- Configuration -------------------
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16
static constexpr uint32_t TOTAL_LINES = LLC_SETS * LLC_WAYS;

// Types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// Sampled-set geometry (power-of-two sets)
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)

// Leader-dueling selector
static constexpr uint8_t GSEL_MAX   = 15;
static constexpr uint8_t GSEL_START = 0;   // default Hawkeye-like (Mode A)
static constexpr uint8_t GSEL_USE_B = 12;  // need strong advantage to switch to Mode B

// PC index (shared) for predictors/signatures
static constexpr uint8_t  PC_BITS   = 7;                 // 128-entry tables
static constexpr uint32_t PC_SIZE   = (1u << PC_BITS);

// Mode A (Hawkeye-like) RRIP
static constexpr uint8_t  A_RRPV_MAX    = 7;             // 3-bit RRIP
static constexpr uint8_t  A_INSERT_YOUNG= 1;             // for friendly PCs
static constexpr uint8_t  A_INSERT_TAIL = 6;             // for averse PCs
static constexpr uint8_t  A_SHCT_MAX    = 7;             // 3-bit counter
static constexpr uint8_t  A_WARM_TH     = 4;             // >=4 => friendly

// Mode B (RunGuard + PC-dead + Multi-hit)
static constexpr uint8_t  B_AGE_MAX     = 3;             // 2-bit age (0=young, 3=old)
static constexpr uint8_t  B_PROMOTE_HITS= 2;             // promote only on 2nd+ demand hit
static constexpr uint8_t  RG_LINE_LOW   = 12;            // low bits of line#
static constexpr uint8_t  RG_CONF_TH    = 2;             // >=2 => treat as stream

// ------------------- Helpers -------------------
static inline bool is_demand(uint32_t t) { return t == ACCESS_LOAD || t == ACCESS_RFO; }
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix; 7-bit index
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 17);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_low(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << RG_LINE_LOW) - 1));
}
template <typename T> static inline void sat_inc(T& x, T maxv) { if (x < maxv) x++; }
template <typename T> static inline void sat_dec(T& x) { if (x > 0) x--; }

// Sampled-set and leaders (64 sampled sets)
static inline bool SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ------------------- Shared per-line state (packed conceptually) -------------------
// We store minimal metadata per line and conceptually pack fields:
//   pcsig: 7 bits (shared by both modes)
//   a_rrpv: 3 bits
//   a_reuse: 1 bit
//   b_age: 2 bits
//   b_meta: 2 bits (0=cold(0 hits), 1=warm(1 hit), 2=sentinel(stream/quarantine), 3=hot(2+ hits))
static uint8_t LINE_PCSIG[LLC_SETS][LLC_WAYS]; // 7-bit index; stored in 8-bit container
static uint8_t A_RRPV     [LLC_SETS][LLC_WAYS]; // 0..7
static uint8_t A_REUSE    [LLC_SETS][LLC_WAYS]; // 0/1
static uint8_t B_AGE      [LLC_SETS][LLC_WAYS]; // 0..3
static uint8_t B_META     [LLC_SETS][LLC_WAYS]; // 0..3

// ------------------- Global predictors/state -------------------
// PC predictor shared (A: friendliness; B: expected-use)
static uint8_t PC_PRED[PC_SIZE]; // 3-bit 0..7

// RunGuard per-PC stream detector
static uint16_t RG_LAST_LINE[PC_SIZE]; // 12-bit value
static uint8_t  RG_CONF     [PC_SIZE]; // 0..3
static uint8_t  RG_LAST_ABS [PC_SIZE]; // 0/1
static uint8_t  RG_LAST_FWD [PC_SIZE]; // 0/1

// Global selector (leader-dueling)
static uint8_t GSEL = GSEL_START;

// ------------------- Mode B helpers -------------------
static inline void b_age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (B_AGE[set][w] < B_AGE_MAX) B_AGE[set][w]++;
    }
}
static inline uint8_t runguard_update(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_low(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? -delta : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS[idx] && RG_LAST_FWD[idx]) sat_inc(RG_CONF[idx], (uint8_t)3);
        else { if (RG_CONF[idx] == 0) RG_CONF[idx] = 1; RG_LAST_ABS[idx] = 1; RG_LAST_FWD[idx] = 1; }
    } else {
        sat_dec(RG_CONF[idx]);
        RG_LAST_ABS[idx] = 0;
        RG_LAST_FWD[idx] = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}
static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer stream/quarantine sentinels (meta==2)
    bool a_s = (B_META[set][a] == 2), b_s = (B_META[set][b] == 2);
    if (a_s != b_s) return a_s;

    // 2) Prefer colder PC (lower PC_PRED)
    uint8_t pa = PC_PRED[ LINE_PCSIG[set][a] ];
    uint8_t pb = PC_PRED[ LINE_PCSIG[set][b] ];
    if (pa != pb) return (pa < pb);

    // 3) Prefer fewer hits: meta order 0 < 1 < 3
    auto score_meta = [](uint8_t m) -> uint8_t { return (m == 2) ? 0 : ((m == 0) ? 1 : ((m == 1) ? 2 : 3)); };
    uint8_t ma = score_meta(B_META[set][a]);
    uint8_t mb = score_meta(B_META[set][b]);
    if (ma != mb) return (ma < mb);

    // 4) Older age more evictable
    return B_AGE[set][a] > B_AGE[set][b];
}

// ------------------- Mode A helpers -------------------
static inline uint32_t find_victim_A(uint32_t set) {
    // Try to find RRPV==max; if none, age (increment) and retry (bounded)
    for (int iter = 0; iter < (A_RRPV_MAX + 1); iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (A_RRPV[set][w] == A_RRPV_MAX) return w;
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (A_RRPV[set][w] < A_RRPV_MAX) A_RRPV[set][w]++;
    }
    return 0; // fallback
}

static inline void train_pc_on_eviction(uint32_t set, uint32_t way) {
    uint8_t pc = LINE_PCSIG[set][way];
    if (A_REUSE[set][way]) sat_inc(PC_PRED[pc], (uint8_t)A_SHCT_MAX);
    else                   sat_dec(PC_PRED[pc]);
}

// ------------------- API: Init -------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            LINE_PCSIG[s][w] = 0;
            A_RRPV[s][w]     = A_RRPV_MAX;
            A_REUSE[s][w]    = 0;
            B_AGE[s][w]      = B_AGE_MAX;
            B_META[s][w]     = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_PRED[i]      = A_WARM_TH - 1; // slightly cold
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS[i]  = 0;
        RG_LAST_FWD[i]  = 0;
    }
    GSEL = GSEL_START;
}

// ------------------- API: Victim selection -------------------
uint32_t GetVictimInSet(uint32_t /*cpu*/, uint32_t set, const BLOCK* current_set,
                        uint64_t /*PC*/, uint64_t /*paddr*/, uint32_t /*type*/) {
    // 1) Invalid way wins
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // 2) Decide policy
    bool useB = LEADER_B(set) || (!LEADER_A(set) && (GSEL >= GSEL_USE_B));

    if (useB) {
        // Composite heuristic
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++)
            if (more_evictable_B(set, w, best)) best = w;
        return best;
    } else {
        return find_victim_A(set);
    }
}

// ------------------- API: State update -------------------
void UpdateReplacementState(uint32_t /*cpu*/, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t /*victim_addr*/,
                            uint32_t type, uint8_t hit) {
    // Ignore writebacks entirely
    if (type == ACCESS_WRITEBACK) return;

    // Demand vs prefetch
    const bool demand = is_demand(type);
    const bool is_pf  = (type == ACCESS_PREFETCH);

    // Update RunGuard on every demand access (before using conf for insertion decisions)
    uint8_t rg_conf = 0;
    if (demand) rg_conf = runguard_update(PC, paddr);

    // Leader-dueling: update global selector on demand misses
    if (!hit && demand) {
        if (LEADER_A(set)) sat_inc(GSEL, GSEL_MAX); // A miss -> tilt toward B
        else if (LEADER_B(set)) sat_dec(GSEL);      // B miss -> tilt toward A
    }

    // Age Mode B on any touch (keeps recency meaningful)
    b_age_all(set);

    // Common PC index and line signature
    uint8_t pcidx = static_cast<uint8_t>(pc_index(PC));

    if (hit) {
        // ---------------------- On Hit ----------------------
        // Mode A: standard RRIP promotion
        A_RRPV[set][way] = 0;
        if (demand) A_REUSE[set][way] = 1; // record reuse only on demand

        // Mode B: strict multi-hit promotion
        if (demand) {
            if (B_META[set][way] == 0) {
                B_META[set][way] = 1;                 // first demand hit
            } else if (B_META[set][way] == 1) {
                B_META[set][way] = 3;                 // 2nd demand hit: promote
                B_AGE[set][way]  = 0;
                sat_inc(PC_PRED[pcidx], (uint8_t)A_SHCT_MAX); // reinforce expected-use
            } else if (B_META[set][way] == 2) {
                B_META[set][way] = 1;                 // was sentinel: first demand hit, no immediate promote
            } else {
                B_AGE[set][way]  = 0;                 // hot: keep young
            }
        } else {
            // Prefetch hit: do not promote in Mode B (quarantine effect)
            // Keep B_META unchanged; slight youth if already hot
            if (B_META[set][way] == 3) B_AGE[set][way] = 0;
        }
        return;
    }

    // ---------------------- On Miss/Fill ----------------------
    // Train PC predictor on the evicted line's reuse (read old state before overwrite)
    train_pc_on_eviction(set, way);

    // Update shared signature for the new line
    LINE_PCSIG[set][way] = pcidx;

    // Mode A insertion
    if (is_pf) {
        A_RRPV[set][way]  = A_RRPV_MAX; // deep tail for prefetch
    } else {
        bool warm = (PC_PRED[pcidx] >= A_WARM_TH);
        A_RRPV[set][way]  = warm ? A_INSERT_YOUNG : A_INSERT_TAIL;
    }
    A_REUSE[set][way] = 0;

    // Mode B insertion
    if (is_pf) {
        B_META[set][way] = 2;                 // quarantine prefetch
        B_AGE[set][way]  = B_AGE_MAX;         // tail
    } else if (rg_conf >= RG_CONF_TH) {
        B_META[set][way] = 2;                 // stream sentinel; near-bypass
        B_AGE[set][way]  = B_AGE_MAX;         // tail
    } else {
        bool warm = (PC_PRED[pcidx] >= A_WARM_TH);
        B_META[set][way] = 0;                 // cold until it proves a hit
        B_AGE[set][way]  = warm ? 1 : B_AGE_MAX;
    }
}

// ------------------- API: Stats -------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}