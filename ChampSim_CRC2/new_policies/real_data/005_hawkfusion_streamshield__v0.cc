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
// Shared PC signature per-line (8-bit)
static constexpr uint32_t PC_SIG_BITS   = 8;                 // 256 signatures
static constexpr uint32_t PC_SIG_SIZE   = (1u << PC_SIG_BITS);

// Mode A (Hawkeye-like) PC predictor table (5-bit counters)
static constexpr uint32_t A_SHCT_SIZE   = 256;               // indexed by 8-bit PC signature
static constexpr uint8_t  A_SHCT_MAX    = 31;
static constexpr uint8_t  A_SHCT_WARM   = 16;                // >= warm => insert young

// Mode B (StreamShield) Expected-Use per-PC (2-bit)
static constexpr uint8_t  EUP_BITS      = 2;
static constexpr uint32_t EUP_SIZE      = 256;               // indexed by 8-bit PC signature
static constexpr uint8_t  EUP_MAX       = (1u << EUP_BITS) - 1;
static constexpr uint8_t  EUP_WARM_TH   = 2;                 // >=2 => warm

// RunGuard stream detector per PC (using 8-bit PC index)
static constexpr uint8_t  RG_PC_BITS    = 8;                 // 256 entries
static constexpr uint8_t  RG_LINE_BITS  = 12;                // compare low 12 line bits
static constexpr uint8_t  RG_CONF_MAX   = 3;
static constexpr uint8_t  RG_BYPASS_TH  = 2;                 // >=2 => stream near-bypass

// RRIP age (shared 3-bit, 0=MRU, 7=LRU)
static constexpr uint8_t  RRPV_MAX      = 7;
static constexpr uint8_t  RRPV_YOUNG    = 1;                 // slightly young insertion

// Multi-hit promotion (Mode B)
static constexpr uint8_t  HITS_TO_PROMOTE = 2;               // promote on 2nd+ demand hit

// Selector: leader sets and global score threshold
static constexpr int32_t  SCORE_UP_TH    = 8;                // hysteresis for switching
static constexpr int32_t  SCORE_DN_TH    = -8;

// ---------------- Per-line state (shared/minimal) ----------------
// Shared across both modes to bound metadata
static uint8_t RRPV   [LLC_SETS][LLC_WAYS];   // 3-bit logical
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC signature
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 0,1,2(=2+) observed demand hits
// Mode B status: 0=normal, 1=stream-guard, 2=quarantine(prefetch)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };
static uint8_t STATUS [LLC_SETS][LLC_WAYS];

// ---------------- Mode A (Hawkeye-like) global predictor ----------------
static uint8_t A_SHCT[A_SHCT_SIZE]; // 5-bit counters per PC signature

// ---------------- Mode B (StreamShield) global predictors ----------------
static uint8_t EUP[EUP_SIZE]; // 2-bit expected-use (per PC signature)

// RunGuard per PC: last line low bits, confidence, flags[0]=abs12, flags[1]=fwd
static uint16_t RG_LAST_LINE[1u << RG_PC_BITS];
static uint8_t  RG_CONF     [1u << RG_PC_BITS];
static uint8_t  RG_FLAGS    [1u << RG_PC_BITS];  // bit0=abs12_seen, bit1=last_fwd

// ---------------- Selector (set dueling) ----------------
static int32_t  policy_score = 0;             // >0 favors Mode B, <0 favors Mode A
static uint8_t  follower_mode[LLC_SETS];      // 0=A, 1=B (latched per epoch)

// bit helpers
static inline uint64_t bitmask_u64(uint32_t l) { return (l == 64) ? ~0ull : ((1ull << l) - 1ull); }
static inline uint64_t bits_u64(uint64_t x, uint32_t i, uint32_t l) { return (x >> i) & bitmask_u64(l); }

// leader-set mapping: 64 A-leaders (low 6 bits == 0), 64 B-leaders (next 6 bits == 0), disjoint
static inline bool LEADER_A(uint32_t set) { return bits_u64(set, 0, 6) == 0; }
static inline bool LEADER_B(uint32_t set) { return (!LEADER_A(set)) && (bits_u64(set, 6, 6) == 0); }

// helpers
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_sig(uint64_t pc) {
    // simple folding hash to 8 bits
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 17);
    return static_cast<uint32_t>(x) & (PC_SIG_SIZE - 1);
}
static inline uint32_t rg_pc_idx(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 21);
    return static_cast<uint32_t>(x) & ((1u << RG_PC_BITS) - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << RG_LINE_BITS) - 1));
}
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }

// Update RunGuard on demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = rg_pc_idx(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        bool prev_abs12 = (RG_FLAGS[idx] & 0x1) != 0;
        bool prev_fwd   = (RG_FLAGS[idx] & 0x2) != 0;
        if (prev_abs12 && prev_fwd) {
            sat_inc_u8(RG_CONF[idx], RG_CONF_MAX);
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_FLAGS[idx] = 0x3; // abs12=1, fwd=1
        }
    } else {
        sat_dec_u8(RG_CONF[idx]); // decay on irregular/backwards/large stride
        RG_FLAGS[idx] = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// -------- Mode A (Hawkeye-like) victim selection --------
static inline uint32_t victim_modeA(uint32_t set) {
    // Look for max RRPV line first
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (RRPV[set][w] == RRPV_MAX) return w;

    // Else choose the line with largest RRPV (ties by highest value found later)
    uint8_t maxv = 0;
    uint32_t vic = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (RRPV[set][w] >= maxv) {
            maxv = RRPV[set][w];
            vic = w;
        }
    }
    return vic;
}

// -------- Mode B (StreamShield) victim selection --------
static inline bool b_more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict sentinels first

    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // evict colder PCs

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // evict fewer-hit lines

    // Tie-break by age: older (higher RRPV) first
    return RRPV[set][a] >= RRPV[set][b];
}
static inline uint32_t victim_modeB(uint32_t set) {
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (b_more_evictable(set, w, best)) best = w;
    }
    return best;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]   = RRPV_MAX;
            PCSIG[s][w]  = 0;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
        }
        follower_mode[s] = 0; // default to Mode A
    }
    for (uint32_t i = 0; i < A_SHCT_SIZE; i++) A_SHCT[i] = A_SHCT_WARM; // neutral-warm
    for (uint32_t i = 0; i < EUP_SIZE; i++)    EUP[i]    = 1;           // slightly cold start
    for (uint32_t i = 0; i < (1u << RG_PC_BITS); i++) {
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_FLAGS[i]     = 0;
    }
    policy_score = 0;
}

// Select victim
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

    // Choose active mode for this set
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    uint8_t mode = follower_mode[set];
    if (leaderA) mode = 0;
    else if (leaderB) mode = 1;
    else {
        // Global preference with hysteresis
        if (policy_score > SCORE_UP_TH) mode = 1;
        else if (policy_score < SCORE_DN_TH) mode = 0;
        // otherwise keep previous follower_mode[set]
    }
    follower_mode[set] = mode;

    return (mode == 0) ? victim_modeA(set) : victim_modeB(set);
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
    // Ignore writebacks (no bypass or training)
    if (type == ACCESS_WRITEBACK) return;

    const bool demand = is_demand(type);
    const uint32_t sig = pc_sig(PC);

    // Selector accounting on leader sets for demand hits only
    if (hit && demand) {
        if (LEADER_B(set)) policy_score++;
        else if (LEADER_A(set)) policy_score--;
    }

    // RunGuard update only on demand references
    uint8_t rg_conf = 0;
    if (demand) rg_conf = update_runguard(PC, paddr);

    // On hit: update both modes' shared state
    if (hit) {
        // Mode A: immediate promotion on any hit (Hawkeye-like)
        RRPV[set][way] = 0; // MRU

        // Mode B: multi-hit gate, and never promote on prefetch-only hit
        if (type != ACCESS_PREFETCH) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Confirmed reuse: clear stream/quarantine and mark MRU
                STATUS[set][way] = ST_NORMAL;
                RRPV[set][way] = 0;
                // Reinforce EUP for the producer PC
                if (EUP[sig] < EUP_MAX) EUP[sig]++;
            }
        }
        return;
    }

    // Miss path: we are filling 'way' with a new line; train on the evicted content first
    const uint8_t old_sig   = PCSIG[set][way];
    const uint8_t old_hits  = HITCNT[set][way];

    // Train Mode A SHCT with old line's outcome
    if (old_hits >= 2) { if (A_SHCT[old_sig] < A_SHCT_MAX) A_SHCT[old_sig]++; }
    else { if (A_SHCT[old_sig] > 0) A_SHCT[old_sig]--; }

    // Train Mode B EUP with old line's outcome
    if (old_hits >= 2) { if (EUP[old_sig] < EUP_MAX) EUP[old_sig]++; }
    else { if (EUP[old_sig] > 0) EUP[old_sig]--; }

    // Reset new line metadata
    PCSIG[set][way]  = static_cast<uint8_t>(sig);
    HITCNT[set][way] = 0;

    // Determine active mode for insertion in this set now (same as in victim selection)
    uint8_t mode = follower_mode[set];
    if (LEADER_A(set)) mode = 0;
    else if (LEADER_B(set)) mode = 1;
    else {
        if (policy_score > SCORE_UP_TH) mode = 1;
        else if (policy_score < SCORE_DN_TH) mode = 0;
    }
    follower_mode[set] = mode;

    // Insertion policy per mode
    if (mode == 0) {
        // Mode A (Hawkeye-like): friendly PCs young, averse old; prefetch quarantined
        if (type == ACCESS_PREFETCH) {
            STATUS[set][way] = ST_QUAR;
            RRPV[set][way] = RRPV_MAX; // tail
        } else {
            STATUS[set][way] = ST_NORMAL;
            if (A_SHCT[sig] >= A_SHCT_WARM) RRPV[set][way] = RRPV_YOUNG;
            else                            RRPV[set][way] = RRPV_MAX;
        }
    } else {
        // Mode B (StreamShield): stream guard, EUP, multi-hit gate, prefetch quarantine
        if (type == ACCESS_PREFETCH) {
            STATUS[set][way] = ST_QUAR;            // quarantine
            RRPV[set][way]   = RRPV_MAX;           // tail insert
        } else if (demand && rg_conf >= RG_BYPASS_TH) {
            STATUS[set][way] = ST_STREAM;          // stream-guard: near-bypass
            RRPV[set][way]   = RRPV_MAX;           // hard tail, never promote until 2nd demand hit
        } else {
            STATUS[set][way] = ST_NORMAL;
            // PC expected-use: warm PCs slightly young, cold PCs old
            if (EUP[sig] >= EUP_WARM_TH) RRPV[set][way] = RRPV_YOUNG;
            else                         RRPV[set][way] = RRPV_MAX;
        }
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}