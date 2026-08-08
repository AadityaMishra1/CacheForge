#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_SIG_BITS     = 8;                 // 256 PC signatures
static constexpr uint32_t PC_SIG_SIZE     = (1u << PC_SIG_BITS);

static constexpr uint8_t  RRPV_MAX        = 7;                 // 3-bit RRIP
static constexpr uint8_t  RRPV_YOUNG_A    = 1;                 // Mode A young insertion
static constexpr uint8_t  RRPV_YOUNG_B    = 2;                 // Mode B slightly older young

// Hawkeye-like PC-friendly SHiP (5-bit)
static constexpr uint32_t A_SHCT_SIZE     = 256;
static constexpr uint8_t  A_SHCT_MAX      = 31;
static constexpr uint8_t  A_SHCT_WARM     = 16;                // >= warm => friendly

// Expected-use (EUP) auxiliary (2-bit)
static constexpr uint32_t EUP_SIZE        = 256;
static constexpr uint8_t  EUP_MAX         = 3;

// RunGuard v2 per-PC (stride ±1/±2 forward run)
static constexpr uint8_t  RG_PC_BITS      = 8;                 // 256 entries
static constexpr uint8_t  RG_LINE_BITS    = 12;                // compare 64B-aligned line low bits
static constexpr uint8_t  RG_CONF_MAX     = 3;                 // 2+ => stream detected
static constexpr uint8_t  RG_BYPASS_TH    = 2;

// Multi-hit promotion gates
static constexpr uint8_t  HITS_TO_PROMOTE_DEMAND   = 2;        // promote on 2nd+ demand hit
static constexpr uint8_t  HITS_TO_PROMOTE_PREF_ORG = 3;        // prefetch-origin needs 3 demand hits

// Selector hysteresis (doubled vs prior)
static constexpr int32_t  SCORE_UP_TH     = 16;
static constexpr int32_t  SCORE_DN_TH     = -16;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline uint64_t bitmask_u64(uint32_t l) { return (l == 64) ? ~0ull : ((1ull << l) - 1ull); }
static inline uint64_t bits_u64(uint64_t x, uint32_t i, uint32_t l) { return (x >> i) & bitmask_u64(l); }
static inline uint32_t pc_sig(uint64_t pc) {
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

// Leader-set mapping: 64 A-leaders (low 5 bits == 0), 64 B-leaders (next 5 bits == 0, and not A)
static inline bool LEADER_A(uint32_t set) { return bits_u64(set, 0, 5) == 0; }
static inline bool LEADER_B(uint32_t set) { return (!LEADER_A(set)) && (bits_u64(set, 5, 5) == 0); }

// ---------------- Shared per-line state ----------------
static uint8_t RRPV   [LLC_SETS][LLC_WAYS];   // 3-bit logical age
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC signature
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2-bit, 0..3 (3 = 3+)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };
static uint8_t STATUS [LLC_SETS][LLC_WAYS];   // 2-bit

// ---------------- Global predictors ----------------
static uint8_t A_SHCT[A_SHCT_SIZE];           // 5-bit friendliness
static uint8_t EUP  [EUP_SIZE];               // 2-bit expected-use (aux)

// RunGuard v2
static uint16_t RG_LAST_LINE[1u << RG_PC_BITS]; // 12 bits used
static uint8_t  RG_CONF     [1u << RG_PC_BITS]; // 0..3
static uint8_t  RG_FLAGS    [1u << RG_PC_BITS]; // bit0=abs12 prev seen, bit1=fwd prev

// ---------------- Selector (set dueling) ----------------
static int32_t policy_score = 0;               // >0 favors Mode B
static uint8_t follower_mode[LLC_SETS];        // 0=A, 1=B for follower sets

// RunGuard update on demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = rg_pc_idx(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad    = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        bool prev_abs12 = (RG_FLAGS[idx] & 0x1) != 0;
        bool prev_fwd   = (RG_FLAGS[idx] & 0x2) != 0;
        if (prev_abs12 && prev_fwd) {
            sat_inc_u8(RG_CONF[idx], RG_CONF_MAX); // two consecutive forward ±1/±2 steps
        } else {
            // arm the detector for next step
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_FLAGS[idx] = 0x3; // remember pattern
        }
    } else {
        sat_dec_u8(RG_CONF[idx]); // decay on non-stream-like behavior
        RG_FLAGS[idx] = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Victim finder: RRIP with aging
static inline uint32_t rrip_victim(uint32_t set, const BLOCK* current_set) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Try to find an RRPV==MAX; age until found (bounded)
    for (uint32_t round = 0; round < (RRPV_MAX + 1); round++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == RRPV_MAX) return w;
        }
        // age all lines that are not already at MAX
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] < RRPV_MAX) RRPV[set][w]++;
        }
    }
    // Fallback (should not happen)
    return 0;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        follower_mode[s] = 0; // default Mode A
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]   = RRPV_MAX;
            PCSIG[s][w]  = 0;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
        }
    }
    for (uint32_t i = 0; i < A_SHCT_SIZE; i++) A_SHCT[i] = A_SHCT_WARM; // start neutral/warm
    for (uint32_t i = 0; i < EUP_SIZE; i++)     EUP[i]    = 1;          // slight expected use
    for (uint32_t i = 0; i < (1u << RG_PC_BITS); i++) {
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_FLAGS[i]     = 0;
    }
    policy_score = 0;
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
    return rrip_victim(set, current_set);
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    bool demand = is_demand(type);

    // Train RunGuard on demand access
    if (demand) update_runguard(PC, paddr);

    if (hit) {
        // Demand hits: strict multi-hit gate; prefetch hits: no promotion
        if (demand) {
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;
            uint8_t gate = (STATUS[set][way] == ST_QUAR) ? HITS_TO_PROMOTE_PREF_ORG : HITS_TO_PROMOTE_DEMAND;
            if (HITCNT[set][way] >= gate) {
                RRPV[set][way] = 0;            // MRU on confirmed reuse
                STATUS[set][way] = ST_NORMAL;  // escape from stream/quarantine
            } else {
                // gentle nudge but never to MRU before gate
                if (RRPV[set][way] > 1) RRPV[set][way]--;
            }
        } else if (type == ACCESS_PREFETCH) {
            // Do not promote on prefetch-only hits; slight nudge at most
            if (RRPV[set][way] > 2) RRPV[set][way]--;
        }
        return;
    }

    // Miss/Fill path: train victim (previous occupant) before overwrite
    if (victim_addr != 0) {
        uint8_t vsig = PCSIG[set][way];
        if (HITCNT[set][way] == 0) {
            if (A_SHCT[vsig] > 0) A_SHCT[vsig]--;
            if (EUP[vsig] > 0)    EUP[vsig]--;
        } else {
            if (A_SHCT[vsig] < A_SHCT_MAX) A_SHCT[vsig]++;
            if (EUP[vsig] < EUP_MAX)       EUP[vsig]++;
        }
    }

    // Selector: update score using leader sets on demand misses only
    if (demand) {
        if (LEADER_A(set)) {
            if (policy_score > (INT32_MIN + 1)) policy_score--;
        } else if (LEADER_B(set)) {
            if (policy_score < (INT32_MAX - 1)) policy_score++;
        }
        // Hysteresis-based switching for follower sets
        if (policy_score >= SCORE_UP_TH) {
            for (uint32_t s = 0; s < LLC_SETS; s++) {
                if (!LEADER_A(s) && !LEADER_B(s)) follower_mode[s] = 1; // Mode B
            }
        } else if (policy_score <= SCORE_DN_TH) {
            for (uint32_t s = 0; s < LLC_SETS; s++) {
                if (!LEADER_A(s) && !LEADER_B(s)) follower_mode[s] = 0; // Mode A
            }
        }
    }

    // Determine which mode applies to this set
    uint8_t mode = 0;
    if (LEADER_A(set)) mode = 0;
    else if (LEADER_B(set)) mode = 1;
    else mode = follower_mode[set];

    // Choose insertion policy
    uint8_t ins_rrpv = RRPV_MAX;
    uint8_t new_status = ST_NORMAL;
    uint32_t sig = pc_sig(PC);
    uint8_t friendly = (A_SHCT[sig] >= A_SHCT_WARM) ? 1 : 0;
    uint8_t rg_conf = RG_CONF[rg_pc_idx(PC)];
    bool streamy = (rg_conf >= RG_BYPASS_TH);

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writebacks; insert near-tail
        ins_rrpv = (RRPV_MAX > 0) ? (RRPV_MAX - 1) : RRPV_MAX;
        new_status = ST_NORMAL;
    } else if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine at the tail
        ins_rrpv = RRPV_MAX;
        new_status = ST_QUAR;
    } else {
        // Demand (LOAD/RFO)
        if (mode == 1) {
            // Mode B: strong stream suppression + cold PC guard
            if (streamy) {
                ins_rrpv = RRPV_MAX;
                new_status = ST_STREAM;
            } else if (!friendly) {
                ins_rrpv = RRPV_MAX;
                new_status = ST_NORMAL;
            } else {
                ins_rrpv = RRPV_YOUNG_B;
                new_status = ST_NORMAL;
            }
        } else {
            // Mode A: Hawkeye-like friendliness with stream guard
            ins_rrpv = friendly ? RRPV_YOUNG_A : RRPV_MAX;
            if (streamy) {
                ins_rrpv = RRPV_MAX;
                new_status = ST_STREAM;
            }
        }
    }

    // Install new line state
    RRPV[set][way]   = ins_rrpv;
    PCSIG[set][way]  = static_cast<uint8_t>(sig);
    HITCNT[set][way] = 0;
    STATUS[set][way] = new_status;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}