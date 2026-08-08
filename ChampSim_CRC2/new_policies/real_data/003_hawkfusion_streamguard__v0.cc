#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Ensemble selector (set dueling) ----------------
// Sample 64 sets: split into 32 leader-A (Hawkeye) and 32 leader-B (RunGuard+EUP).
static constexpr uint32_t SET_BITS   = 11;  // log2(2048)
static constexpr uint32_t SAMPLE_L   = 6;   // 64 sampled sets
static inline bool is_sampled(uint32_t set) {
    uint32_t low  = set & ((1u << SAMPLE_L) - 1u);
    uint32_t high = (set >> (SET_BITS - SAMPLE_L)) & ((1u << SAMPLE_L) - 1u);
    return low == high;
}
static inline bool leaderA(uint32_t set) { return is_sampled(set) && (((set >> SAMPLE_L) & 1u) == 1u); }
static inline bool leaderB(uint32_t set) { return is_sampled(set) && (((set >> SAMPLE_L) & 1u) == 0u); }

// duel_score > 0 => prefer Mode B; <= 0 => Mode A
static int32_t duel_score = 0;
static inline bool use_modeB(uint32_t set) {
    if (leaderA(set)) return false;
    if (leaderB(set)) return true;
    return (duel_score > 0); // default Hawkeye when not clearly better
}

// ---------------- Tunables (knobs) ----------------
// PC-indexed tables
static constexpr uint8_t  PC_BITS        = 8;               // 256-entry tables (compact)
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);

// Expected-Use PC predictor (EUP): 2-bit (0..3)
static constexpr uint8_t  EUP_INIT       = 1;               // slightly cold
static constexpr uint8_t  EUP_WARM_TH    = 2;               // >=2 => warm

// RunGuard: per-PC ±1/±2 forward-run stream detector
static constexpr uint8_t  LINE_LOW_BITS  = 12;              // low bits of line# to compare
static constexpr uint8_t  RG_BYPASS_TH   = 2;               // >=2 => stream near-bypass

// RRIP (shared storage for both modes) 0..7
static constexpr uint8_t  maxRRPV        = 7;

// Insertion depths (RRIP): lower is younger
static constexpr uint8_t  INS_FRIENDLY   = 0;               // Mode A friendly
static constexpr uint8_t  INS_AVERSE     = maxRRPV - 1;     // Mode A averse
static constexpr uint8_t  INS_WARM       = 2;               // Mode B warm (slightly young)
static constexpr uint8_t  INS_COLD       = maxRRPV - 1;     // Mode B cold
static constexpr uint8_t  INS_STREAM     = maxRRPV;         // Mode B stream hard tail
static constexpr uint8_t  INS_PREFETCH   = maxRRPV;         // Prefetch quarantine

// Multi-hit promotion
static constexpr uint8_t  HITS_TO_PROMOTE = 2;              // promote on 2nd+ demand hit

// ---------------- Per-line state (packed; reported info-theoretically) ----------------
static uint8_t RRPV[LLC_SETS][LLC_WAYS];   // 3-bit conceptual (0..7), shared by A/B
static uint8_t HITCNT[LLC_SETS][LLC_WAYS]; // 2-bit conceptual (0,1,2=2+)
static uint8_t STATUS[LLC_SETS][LLC_WAYS]; // 2-bit: 0=normal,1=stream,2=quarantine
static uint8_t PCSIG [LLC_SETS][LLC_WAYS]; // 8-bit PC index

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors ----------------
// EUP: 2-bit counters per PC (shared by A/B)
static uint8_t EUP[PC_SIZE];

// RunGuard per-PC state (compact, reported info-theoretically)
static uint16_t RG_LAST_LINE[PC_SIZE];   // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];        // 2-bit (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];  // 1-bit
static uint8_t  RG_LAST_FWD[PC_SIZE];    // 1-bit

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
static inline void sat_dec(uint8_t &x)               { if (x > 0) x--; }
static inline void inc_eup(uint8_t idx)              { if (EUP[idx] < 3) EUP[idx]++; }
static inline void dec_eup(uint8_t idx)              { if (EUP[idx] > 0) EUP[idx]--; }

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

// Compare candidates for Mode B: true if a is more evictable than b
static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // stream/quar first

    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // colder PCs first

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // fewer hits first

    return RRPV[set][a] > RRPV[set][b]; // older last
}

// ---------------- CRC2 hooks ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]    = maxRRPV;
            HITCNT[s][w]  = 0;
            STATUS[s][w]  = ST_NORMAL;
            PCSIG[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]           = EUP_INIT;
        RG_LAST_LINE[i]  = 0;
        RG_CONF[i]       = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
    }
    duel_score = 0;

    std::cout << "Initialize HawkFusion-StreamGuard" << std::endl;
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

    if (use_modeB(set)) {
        // Mode B: composite priority selection
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable_B(set, w, best)) best = w;
        }
        return best;
    } else {
        // Mode A: Hawkeye-Lite: evict any line with RRPV==max; else choose the one with highest RRPV
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (RRPV[set][w] == maxRRPV) return w;
        }
        uint32_t victim = 0, best_rrip = RRPV[set][0];
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (RRPV[set][w] >= best_rrip) {
                best_rrip = RRPV[set][w];
                victim = w;
            }
        }
        return victim;
    }
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
    if (type == ACCESS_WRITEBACK) return; // never bypass on writeback; ignore

    const bool demand = is_demand(type);
    const uint8_t old_hits = HITCNT[set][way];
    const uint8_t old_status = STATUS[set][way];
    const uint8_t old_pcsig = PCSIG[set][way];

    // Dueling bookkeeping on sampled sets for demand requests (misses only)
    if (is_sampled(set) && demand && (hit == 0)) {
        if (leaderA(set)) duel_score -= 1;
        else if (leaderB(set)) duel_score += 1;
        // saturate duel_score softly to avoid overflow
        if (duel_score > 1024) duel_score = 1024;
        if (duel_score < -1024) duel_score = -1024;
    }

    // Shared per-line training on eviction (before overwrite): reward multi-hit, penalize 0/1-use
    if (hit == 0) {
        if (old_status != ST_QUAR) { // ignore pure prefetch quarantines
            if (old_hits >= HITS_TO_PROMOTE) inc_eup(old_pcsig);
            else                              dec_eup(old_pcsig);
        }
    }

    // Mode decision (deterministic wrt set)
    const bool modeB = use_modeB(set);

    if (hit) {
        // On hit
        if (demand) {
            // Demand hit: multi-hit promotion gate
            if (HITCNT[set][way] < HITS_TO_PROMOTE) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                // Promote to MRU in both modes
                RRPV[set][way] = 0;
                STATUS[set][way] = ST_NORMAL; // shed stream/quarantine tag after proof of reuse
                // Small positive reinforcement for the PC
                inc_eup( pc_index(PC) );
            } else {
                // gentle aging improvement
                if (RRPV[set][way] > 0) RRPV[set][way]--;
            }
        } else {
            // Prefetch hit: never promote aggressively
            if (RRPV[set][way] > 0) RRPV[set][way]--;
        }
        return;
    }

    // Miss + fill path
    const uint32_t pc_idx = pc_index(PC);

    // Update RunGuard only on demand references
    uint8_t rg_conf = 0;
    if (demand) rg_conf = update_runguard(PC, paddr);

    // Install/overwrite the victim's state
    PCSIG[set][way]   = static_cast<uint8_t>(pc_idx & 0xFF);
    HITCNT[set][way]  = 0;

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine at hard tail; never promote until 2nd demand hit
        STATUS[set][way] = ST_QUAR;
        RRPV[set][way]   = INS_PREFETCH;
        return;
    }

    if (modeB) {
        // Mode B insertion
        if (rg_conf >= RG_BYPASS_TH) {
            STATUS[set][way] = ST_STREAM;
            RRPV[set][way]   = INS_STREAM;  // near-bypass for streams
        } else if (EUP[pc_idx] >= EUP_WARM_TH) {
            STATUS[set][way] = ST_NORMAL;
            RRPV[set][way]   = INS_WARM;    // slightly young
        } else {
            STATUS[set][way] = ST_NORMAL;
            RRPV[set][way]   = INS_COLD;    // cold tail
        }
    } else {
        // Mode A (Hawkeye-Lite) insertion
        STATUS[set][way] = ST_NORMAL;
        if (EUP[pc_idx] >= EUP_WARM_TH) {
            RRPV[set][way] = INS_FRIENDLY;  // friendly PCs at MRU
        } else {
            RRPV[set][way] = INS_AVERSE;    // averse PCs near LRU
        }
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}