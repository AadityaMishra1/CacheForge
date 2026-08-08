#include <vector>
#include <cstdint>
#include <iostream>
#include <cmath>
#include "../inc/champsim_crc2.h"

// HSB Fusion: Mode A (Hawkeye-like RRIP path) + Mode B (RunFiltered-PC Boost)
// Selection: per-epoch, leader-set dueling; followers inherit winner (defaults to Mode A).

// ------------ ChampSim constants ------------
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ------------ Bit helpers ------------
static inline uint64_t bitmask64(uint32_t l) { return (l == 64) ? ~0ULL : ((1ULL << l) - 1ULL); }
static inline uint64_t bits64(uint64_t x, uint32_t i, uint32_t l) { return (x >> i) & bitmask64(l); }

// Sample leaders: 32 sets each
#define LEADER_A(set) ((set & 63u) == 0u)
#define LEADER_B(set) ((set & 63u) == 1u)

// ------------ Tunables (knobs) ------------
// Mode B
static constexpr uint8_t  PC_BITS        = 9;               // 512-entry tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);
static constexpr uint8_t  EUP_INIT       = 1;               // 2-bit counter (0..3)
static constexpr uint8_t  EUP_WARM_TH    = 2;               // >=2 => warm
static constexpr uint8_t  LINE_LOW_BITS  = 12;              // low bits of line# used by RunGuard
static constexpr uint8_t  RG_BYPASS_TH   = 2;               // >=2 => treat as stream
static constexpr uint8_t  AGE_MAX        = 3;               // 2-bit age (0 young .. 3 old)
static constexpr uint8_t  AGE_YOUNG      = 1;               // slightly young for warm PCs
static constexpr uint8_t  HITS_TO_PROMOTE = 2;              // promote on 2nd+ demand hit
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };     // per-line status

// Mode A (Hawkeye-like RRIP path)
static constexpr uint8_t  RRPV_MAX       = 7;               // 3-bit RRIP (0..7)
static constexpr uint8_t  RRPV_INIT_FRIENDLY = 0;           // friendly insert
static constexpr uint8_t  RRPV_INIT_AVERSE   = 6;           // averse insert

// Selection
static constexpr uint64_t EPOCH_LENGTH   = (1ULL << 18);    // accesses per epoch (~262k)
static constexpr uint32_t DUEL_MARGIN    = 64;              // hysteresis (absolute hits)

// ------------ Mode B: per-line state ------------
static uint8_t AGE   [LLC_SETS][LLC_WAYS];   // 2 bits
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];   // 2 bits: ST_*
static uint16_t PCSIG[LLC_SETS][LLC_WAYS];   // 9 bits stored in 16-bit container

// Mode B: global predictors
static uint8_t  EUP[PC_SIZE];                // 2 bits per PC
static uint16_t RG_LAST_LINE[PC_SIZE];       // 12-bit low line number
static uint8_t  RG_CONF[PC_SIZE];            // 2-bit
static uint8_t  RG_LAST_ABS12[PC_SIZE];      // 1-bit
static uint8_t  RG_LAST_FWD[PC_SIZE];        // 1-bit

// ------------ Mode A: RRIP path ------------
static uint8_t RRPV[LLC_SETS][LLC_WAYS];     // 3-bit logical

// ------------ Selector: per-set mode bit (0=A, 1=B) ------------
static uint8_t SET_MODE[LLC_SETS];           // followers adopt winner each epoch
static uint64_t access_count = 0;

// Leader-set dueling stats (counts since last epoch)
static uint64_t leaderA_hits = 0, leaderA_acc = 0;
static uint64_t leaderB_hits = 0, leaderB_acc = 0;
static uint8_t  global_mode = 0;             // 0 = Mode A (default), 1 = Mode B

// ------------ Helpers ------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }

// Update RunGuard; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc_u8(RG_CONF[idx], 3);        // two consecutive forward steps
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RG_CONF[idx]);               // decay on break/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

static inline void age_all_B(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Compare B-candidates: true if a is more evictable than b
static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // prefer evicting sentinels

    uint8_t ea = EUP[ PCSIG[set][a] & (PC_SIZE - 1) ];
    uint8_t eb = EUP[ PCSIG[set][b] & (PC_SIZE - 1) ];
    if (ea != eb) return (ea < eb); // colder PC first

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // fewer hits first

    return AGE[set][a] > AGE[set][b]; // older first
}

static inline uint32_t pick_victim_B(uint32_t set) {
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable_B(set, w, best)) best = w;
    }
    return best;
}

static inline uint32_t pick_victim_A(uint32_t set) {
    // Return first with max RRPV; otherwise the one with largest RRPV
    uint32_t lru_victim = 0;
    uint8_t maxv = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (RRPV[set][w] == RRPV_MAX) return w;
        if (RRPV[set][w] >= maxv) { maxv = RRPV[set][w]; lru_victim = w; }
    }
    return lru_victim;
}

static inline void epoch_maybe_roll() {
    access_count++;
    if ((access_count % EPOCH_LENGTH) != 0) return;

    // Decide winner with hysteresis
    uint8_t next_mode = 0; // default A
    if (leaderB_acc > 0 && leaderA_acc > 0) {
        // Compare hits; add small margin to avoid flapping
        if (leaderB_hits + DUEL_MARGIN > leaderA_hits) next_mode = 1;
    }
    global_mode = next_mode;

    // Apply to followers
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (!LEADER_A(s) && !LEADER_B(s)) SET_MODE[s] = global_mode;
    }

    // reset stats
    leaderA_hits = leaderA_acc = 0;
    leaderB_hits = leaderB_acc = 0;
}

// ------------ ChampSim API ------------

// Initialize replacement state
void InitReplacementState() {
    // Mode B
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
        }
        SET_MODE[s] = 0; // start in Mode A
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i] = EUP_INIT;
        RG_LAST_LINE[i] = 0;
        RG_CONF[i] = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
    }

    // Mode A
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w] = RRPV_MAX;
        }
    }

    global_mode = 0; // Hawkeye-like by default
    access_count = 0;
    leaderA_hits = leaderA_acc = 0;
    leaderB_hits = leaderB_acc = 0;
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

    // Resolve mode for this set
    uint8_t mode = LEADER_A(set) ? 0 : (LEADER_B(set) ? 1 : SET_MODE[set]);

    if (mode == 0) {
        return pick_victim_A(set);
    } else {
        return pick_victim_B(set);
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
    // Ignore writebacks for policy decisions
    if (type == ACCESS_WRITEBACK) {
        epoch_maybe_roll();
        return;
    }

    const bool demand = is_demand(type);
    const uint32_t pc_idx = pc_index(PC);

    // Update selector leader stats
    if (LEADER_A(set)) {
        leaderA_acc++;
        if (hit) leaderA_hits++;
    } else if (LEADER_B(set)) {
        leaderB_acc++;
        if (hit) leaderB_hits++;
    }

    // Common: update RunGuard only on demand
    if (demand) {
        update_runguard(PC, paddr);
    }

    // Mode resolution (for this event)
    uint8_t mode = LEADER_A(set) ? 0 : (LEADER_B(set) ? 1 : SET_MODE[set]);

    // Mode B: pre-eviction training on the line being replaced (if miss)
    if (!hit) {
        // Train EUP on the victim's observed reuse (if valid metadata present)
        uint16_t old_pcsig = PCSIG[set][way];
        uint8_t old_hits   = HITCNT[set][way];
        // Prefetch quarantine and stream lines are likely dead; count as cold if <2 hits
        if (old_hits >= HITS_TO_PROMOTE) sat_inc_u8(EUP[old_pcsig & (PC_SIZE - 1)], 3);
        else                              sat_dec_u8(EUP[old_pcsig & (PC_SIZE - 1)]);
    }

    // ---------------- Mode A (Hawkeye-like RRIP) updates ----------------
    // Minimal friendly/averse via EUP: warm -> friendly
    if (mode == 0) {
        if (hit) {
            // On hit, reset to MRU
            RRPV[set][way] = 0;
        } else {
            // Insert based on PC warmth; prefetches at tail
            if (type == ACCESS_PREFETCH) {
                RRPV[set][way] = RRPV_MAX;
            } else {
                if (EUP[pc_idx] >= EUP_WARM_TH) RRPV[set][way] = RRPV_INIT_FRIENDLY;
                else                            RRPV[set][way] = RRPV_INIT_AVERSE;
            }
        }
    }

    // ---------------- Mode B (Boost) updates ----------------
    // Lightweight age for B on every access to this set
    age_all_B(set);

    if (hit) {
        // Update hit counters and conditional promotion
        if (demand) {
            if (HITCNT[set][way] < HITS_TO_PROMOTE) HITCNT[set][way]++;
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                if (STATUS[set][way] == ST_QUAR) STATUS[set][way] = ST_NORMAL; // shed quarantine on proven reuse
                if (STATUS[set][way] != ST_STREAM) {
                    AGE[set][way] = 0; // promote only if not streaming
                    // reinforce EUP on successful promotion
                    sat_inc_u8(EUP[ PCSIG[set][way] & (PC_SIZE - 1) ], 3);
                }
            }
        }
        // Mode A also gets benefit on hit (above); Mode B status adjusts here
    } else {
        // Miss fill: initialize per-line state for B
        PCSIG[set][way]  = static_cast<uint16_t>(pc_idx);
        HITCNT[set][way] = 0;

        if (type == ACCESS_PREFETCH) {
            STATUS[set][way] = ST_QUAR;
            AGE[set][way]    = AGE_MAX; // quarantine at tail
        } else {
            // Stream detection via RunGuard
            uint8_t rgc = RG_CONF[pc_idx];
            if (rgc >= RG_BYPASS_TH) {
                STATUS[set][way] = ST_STREAM;
                AGE[set][way]    = AGE_MAX; // hard tail; never promote
            } else {
                STATUS[set][way] = ST_NORMAL;
                // Warm PCs get slight head-start; cold PCs at tail
                AGE[set][way] = (EUP[pc_idx] >= EUP_WARM_TH) ? AGE_YOUNG : AGE_MAX;
            }
        }
    }

    epoch_maybe_roll();
}

// Print end-of-simulation statistics
void PrintStats() {
    // minimal/no prints by design
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // minimal/no prints by design
}