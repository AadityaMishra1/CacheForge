#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS        = 6;      // 64-entry PC tables (compact)
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);
static constexpr uint8_t  LINE_LOW_BITS  = 12;     // for run/stride detection
static constexpr uint8_t  STREAM_TH      = 2;      // stream confidence threshold
static constexpr uint8_t  HITS_TO_PROMOTE= 2;      // promote on 2nd+ demand hit
static constexpr uint8_t  DEPTH_MAX      = 7;      // shared with RRIP (Mode A)
static constexpr uint8_t  DEPTH_YOUNG    = 2;      // slightly young on insert
static constexpr uint16_t EPOCH_LEN      = 256;    // per-set epoch to judge Mode B

// ---------------- Per-line state (shared across modes) ----------------
// DEPTH: acts as both RRIP (Mode A) and "age" (Mode B)
static uint8_t DEPTH[LLC_SETS][LLC_WAYS];   // 3-bit conceptual (0..7)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];  // 2-bit (0,1,2=2+)
static uint8_t STATUS_[LLC_SETS][LLC_WAYS]; // 0=normal, 1=stream, 2=quarantine(prefetch)
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];   // 6-bit PC index (0..63)
static uint8_t INSRC [LLC_SETS][LLC_WAYS];  // 0=Mode A, 1=Mode B at insertion

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors (Mode B) ----------------
static uint8_t EUP[PC_SIZE];                // 2-bit expected-use per PC (0..3)
static uint16_t RG_LAST_LINE[PC_SIZE];      // low bits of last line number
static uint8_t  RG_CONF[PC_SIZE];           // 2-bit stream confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];     // last step had |delta| in {1,2}
static uint8_t  RG_LAST_FWD[PC_SIZE];       // last step forward?

// ---------------- Per-set selector (defaults to Hawkeye-like Mode A) ----------------
static uint8_t  SEL_CHOICE[LLC_SETS];       // 0=Mode A, 1=Mode B
static int8_t   SEL_SCORE [LLC_SETS];       // signed small score for Mode B
static uint16_t SEL_EPOCH [LLC_SETS];       // progress within epoch
static uint8_t  LAST_VICTIM_WAS_INVALID[LLC_SETS]; // for eviction training gating

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // simple 6-bit fold hash
    uint64_t x = pc ^ (pc >> 11) ^ (pc >> 23) ^ (pc >> 37);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_low(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return type == ACCESS_LOAD || type == ACCESS_RFO;
}
static inline void sat_inc_u2(uint8_t &v) { if (v < 3) v++; }
static inline void sat_dec_u2(uint8_t &v) { if (v > 0) v--; }
static inline void sat_inc_s(int8_t &v, int8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec_s(int8_t &v, int8_t minv) { if (v > minv) v--; }

// Update RunGuard confidence on a demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_low(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc_u2(RG_CONF[idx]); // two consecutive forward small steps
        } else {
            // weakly enable stream tracking
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u2(RG_CONF[idx]);     // decay on breaks/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Mode B victim comparison
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict quarantined/stream-tagged first
    bool a_s = (STATUS_[set][a] != ST_NORMAL);
    bool b_s = (STATUS_[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer lower EUP (colder PC)
    uint8_t ea = EUP[ PCSIZ[set][a] & (PC_SIZE-1) ];
    uint8_t eb = EUP[ PCSIZ[set][b] & (PC_SIZE-1) ];
    if (ea != eb) return (ea < eb);

    // 3) Prefer fewer observed demand hits
    if (HITCNT[set][a] != HITCNT[set][b]) return (HITCNT[set][a] < HITCNT[set][b]);

    // 4) Tie-break by deeper (older) depth
    return DEPTH[set][a] > DEPTH[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            DEPTH[s][w]  = DEPTH_MAX;
            HITCNT[s][w] = 0;
            STATUS_[s][w]= ST_NORMAL;
            PCSIG[s][w]  = 0;
            INSRC[s][w]  = 0; // default Mode A
        }
        SEL_CHOICE[s] = 0;    // start in Hawkeye-like mode
        SEL_SCORE[s]  = 0;
        SEL_EPOCH[s]  = 0;
        LAST_VICTIM_WAS_INVALID[s] = 0;
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]          = 1;  // slightly cold
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS12[i]= 0;
        RG_LAST_FWD[i]  = 0;
    }
    std::cout << "Init PhaseHawk-Fusion (ModeA: RRIP-like, ModeB: RunGuard+EUP, selector default=A)" << std::endl;
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
    // return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            LAST_VICTIM_WAS_INVALID[set] = 1;
            return w;
        }
    }
    LAST_VICTIM_WAS_INVALID[set] = 0;

    // Select policy for the set
    bool use_mode_b = (SEL_CHOICE[set] != 0);

    if (use_mode_b) {
        // Mode B: multi-signal victim selection
        uint32_t vic = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable(set, w, vic)) vic = w;
        }
        return vic;
    } else {
        // Mode A: Hawkeye-like RRIP (no bypass)
        // find a line at max depth; if none, age all and retry (bounded)
        for (int it = 0; it < 8; it++) {
            for (uint32_t w = 0; w < LLC_WAYS; w++) {
                if (DEPTH[set][w] == DEPTH_MAX) return w;
            }
            // increment all (saturating) to create a victim next iteration
            for (uint32_t w = 0; w < LLC_WAYS; w++) {
                if (DEPTH[set][w] < DEPTH_MAX) DEPTH[set][w]++;
            }
        }
        // fallback (should not happen)
        return 0;
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
    if (type == ACCESS_WRITEBACK) return;

    const bool is_dem = is_demand(type);
    uint32_t pcidx = pc_index(PC);

    // Eviction-time training for the line being replaced (on fills only)
    if (!hit && !LAST_VICTIM_WAS_INVALID[set]) {
        // Use existing line metadata before overwrite
        uint8_t old_hits = HITCNT[set][way];
        uint8_t old_pc   = PCSIG[set][way] & (PC_SIZE - 1);
        uint8_t old_stat = STATUS_[set][way];
        uint8_t old_insb = INSRC[set][way];

        // Useless fill: no demand hits observed
        if (old_hits == 0) {
            // Train PC expected-use colder (ignore pure prefetched quarantines if desired)
            if (old_stat != ST_QUAR) { if (EUP[old_pc] > 0) EUP[old_pc]--; }
            // Penalize Mode B only if it inserted this line
            if (old_insb) sat_dec_s(SEL_SCORE[set], -7);
        } else {
            // Beneficial line: warm up PC and reward Mode B if it owned insertion
            if (EUP[old_pc] < 3) EUP[old_pc]++;
            if (old_insb) sat_inc_s(SEL_SCORE[set], +7);
        }
        // clear old line's per-line state (optional; will be overwritten below)
    }

    // Demand hit: multi-hit promotion gate
    if (hit && is_dem) {
        if (HITCNT[set][way] < 2) HITCNT[set][way]++;
        if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
            // Confirmed reuse: MRU and clear stream/quarantine
            DEPTH[set][way]  = 0;
            STATUS_[set][way]= ST_NORMAL;
            // reinforce EUP
            uint8_t sig = PCSIG[set][way] & (PC_SIZE - 1);
            if (EUP[sig] < 3) EUP[sig]++;
        } else {
            // soft touch
            if (DEPTH[set][way] > 1) DEPTH[set][way] = 1;
        }
        return;
    }

    // Prefetches are quarantined; streaming PCs go tail, no promotion
    uint8_t stream_conf = 0;
    if (is_dem) {
        stream_conf = update_runguard(PC, paddr);
    } else {
        // Non-demand: decay stream conf slightly to avoid over-sticking
        sat_dec_u2(RG_CONF[pcidx]);
        stream_conf = RG_CONF[pcidx];
    }

    // Set-level choice (defaults to Mode A); but enforce stream near-bypass immediately
    const bool set_mode_b = (SEL_CHOICE[set] != 0);
    const bool force_stream = (stream_conf >= STREAM_TH);
    const bool use_mode_b_now = set_mode_b || force_stream;

    // Insertion policy
    PCSIG[set][way] = static_cast<uint8_t>(pcidx);
    INSRC[set][way] = use_mode_b_now ? 1 : 0;
    HITCNT[set][way]= 0;

    if (type == ACCESS_PREFETCH) {
        // quarantine prefetches at the hard tail
        STATUS_[set][way] = ST_QUAR;
        DEPTH[set][way]   = DEPTH_MAX;
    } else if (use_mode_b_now) {
        // Mode B: RunGuard + EUP
        if (force_stream) {
            STATUS_[set][way] = ST_STREAM; // never promote until 2nd demand hit
            DEPTH[set][way]   = DEPTH_MAX;
        } else {
            STATUS_[set][way] = ST_NORMAL;
            uint8_t eup = EUP[pcidx];
            DEPTH[set][way]   = (eup >= 2) ? DEPTH_YOUNG : DEPTH_MAX;
        }
    } else {
        // Mode A: Hawkeye-like RRIP (EUP as friend/foe proxy)
        STATUS_[set][way] = ST_NORMAL;
        uint8_t eup = EUP[pcidx];
        DEPTH[set][way]   = (eup >= 2) ? DEPTH_YOUNG : DEPTH_MAX;
    }

    // Epoch and selector update (only accounts Mode B performance)
    if (!hit) {
        SEL_EPOCH[set]++;
        if (SEL_EPOCH[set] >= EPOCH_LEN) {
            // Flip to Mode B only on net positive score; otherwise default to A
            SEL_CHOICE[set] = (SEL_SCORE[set] > 0) ? 1 : 0;
            // gentle decay toward zero to avoid sticky decisions
            if (SEL_SCORE[set] > 0) SEL_SCORE[set]--;
            else if (SEL_SCORE[set] < 0) SEL_SCORE[set]++;
            SEL_EPOCH[set] = 0;
        }
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}