#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

// HawkFusion-StreamGuard: Mode A (Hawkeye-like) + Mode B (RunGuard++)
// - Per-set selector defaults to Mode A and switches to Mode B only when B's score > A's score.
// - No real bypass: we perform "near-bypass" by hard-tail insertion and no-promote for streams.
// - Multi-hit promotion gate for Mode B: promote only on 2nd+ demand hit; prefetch hits never promote.

// --------------- Configuration ---------------
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// Tunables
static constexpr uint8_t  PC_BITS        = 5;               // 32-entry PC-indexed tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);
static constexpr uint8_t  HAWK_BITS      = 5;               // Hawkeye-like counter width (0..31)
static constexpr uint8_t  HAWK_FRIENDLY  = 16;              // >=16 => friendly

// Mode A: RRIP with 2-bit counters (0..3)
static constexpr uint8_t  RRPV_MAX_A     = 3;

// Mode B: age (0=MRU..3=LRU)
static constexpr uint8_t  AGE_MAX_B      = 3;
static constexpr uint8_t  AGE_WARM_B     = 1;

// Multi-hit promotion threshold (Mode B)
static constexpr uint8_t  HITS_TO_PROMOTE = 2;              // Promote on 2nd+ demand hit

// RunGuard stream detection (per-PC), use low bits of line number
static constexpr uint8_t  LINE_LOW_BITS  = 12;              // compare low line bits
static constexpr uint8_t  RG_BYPASS_TH   = 2;               // >=2 => treat as stream (near-bypass)

// Status encodings (Mode B)
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// Selector
enum : uint8_t { MODE_A=0, MODE_B=1 };
static constexpr uint8_t  SEL_HYST       = 1;               // B must exceed A by >0 to win
static constexpr uint8_t  SEL_MAX        = 7;               // 3-bit scores

// --------------- Per-line state ---------------
// Shared per-line PC signature (5 bits compressed)
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];   // 0..31

// Mode A (Hawkeye-like) line state
static uint8_t RRPV_A[LLC_SETS][LLC_WAYS];  // 2-bit RRIP (0..3)

// Mode B line state
static uint8_t AGE_B   [LLC_SETS][LLC_WAYS]; // 2 bits (0..3)
static uint8_t HITCNT_B[LLC_SETS][LLC_WAYS]; // 0,1,2(=2+), store 0..2
static uint8_t STATUS_B[LLC_SETS][LLC_WAYS]; // ST_*

// Owner path of the line (insertion owner): 0=A, 1=B
static uint8_t OWNER[LLC_SETS][LLC_WAYS];    // 1 bit logically

// --------------- Global / per-PC tables ---------------
// EUP: PC coldness predictor (2-bit)
static uint8_t EUP[PC_SIZE];                 // 0..3

// RunGuard per-PC
static uint16_t RG_LAST_LINE[PC_SIZE];       // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];            // 0..3
static uint8_t  RG_LAST_ABS12[PC_SIZE];      // 0/1
static uint8_t  RG_LAST_FWD[PC_SIZE];        // 0/1

// Hawkeye-like per-PC SHCT (use the same 32-entry PC index for compactness)
static uint8_t HAWK_SHCT[PC_SIZE];           // 0..31

// --------------- Per-set selector ---------------
static uint8_t SEL_MODE[LLC_SETS];           // 0=A, 1=B (default A)
static uint8_t SEL_SCORE_A[LLC_SETS];        // 3-bit saturating score
static uint8_t SEL_SCORE_B[LLC_SETS];        // 3-bit saturating score

// --------------- Helpers ---------------
static inline uint32_t pc_index(uint64_t pc) {
    // Compact 5-bit PC signature (XOR-folding)
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 17) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x)              { if (x > 0) x--; }

// RunGuard update: returns new stream confidence
static inline uint8_t update_runguard(uint32_t pc_idx, uint64_t paddr) {
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[pc_idx]));
    int16_t ad    = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12    = (ad == 1) || (ad == 2);
    bool fwd      = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[pc_idx] && RG_LAST_FWD[pc_idx]) {
            sat_inc(RG_CONF[pc_idx], 3); // two consecutive forward ±1/±2 steps
        } else {
            if (RG_CONF[pc_idx] == 0) RG_CONF[pc_idx] = 1;
            RG_LAST_ABS12[pc_idx] = 1;
            RG_LAST_FWD[pc_idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[pc_idx]); // decay on break/backward/irregular
        RG_LAST_ABS12[pc_idx] = 0;
        RG_LAST_FWD[pc_idx]   = 0;
    }
    RG_LAST_LINE[pc_idx] = curr;
    return RG_CONF[pc_idx];
}

// Mode B victim comparison
static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS_B[set][a] != ST_NORMAL);
    bool b_s = (STATUS_B[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict sentinels first

    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // colder PCs first

    uint8_t ha = HITCNT_B[set][a];
    uint8_t hb = HITCNT_B[set][b];
    if (ha != hb) return (ha < hb); // fewer hits first

    return AGE_B[set][a] > AGE_B[set][b];    // older first
}

// Apply Mode A RRIP promotion on hit
static inline void promote_A(uint32_t set, uint32_t way) {
    RRPV_A[set][way] = 0;
}

// Apply Mode B promotion on reaching threshold
static inline void promote_B(uint32_t set, uint32_t way) {
    AGE_B[set][way] = 0;
    STATUS_B[set][way] = ST_NORMAL;
}

// Selector: choose mode based on scores (B wins only if strictly better)
static inline uint8_t choose_mode(uint32_t set) {
    return (SEL_SCORE_B[set] > static_cast<uint8_t>(SEL_SCORE_A[set] + SEL_HYST)) ? MODE_B : MODE_A;
}

// --------------- Initialization ---------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        SEL_MODE[s]     = MODE_A;
        SEL_SCORE_A[s]  = 0;
        SEL_SCORE_B[s]  = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            PCSIG[s][w]   = 0;
            RRPV_A[s][w]  = RRPV_MAX_A;
            AGE_B[s][w]   = AGE_MAX_B;
            HITCNT_B[s][w]= 0;
            STATUS_B[s][w]= ST_NORMAL;
            OWNER[s][w]   = MODE_A;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]           = 1;  // slightly cold start
        RG_LAST_LINE[i]  = 0;
        RG_CONF[i]       = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
        HAWK_SHCT[i]     = 8;  // slightly averse start
    }
    std::cout << "Initialize HawkFusion-StreamGuard" << std::endl;
}

// --------------- Victim selection ---------------
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

    // Choose current mode for this set
    uint8_t mode = choose_mode(set);

    if (mode == MODE_A) {
        // Mode A: pick any line with max RRPV; else the one with largest RRPV
        uint32_t best = 0;
        uint8_t  best_rrpv = RRPV_A[set][0];
        if (best_rrpv == RRPV_MAX_A) return 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            uint8_t r = RRPV_A[set][w];
            if (r == RRPV_MAX_A) return w;
            if (r > best_rrpv) { best_rrpv = r; best = w; }
        }
        return best;
    } else {
        // Mode B: composite priority selection
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable_B(set, w, best)) best = w;
        }
        return best;
    }
}

// --------------- State update ---------------
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
    // Ignore writebacks for policy moves (never bypass writebacks)
    if (type == ACCESS_WRITEBACK) return;

    const bool demand = is_demand(type);
    const bool prefetch = (type == ACCESS_PREFETCH);
    uint32_t pc_idx = pc_index(PC);

    // STREAM detector: update only on demand accesses
    if (demand) {
        (void)update_runguard(pc_idx, paddr);
    }

    // Local references
    uint8_t &line_rrpv  = RRPV_A[set][way];
    uint8_t &line_age   = AGE_B[set][way];
    uint8_t &line_hits  = HITCNT_B[set][way];
    uint8_t &line_stat  = STATUS_B[set][way];
    uint8_t &line_owner = OWNER[set][way];
    uint8_t &line_pcsig = PCSIG[set][way];

    if (hit) {
        // On hit, update both modes' state consistently
        // Mode A: immediate promotion
        promote_A(set, way);
        // Mode B: strict multi-hit promotion gate (demand only)
        if (demand) {
            if (line_hits < 2) line_hits++;
            if (line_hits >= HITS_TO_PROMOTE) {
                promote_B(set, way);
            }
            // Positive reinforcement for EUP and Hawkeye-like table
            if (EUP[pc_idx] < 3) EUP[pc_idx]++;
            if (HAWK_SHCT[pc_idx] < 31) HAWK_SHCT[pc_idx]++;
        }
        // Selector scoring: reward the owner path on demand hit
        if (demand) {
            if (line_owner == MODE_A) sat_inc(SEL_SCORE_A[set], SEL_MAX);
            else                      sat_inc(SEL_SCORE_B[set], SEL_MAX);
        }
        return;
    }

    // Miss path (fill/replace)
    // Train selector and predictors on dead evictions (the evicted line resides at 'way')
    // If the evicted line had no demand hit since fill, penalize its owner
    if (line_hits == 0) {
        if (line_owner == MODE_A) sat_dec(SEL_SCORE_A[set]);
        else                      sat_dec(SEL_SCORE_B[set]);
        // Train Hawkeye-like SHCT negatively for the evicted line's PC signature
        uint32_t ev_sig = line_pcsig;
        if (HAWK_SHCT[ev_sig] > 0) HAWK_SHCT[ev_sig]--;
    }

    // Determine current mode and update insertion for BOTH modes (so we can switch later)
    uint8_t cur_mode = choose_mode(set);

    // Common: refresh per-line PC signature
    line_pcsig = static_cast<uint8_t>(pc_idx & (PC_SIZE - 1));
    line_owner = cur_mode; // remember owner for selector scoring
    line_hits  = 0;

    // Mode A insertion (Hawkeye-like RRIP)
    {
        bool friendly = (HAWK_SHCT[pc_idx] >= HAWK_FRIENDLY);
        if (prefetch) {
            line_rrpv = RRPV_MAX_A; // quarantine prefetches
        } else {
            line_rrpv = friendly ? 0 : RRPV_MAX_A;
        }
    }

    // Mode B insertion (RunGuard++ with EUP and stream quarantine)
    {
        bool is_stream = (RG_CONF[pc_idx] >= RG_BYPASS_TH) && demand;
        if (prefetch) {
            line_stat = ST_QUAR;
            line_age  = AGE_MAX_B;
        } else if (is_stream) {
            line_stat = ST_STREAM;
            line_age  = AGE_MAX_B;
        } else {
            line_stat = ST_NORMAL;
            // PC coldness: colder PCs insert older; warmer PCs slightly young
            uint8_t e = EUP[pc_idx];
            line_age = (e >= 2) ? AGE_WARM_B : AGE_MAX_B;
        }
    }

    // Optional: very light aging of Mode B on fills (keeps order progressing)
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE_B[set][w] < AGE_MAX_B) AGE_B[set][w]++;
    }

    // Update selector mode decision
    SEL_MODE[set] = choose_mode(set);
}

void PrintStats_Heartbeat() {}
void PrintStats() {}