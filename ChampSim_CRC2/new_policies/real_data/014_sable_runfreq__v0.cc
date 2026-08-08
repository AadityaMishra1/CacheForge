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
static constexpr uint8_t  PC_BITS           = 8;    // 256-entry per-PC tables
static constexpr uint32_t PC_SIZE           = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS     = 12;   // line# low bits for stride
static constexpr uint8_t  STREAM_CONF_TH    = 2;    // >=2 => stream near-bypass

static constexpr uint8_t  HITS_TO_PROMOTE   = 2;    // promote on 2nd+ demand hit

// Age: 0=young/MRU .. 3=old/LRU (tail)
static constexpr uint8_t  AGE_MAX           = 3;
static constexpr uint8_t  AGE_YOUNG         = 1;

// HF insertion/promotion thresholds
static constexpr uint8_t  HF_WARM_TH        = 3;    // HF >=3 => warm
static constexpr uint8_t  HF_MAX            = 7;

// RDB (reuse-distance bias): 0..3 (higher = longer RD => colder)
static constexpr uint8_t  RDB_MAX           = 3;
static constexpr uint8_t  RDB_COOL_TH       = 1;    // RDB <=1 considered not long-RD

// Sampled-set density and long-RD tick threshold
static constexpr uint8_t  SAMP_LOG2         = 6;    // 1/64 sets
static constexpr uint8_t  LONG_RD_TICKS     = 8;    // >=8 ticks => long reuse distance

// HF decay cadence on accesses to this PC (power-of-two mask)
static constexpr uint32_t HF_DECAY_MASK     = 0x3F; // decay about every 64th access to this PC

// ---------------- Status per line ----------------
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2, ST_PIN=3 }; // ST_PIN unused, reserved

// ---------------- Per-line state (compact) ----------------
// Note: We use byte arrays; report minimal info-theoretic bits in storage section.
static uint8_t AGE     [LLC_SETS][LLC_WAYS];  // conceptual 2 bits (0..3)
static uint8_t HITCNT  [LLC_SETS][LLC_WAYS];  // conceptual 2 bits (0,1,2=2+)
static uint8_t STATUS  [LLC_SETS][LLC_WAYS];  // conceptual 2 bits (ST_*)
static uint8_t PCSIG   [LLC_SETS][LLC_WAYS];  // 8-bit PC index

// ---------------- Global per-PC predictors ----------------
// Hit frequency (HF): 3-bit leaky bucket per PC
static uint8_t HF[PC_SIZE];                   // 0..7
// Reuse-distance bias (RDB): 2-bit, higher = long RD (colder)
static uint8_t RDB[PC_SIZE];                  // 0..3

// RunStride per PC: forward ±1/±2 detector
static uint16_t RS_LAST_LINE[PC_SIZE];        // low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit: last step |delta| in {1,2}
static uint8_t  RS_LAST_FWD[PC_SIZE];         // 1-bit: last step forward

// ---------------- Sampled-set reuse-distance sketch ----------------
static constexpr uint32_t SAMP_SETS = (LLC_SETS >> SAMP_LOG2);
static uint8_t SAMP_TICK[SAMP_SETS];                // 8-bit per-sampled-set tick
static uint8_t SAMP_FILL_TICK[SAMP_SETS][LLC_WAYS]; // 8-bit fill tick per way (sampled sets)

// Global access counter for HF decay sampling
static uint32_t GLOBAL_ACC_TICK = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 21);
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
static inline bool is_sampled_set(uint32_t set) {
    return ((set & ((1u << SAMP_LOG2) - 1)) == 0);
}
static inline uint32_t sampled_index(uint32_t set) {
    return (set >> SAMP_LOG2);
}

// Update RunStride on a demand access; returns updated confidence
static inline uint8_t update_runstride(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc(RS_CONF[idx], 3); // two consecutive forward ±1/±2 steps
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RS_CONF[idx]);        // decay on irregular/backward/large stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// Composite friendliness score (higher = warmer)
static inline int friend_score(uint8_t pcs) {
    int hf  = static_cast<int>(HF[pcs]);               // 0..7
    int rdb = static_cast<int>(RDB[pcs]);              // 0..3
    return hf - (rdb << 1); // penalize long-RD bias stronger than a single HF step
}

// Compare candidates: return true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting sentinels (stream/quarantine) first
    bool a_s = (STATUS[set][a] == ST_STREAM) || (STATUS[set][a] == ST_QUAR);
    bool b_s = (STATUS[set][b] == ST_STREAM) || (STATUS[set][b] == ST_QUAR);
    if (a_s != b_s) return a_s;

    // 2) Prefer lines from colder PCs (lower friend_score)
    int fa = friend_score(PCSIG[set][a]);
    int fb = friend_score(PCSIG[set][b]);
    if (fa != fb) return (fa < fb);

    // 3) Prefer fewer observed demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older is more evictable)
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
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        HF[i]            = 0;
        RDB[i]           = 0;
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
    }
    for (uint32_t si = 0; si < SAMP_SETS; si++) {
        SAMP_TICK[si] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            SAMP_FILL_TICK[si][w] = 0;
        }
    }
    GLOBAL_ACC_TICK = 0;
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
    GLOBAL_ACC_TICK++;

    // Age the set on every access (bounded)
    age_all(set);

    const bool demand = is_demand(type);

    // RunStride update only for demand accesses (prefetch/WB excluded)
    uint8_t rs_conf = 0;
    if (demand) {
        rs_conf = update_runstride(PC, paddr);
    }

    uint32_t pcs = pc_index(PC);

    // HF decay sampling: on any access from this PC, periodically decay its HF
    if ((GLOBAL_ACC_TICK & HF_DECAY_MASK) == 0) {
        sat_dec(HF[pcs]);
        // mild decay for RDB as well to allow recovery from long phases
        sat_dec(RDB[pcs]);
    }

    if (hit) {
        // On hit: update per-line hit counters and promotion gate
        if (demand) {
            // demand hit counts toward promotion
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;
            // Train HF on demand hit
            sat_inc(HF[pcs], HF_MAX);
        }
        // No promotion on first hit or on prefetch hits
        if (demand && (HITCNT[set][way] >= HITS_TO_PROMOTE)) {
            // Promote to MRU and clear any sentinel (enables short-reuse)
            AGE[set][way] = 0;
            STATUS[set][way] = ST_NORMAL;
            // Promotion implies short reuse: reduce long-RD bias
            sat_dec(RDB[ pcs ]);
        }
        return;
    }

    // Miss path: train victim (old block) before inserting new one
    // Train per-PC HF and RDB using victim's history
    uint8_t old_pcs = PCSIG[set][way];
    uint8_t old_hits = HITCNT[set][way];

    if (old_hits >= 2) sat_inc(HF[old_pcs], HF_MAX);
    else               sat_dec(HF[old_pcs]);

    if (is_sampled_set(set)) {
        uint32_t si = sampled_index(set);
        uint8_t curr_tick = SAMP_TICK[si];
        uint8_t fill_tick = SAMP_FILL_TICK[si][way];
        uint8_t delta = static_cast<uint8_t>(curr_tick - fill_tick); // modulo 256
        bool long_rd = (delta >= LONG_RD_TICKS);
        if (old_hits <= 1 && long_rd) sat_inc(RDB[old_pcs], RDB_MAX);
        else                          sat_dec(RDB[old_pcs]);
    }

    // Decide insertion for the incoming line
    uint8_t new_status = ST_NORMAL;
    uint8_t new_age    = AGE_MAX;
    uint8_t new_hc     = 0; // reset hit count

    if (type == ACCESS_PREFETCH) {
        // Prefetch: always quarantine at tail
        new_status = ST_QUAR;
        new_age    = AGE_MAX;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass on writebacks, but insert cold to avoid pollution
        new_status = ST_QUAR;
        new_age    = AGE_MAX;
    } else {
        // Demand load/RFO
        bool is_stream = (rs_conf >= STREAM_CONF_TH);
        if (is_stream) {
            // Near-bypass: tail insert with stream sentinel; never promotes on first hit
            new_status = ST_STREAM;
            new_age    = AGE_MAX;
        } else {
            // Non-stream: use PC warmth (HF) damped by long-RD bias (RDB)
            int fs = friend_score(pcs);
            if (fs >= static_cast<int>(HF_WARM_TH) && RDB[pcs] <= RDB_COOL_TH) {
                new_age = AGE_YOUNG; // slightly young start for warm PCs
            } else {
                new_age = AGE_MAX;   // tail
            }
            new_status = ST_NORMAL;
        }
    }

    // Install the new block's state
    AGE[set][way]    = new_age;
    HITCNT[set][way] = new_hc;
    STATUS[set][way] = new_status;
    PCSIG[set][way]  = pcs;

    // Update sampled-set timestamps for reuse-distance sketch
    if (is_sampled_set(set)) {
        uint32_t si = sampled_index(set);
        SAMP_FILL_TICK[si][way] = SAMP_TICK[si];
        SAMP_TICK[si]++; // advance tick per fill
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