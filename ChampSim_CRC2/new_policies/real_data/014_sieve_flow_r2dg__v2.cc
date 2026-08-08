#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS                 = 8;     // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE                 = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS           = 12;    // compare low bits of line number
static constexpr uint8_t  STREAM_NEAR_TH          = 2;     // >=2 => stream hard-tail insertion
static constexpr uint8_t  STREAM_BYPASS_TH        = 3;     // >=3 => sustained run

// Promotion gates (dual): warm PCs(>=2) => 2 hits, cold PCs => 3 hits
static constexpr uint8_t  HITS_WARM_PROMOTE       = 2;
static constexpr uint8_t  HITS_COLD_PROMOTE       = 3;

// Insertion ages: 0=MRU (young) .. 3=LRU (old)
static constexpr uint8_t  AGE_MAX                 = 3;     // hard tail
static constexpr uint8_t  AGE_YOUNG               = 1;     // slightly young for warm PCs
static constexpr uint8_t  AGE_STREAM_YOUNG        = 2;     // admitted stream (budgeted) age

// VUT thresholds (2-bit counters: 0..3)
static constexpr uint8_t  VUT_INIT                = 1;     // start slightly cold
static constexpr uint8_t  VUT_WARM_TH             = 2;     // >=2 => warm PC

// Per-set stream token budget (admit a few stream lines "young")
static constexpr uint8_t  STREAM_TOKENS_BUDGET    = 2;

// Status encoding (2 bits total)
// 0: normal, 1: stream-tagged, 2: prefetch quarantine depth 1, 3: prefetch quarantine depth 2
enum : uint8_t { S_NORMAL=0, S_STREAM=1, S_QUAR1=2, S_QUAR2=3 };

// ---------------- Per-line state (packed; bytes in code, bits accounted in storage) ----------------
static uint8_t AGE    [LLC_SETS][LLC_WAYS];  // 2 bits (0..3)
static uint8_t DHITS  [LLC_SETS][LLC_WAYS];  // 2 bits (0..3 demand hits)
static uint8_t STATUS [LLC_SETS][LLC_WAYS];  // 2 bits: S_*
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];  // 8-bit PC index
static uint8_t LVALID [LLC_SETS][LLC_WAYS];  // 1-bit mirror of block validity

// ---------------- Per-set state ----------------
static uint8_t STREAM_TOKENS[LLC_SETS];      // 2 bits: 0..2

// ---------------- Global predictors ----------------
// Verified-Use Table (VUT): per-PC friendliness (2-bit 0..3)
static uint8_t VUT[PC_SIZE];

// Forward-run detector per PC
static uint16_t RD_LAST_LINE[PC_SIZE];  // low LINE_LOW_BITS of line#
static uint8_t  RD_CONF[PC_SIZE];       // 2-bit confidence (0..3)
static uint8_t  RD_LAST_ABS12[PC_SIZE]; // 1-bit: last step |delta| in {1,2}
static uint8_t  RD_LAST_FWD[PC_SIZE];   // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 15) ^ (pc >> 23);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update forward-run detector on demand access; returns updated confidence
static inline uint8_t update_run(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RD_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RD_LAST_ABS12[idx] && RD_LAST_FWD[idx]) {
            sat_inc(RD_CONF[idx], 3); // consecutive forward small strides
        } else {
            if (RD_CONF[idx] == 0) RD_CONF[idx] = 1;
            RD_LAST_ABS12[idx] = 1;
            RD_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RD_CONF[idx]);       // decay on breaks/backward/large strides
        RD_LAST_ABS12[idx] = 0;
        RD_LAST_FWD[idx]   = 0;
    }
    RD_LAST_LINE[idx] = curr;
    return RD_CONF[idx];
}

// Eviction ordering: stream/quarantine first, then colder PC, then fewer demand hits, then older age
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != S_NORMAL);
    bool b_s = (STATUS[set][b] != S_NORMAL);
    if (a_s != b_s) return a_s;

    uint8_t va = VUT[ PCSIG[set][a] ];
    uint8_t vb = VUT[ PCSIG[set][b] ];
    if (va != vb) return (va < vb);

    uint8_t ha = DHITS[set][a];
    uint8_t hb = DHITS[set][b];
    if (ha != hb) return (ha < hb);

    return AGE[set][a] > AGE[set][b];
}

// ---------------- CRC2 interface ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            DHITS[s][w]  = 0;
            STATUS[s][w] = S_NORMAL;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
        }
        STREAM_TOKENS[s] = STREAM_TOKENS_BUDGET;
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        VUT[i]           = VUT_INIT;
        RD_LAST_LINE[i]  = 0;
        RD_CONF[i]       = 0;
        RD_LAST_ABS12[i] = 0;
        RD_LAST_FWD[i]   = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Composite victim selection
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
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
    uint32_t pcidx = pc_index(PC);

    // Demand accesses update run detector early
    if (is_demand(type)) {
        (void)update_run(PC, paddr);
    }

    if (hit) {
        // On hit: apply quarantine countdown and dual-gated promotion on demand
        if (type == ACCESS_PREFETCH) {
            // Prefetch hits do not alter reuse score or promotion/quarantine
            return;
        }

        if (STATUS[set][way] == S_QUAR2) {
            STATUS[set][way] = S_QUAR1; // first demand touch
        } else if (STATUS[set][way] == S_QUAR1) {
            STATUS[set][way] = S_NORMAL; // second demand touch exits quarantine
        }

        // Count only demand hits toward reuse
        if (is_demand(type)) {
            sat_inc(DHITS[set][way], 3);
            uint8_t hits_to_promote = (VUT[ PCSIG[set][way] ] >= VUT_WARM_TH) ? HITS_WARM_PROMOTE : HITS_COLD_PROMOTE;

            if (DHITS[set][way] >= hits_to_promote) {
                // Verified reuse: clear stream tag if present, refund a token
                if (STATUS[set][way] == S_STREAM) {
                    STATUS[set][way] = S_NORMAL;
                    if (STREAM_TOKENS[set] < STREAM_TOKENS_BUDGET) STREAM_TOKENS[set]++;
                }
                AGE[set][way] = 0; // promote to MRU only after verified reuse
                // Train VUT on second demand hit
                if (DHITS[set][way] == 2) {
                    sat_inc(VUT[ PCSIG[set][way] ], 3);
                }
            } else {
                // No promotion before gate; keep age as is
                if (DHITS[set][way] == 2) {
                    sat_inc(VUT[ PCSIG[set][way] ], 3);
                }
            }
        }
        return;
    }

    // Miss path: train on victim (if valid), then insert new line
    if (LVALID[set][way]) {
        // Victim feedback: penalize cold PCs; release a token if a stream line is evicted
        if (DHITS[set][way] <= 1) {
            sat_dec(VUT[ PCSIG[set][way] ]);
        }
        if (STATUS[set][way] == S_STREAM) {
            if (STREAM_TOKENS[set] < STREAM_TOKENS_BUDGET) STREAM_TOKENS[set]++;
        }
    }

    // Aging step to keep lightweight recency
    age_all(set);

    // Decide insertion for the new line
    uint8_t ins_age   = AGE_MAX;
    uint8_t ins_stat  = S_NORMAL;
    uint8_t ins_dhits = 0;

    if (type == ACCESS_PREFETCH) {
        // Quarantine prefetches at hard tail with depth=2
        ins_stat = S_QUAR2;
    } else if (is_demand(type)) {
        uint8_t conf = RD_CONF[pcidx]; // from previous update_run
        bool stream_suspected = (conf >= STREAM_NEAR_TH);

        if (stream_suspected) {
            ins_stat = S_STREAM;
            if (conf >= STREAM_BYPASS_TH && STREAM_TOKENS[set] > 0) {
                ins_age = AGE_STREAM_YOUNG;
                STREAM_TOKENS[set]--; // admit a few young stream lines
            } else {
                ins_age = AGE_MAX; // hard tail (near-bypass)
            }
        } else {
            // PC-based adaptive insertion
            if (VUT[pcidx] >= VUT_WARM_TH) ins_age = AGE_YOUNG; // mild priority for warm PCs
            else                           ins_age = AGE_MAX;   // cold PCs insert old
        }
    } else {
        // WRITEBACK: never bypass; insert at tail as normal
        ins_stat = S_NORMAL;
        ins_age  = AGE_MAX;
    }

    // Install new line
    AGE[set][way]    = ins_age;
    STATUS[set][way] = ins_stat;
    DHITS[set][way]  = ins_dhits;
    PCSIG[set][way]  = static_cast<uint8_t>(pcidx);
    LVALID[set][way] = 1;
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}