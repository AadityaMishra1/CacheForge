#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

#ifndef LOAD
#define LOAD 0
#endif
#ifndef RFO
#define RFO 1
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif
#ifndef WRITEBACK
#define WRITEBACK 3
#endif

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32                  // 1 of 32 sets are sampled
#define PC_IDX_SIZE 2048               // 11-bit PC index
#define OBJ_SIG_SIZE 1024              // 10-bit object signature (PC_idx ⊕ page)
#define TS_BITS 12                     // sampler epoch bits
#define TS_MASK ((1u << TS_BITS) - 1)
#define NEAR_T 64                      // near reuse threshold (sampler epochs)

#define AGE_MAX 7                      // 3-bit per-line age (0 young ... 7 old)
#define SCAN_RUN_THR 3                 // run length to assert scan clamp (≈ streaming)

// ---------------- Per-line metadata (LLC-wide) ----------------
struct LineMeta {
    uint8_t age;   // 0..AGE_MAX (NRU-like age)
    uint8_t hits;  // 0..3 demand hits seen
    uint8_t pf;    // 0/1 prefetched origin
    uint8_t cls;   // 0=never, 1=far, 2=near (predicted at fill)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set scan detection ----------------
static uint8_t  set_scan_debt[LLC_SETS];   // 0..3
static uint16_t set_last_line[LLC_SETS];   // low bits of last line addr
static int8_t   set_last_stride[LLC_SETS]; // last stride in lines (clamped)
static uint8_t  set_run_len[LLC_SETS];     // 0..3

// ---------------- OPTgen-lite sampler (only for sampled sets) ----------------
static uint16_t samp_pc_idx[LLC_SETS / SAMPLE_RATE][LLC_WAYS];  // low 11b used
static uint16_t samp_epoch  [LLC_SETS / SAMPLE_RATE][LLC_WAYS];  // TS_BITS
static uint8_t  samp_valid  [LLC_SETS / SAMPLE_RATE][LLC_WAYS];  // 0/1
static uint8_t  samp_trained[LLC_SETS / SAMPLE_RATE][LLC_WAYS];  // 0/1

// ---------------- Predictors ----------------
static uint8_t PC_RR[PC_IDX_SIZE];    // 2-bit state: 0=never,1=far,2=near (we cap at 2)
static uint8_t OBJ_DOA[OBJ_SIG_SIZE]; // 2-bit object early-death confidence

// ---------------- Epoch ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 7) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t obj_sig_from(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12); // 4KB page
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7);
    return static_cast<uint32_t>(mix) & (OBJ_SIG_SIZE - 1);
}
static inline bool is_sampled_set(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
static inline uint32_t sample_set_idx(uint32_t set) { return set / SAMPLE_RATE; }

static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }

// Scan detector update
static inline void update_scan_state(uint32_t set, uint64_t paddr) {
    uint32_t line = static_cast<uint32_t>(paddr >> 6);
    uint16_t low = static_cast<uint16_t>(line & 0xFFFF);
    int16_t stride = static_cast<int16_t>(low) - static_cast<int16_t>(set_last_line[set]);
    // clamp stride to small range
    if (stride > 7) stride = 7;
    if (stride < -8) stride = -8;

    if ((stride == set_last_stride[set]) && (stride == 1 || stride == -1)) {
        if (set_run_len[set] < 3) set_run_len[set]++;
    } else {
        if (set_run_len[set] > 0) set_run_len[set]--;
        set_last_stride[set] = static_cast<int8_t>(stride);
    }

    if (set_run_len[set] >= SCAN_RUN_THR) {
        if (set_scan_debt[set] < 2) set_scan_debt[set]++; // clamp to 2 effectively
    } else {
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
    set_last_line[set] = low;
}

static inline uint8_t clamp_cls(uint8_t s) {
    if (s > 2) return 2;
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        set_last_line[s] = 0;
        set_last_stride[s] = 0;
        set_run_len[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].age  = AGE_MAX;   // start cold
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
            meta[s][w].cls  = 1;         // conservative FAR
        }
    }
    for (uint32_t i = 0; i < (LLC_SETS / SAMPLE_RATE); i++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            samp_pc_idx[i][w] = 0;
            samp_epoch[i][w] = 0;
            samp_valid[i][w] = 0;
            samp_trained[i][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) PC_RR[i] = 1; // FAR default
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) OBJ_DOA[i] = 0;
    event_ctr = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Prefer evicting predicted-dead lines that have not yet seen a demand hit
    uint32_t best_dead = LLC_WAYS;
    uint8_t best_dead_age = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].cls == 0 && meta[set][w].hits == 0) {
            if (meta[set][w].age >= best_dead_age) {
                best_dead_age = meta[set][w].age;
                best_dead = w;
            }
        }
    }
    if (best_dead < LLC_WAYS) return best_dead;

    // Otherwise choose the oldest age (NRU-style)
    uint32_t victim = 0;
    uint8_t max_age = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].age >= max_age) {
            max_age = meta[set][w].age;
            victim = w;
        }
    }
    return victim; // replaced block index
}

// Update replacement state
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    (void)cpu;
    event_ctr++;

    // Update scan detector on every access
    update_scan_state(set, paddr);

    // Helper lambdas
    auto age_sat_inc_others = [&](uint32_t s, uint32_t accessed_way) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (w == accessed_way) continue;
            if (meta[s][w].age < AGE_MAX) meta[s][w].age++;
        }
    };

    // Sampled-set bookkeeping
    const bool is_samp = is_sampled_set(set);
    const uint32_t samp_idx = is_samp ? sample_set_idx(set) : 0;

    if (hit) {
        // Age others
        age_sat_inc_others(set, way);

        // Multi-hit promotion with scan clamp and prefetch quarantine
        LineMeta &m = meta[set][way];
        if (type == LOAD || type == RFO) {
            if (m.hits < 3) m.hits++;
            uint8_t clamp_min = (set_scan_debt[set] >= 1) ? 2 : 1;
            if (m.pf) {
                // Prefetch quarantine: require two demand hits to warm
                if (m.hits >= 2) {
                    if (m.age > clamp_min) m.age = clamp_min;
                    // keep pf bit; it only limits first promotions
                }
            } else {
                // Demand-origin: no first-hit promotion
                if (m.hits == 2) {
                    if (m.age > clamp_min) m.age = clamp_min;
                } else if (m.hits >= 3) {
                    if (m.age > 0) m.age = 0;
                }
            }
        } else {
            // Non-demand hits: do not promote
        }

        // Train OPTgen-lite on first demand hit in sampled sets
        if (is_samp && samp_valid[samp_idx][way] && !samp_trained[samp_idx][way] && (type == LOAD || type == RFO)) {
            uint16_t start = samp_epoch[samp_idx][way];
            uint16_t now = static_cast<uint16_t>(event_ctr & TS_MASK);
            uint16_t delta = static_cast<uint16_t>((now - start) & TS_MASK);
            uint16_t pcidx = samp_pc_idx[samp_idx][way];
            // classify near/far
            if (delta <= NEAR_T) {
                // near
                if (PC_RR[pcidx] < 2) PC_RR[pcidx]++;
            } else {
                // far
                if (PC_RR[pcidx] > 1) PC_RR[pcidx]--; // pull toward FAR
            }
            // Any reuse reduces object DOA for this object
            uint32_t obj = obj_sig_from(pcidx, paddr);
            sat_dec_u8(OBJ_DOA[obj]);
            samp_trained[samp_idx][way] = 1;
        }
        return;
    }

    // Miss path (fill): handle sampler eviction first
    if (is_samp) {
        if (samp_valid[samp_idx][way]) {
            uint16_t pcidx_v = samp_pc_idx[samp_idx][way];
            // If never trained (no reuse), it's dead
            if (!samp_trained[samp_idx][way]) {
                PC_RR[pcidx_v] = 0; // dead
                uint32_t obj_v = obj_sig_from(pcidx_v, victim_addr);
                sat_inc_u8(OBJ_DOA[obj_v], 3);
            }
            // Clear old sample entry
            samp_valid[samp_idx][way] = 0;
            samp_trained[samp_idx][way] = 0;
        }
    }

    // Age others before inserting the new line
    age_sat_inc_others(set, way);

    // Determine insertion for the new line
    LineMeta &nm = meta[set][way];
    nm.hits = 0;
    uint8_t pred_cls = 1; // default FAR
    uint8_t pf = (type == PREFETCH) ? 1 : 0;
    nm.pf = pf;

    if (type == WRITEBACK) {
        // Never bypass writebacks; conservative cold
        pred_cls = 1; // FAR
    } else {
        uint32_t pcidx = pc_index(PC);
        pred_cls = clamp_cls(PC_RR[pcidx]);
        if (type == PREFETCH) {
            // quarantine prefetched lines
            pred_cls = 1; // treat as FAR regardless of PC
        } else {
            if (pred_cls == 0) {
                // gate dead-bypass via object dwell (PC⊕page)
                uint32_t obj = obj_sig_from(pcidx, paddr);
                if (OBJ_DOA[obj] < 2) {
                    pred_cls = 1; // degrade to FAR until two consecutive early deaths
                }
            }
        }

        // Install a new sample entry for demand/prefetch lines in sampled sets
        if (is_samp && type != WRITEBACK) {
            samp_pc_idx[samp_idx][way] = static_cast<uint16_t>(pcidx);
            samp_epoch[samp_idx][way] = static_cast<uint16_t>(event_ctr & TS_MASK);
            samp_valid[samp_idx][way] = 1;
            samp_trained[samp_idx][way] = 0;
        }
    }

    // Map class + scan debt + prefetch quarantine to insertion age
    uint8_t ins_age;
    if (pred_cls == 0) {
        ins_age = AGE_MAX; // hard-bypass (coldest)
    } else if (pred_cls == 1) {
        ins_age = (AGE_MAX > 0) ? (AGE_MAX - 1) : AGE_MAX; // cold
    } else {
        // near: still not MRU; scan clamp pushes colder
        uint8_t base = 2;
        uint8_t add = (set_scan_debt[set] > 2) ? 2 : set_scan_debt[set];
        uint8_t a = base + add;
        if (a > (AGE_MAX - 1)) a = AGE_MAX - 1;
        ins_age = a;
    }
    // Prefetch quarantine: enforce coldest
    if (pf) ins_age = AGE_MAX;

    nm.age = ins_age;
    nm.cls = pred_cls;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}