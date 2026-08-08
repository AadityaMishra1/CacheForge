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

// Tunables
#define PC_IDX_SIZE 2048
#define FOOT_SIG_SIZE 2048
#define LAST_LINE_BITS 12
#define DECAY_PERIOD 4096
#define SIG_SAMP_MASK 7  // 1/8 sets sampled for per-line signature feedback

// Per-line metadata
struct LineMeta {
    uint8_t lru;      // 0..15 (0=MRU)
    uint8_t ttl;      // 0..3 (RRPV-like)
    uint8_t mhits;    // 0..3 (demand hit count)
    uint8_t pf;       // 0/1 (prefetch filled)
    uint8_t stream;   // 0/1 (stream-clamped: no promotion)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// Per-line PC signature (used only in sampled sets for predictor feedback)
static uint16_t sig_pc[LLC_SETS][LLC_WAYS]; // information-theoretically 11 bits, but typed for simplicity

// PC coldness predictor (2-bit saturating)
static uint8_t pc_cold[PC_IDX_SIZE]; // 0=strong cold ... 3=hot

// Footprint streaming sentinel (PC⊕page)
static uint16_t foot_last_line[FOOT_SIG_SIZE]; // 12-bit line id per signature
static uint8_t  foot_run[FOOT_SIG_SIZE];       // 0..3 forward +1 stride run

// Per-set stream quota (throttles "hard" streaming tail-inserts)
static uint8_t stream_quota[LLC_SETS]; // info-theoretically 2 bits (0..3)

// Time/decay
static uint64_t event_ctr = 0;
static uint16_t decay_ptr_pc = 0;
static uint32_t decay_ptr_set = 0;

// Helpers
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t foot_sig(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12);
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7) ^ (page << 13);
    return static_cast<uint32_t>(mix) & (FOOT_SIG_SIZE - 1);
}
static inline bool is_sampled_set(uint32_t set) {
    return ((set & SIG_SAMP_MASK) == 0);
}

// periodic decay/refill
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_cold (cool down)
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (decay_ptr_pc + i) & (PC_IDX_SIZE - 1);
        if (pc_cold[idx] > 0) pc_cold[idx]--;
    }
    // Bleed footprint runs sparsely
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr_pc + (i << 1)) & (FOOT_SIG_SIZE - 1);
        if (foot_run[idx] > 0) foot_run[idx]--;
    }
    // Refill a small stripe of per-set stream quota
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t s = (decay_ptr_set + i);
        if (s >= LLC_SETS) break;
        if (stream_quota[s] < 3) stream_quota[s]++;
    }
    decay_ptr_pc = (decay_ptr_pc + 1) & (PC_IDX_SIZE - 1);
    decay_ptr_set += 32;
    if (decay_ptr_set >= LLC_SETS) decay_ptr_set = 0;
}

// Update footprint sentinel; return current run (0..3) after update
static inline uint8_t update_stream_sentinel(uint32_t fsig, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return foot_run[fsig];
    uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
    uint16_t cur = static_cast<uint16_t>(line) & mask;
    uint16_t prev = foot_last_line[fsig];
    uint16_t delta = static_cast<uint16_t>((cur - prev) & mask);

    if (delta == 1) {
        if (foot_run[fsig] < 3) foot_run[fsig]++;
    } else if (cur != prev) {
        if (foot_run[fsig] > 0) foot_run[fsig]--;
    }
    foot_last_line[fsig] = cur;
    return foot_run[fsig];
}

// LRU helpers
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru < old) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = 0;
}
static inline void lru_insert_pos(uint32_t set, uint32_t way, uint8_t pos) {
    if (pos >= LLC_WAYS) pos = LLC_WAYS - 1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru <= pos) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = pos;
}

// Victim scoring (higher -> better victim)
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                            // older preferred
    s += (3 - m.ttl) * 6;                  // low TTL preferred
    s += (m.mhits == 0 ? 6 : (m.mhits == 1 ? 2 : 0)); // single/no-hit preferred
    s += m.pf ? 5 : 0;                     // prefetch quarantined
    s += m.stream ? 7 : 0;                 // stream-clamped
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = w;
            meta[s][w].ttl = 1;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            sig_pc[s][w] = 0;
        }
        stream_quota[s] = 3; // allow a few "hard" stream tail-inserts per set
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) pc_cold[i] = 1; // slightly cold start
    for (uint32_t i = 0; i < FOOT_SIG_SIZE; i++) {
        foot_last_line[i] = 0;
        foot_run[i] = 0;
    }
    event_ctr = 0;
    decay_ptr_pc = 0;
    decay_ptr_set = 0;
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
    // Prefer any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Score-based victim among valid ways
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    bool inited = false;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (!inited || sc > best_score) {
            best_score = sc;
            best_way = w;
            inited = true;
        }
    }
    return best_way;
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
    (void)cpu; (void)victim_addr;
    event_ctr++;
    periodic_decay();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = line_number(paddr);
    uint32_t fsig = foot_sig(pc_idx, paddr);
    uint8_t run = update_stream_sentinel(fsig, line, type); // refresh run every access

    if (hit) {
        // On hit: demand-only increments; first demand hit does NOT promote
        if (is_demand(type)) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;
            // Prefetch quarantine lifted only by demand (but still gated)
            meta[set][way].pf = 0;

            // Adaptive gating: hot PCs promote on 2nd hit, cold on 3rd
            uint8_t threshold = (pc_cold[pc_idx] >= 2) ? 2 : 3;
            if (!meta[set][way].stream && meta[set][way].mhits >= threshold) {
                // Strong keep
                lru_make_mru(set, way);
                if (meta[set][way].ttl < 3) meta[set][way].ttl = 3;
                // Reward PC
                if (pc_cold[pc_idx] < 3) pc_cold[pc_idx]++;
            } else {
                // Keep near tail; slightly increase TTL for first reuse
                if (meta[set][way].ttl < 2) meta[set][way].ttl++;
                // Do not change LRU position (no promotion on first hit)
            }
        } else {
            // Prefetch hit: never unlock promotion
            // keep at tail, keep pf=1; damp TTL
            if (meta[set][way].ttl > 0) meta[set][way].ttl--;
        }
        return;
    }

    // Miss fill: provide SHiP-like feedback from evicted line (sampled sets only)
    if (is_sampled_set(set)) {
        uint32_t old_sig = sig_pc[set][way] & (PC_IDX_SIZE - 1);
        bool reused = (meta[set][way].mhits > 0);
        if (reused) {
            if (pc_cold[old_sig] < 3) pc_cold[old_sig]++;
        } else {
            if (pc_cold[old_sig] > 0) pc_cold[old_sig]--;
        }
    }

    // Decide insertion policy for the new line
    bool is_stream_soft = (type != WRITEBACK) && (run >= 2);
    bool is_stream_hard = (type != WRITEBACK) && (run >= 3) && (stream_quota[set] > 0);

    LineMeta &m = meta[set][way];
    m.mhits = 0;
    m.pf = (type == PREFETCH) ? 1 : 0;
    m.stream = (is_stream_soft || is_stream_hard) ? 1 : 0;

    if (type == WRITEBACK) {
        // Never bypass on writeback; give moderate TTL, near-tail insert
        m.ttl = 2;
        lru_insert_pos(set, way, LLC_WAYS - 2);
    } else if (is_stream_hard) {
        // Emulated hard BYPASS: deepest TTL, strict LRU, no promotion ever
        m.ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
        if (stream_quota[set] > 0) stream_quota[set]--;
    } else if (is_stream_soft) {
        // Soft streaming: deepest TTL, strict LRU, no promotion
        m.ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
    } else {
        // Dead-block gating via PC coldness
        uint8_t cold = pc_cold[pc_idx];
        if (cold <= 1) {
            // predicted cold -> deepest insert
            m.ttl = 0;
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else {
            // predicted warm/hot -> slightly better insert
            m.ttl = 1;
            lru_insert_pos(set, way, LLC_WAYS - 2);
        }
    }

    // Record signature in sampled sets for future feedback
    if (is_sampled_set(set)) {
        sig_pc[set][way] = static_cast<uint16_t>(pc_idx);
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