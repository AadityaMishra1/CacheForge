#include <cstdint>
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
#define PC_COLD_SIZE 1024            // 1K-entry PC cold table
#define PC_COLD_BITS 3               // 3-bit saturating counters
#define STREAM_PC_SIZE 1024          // per-PC stream sentinel entries
#define LAST_LINE_BITS 12            // track low 12 bits of line number for stride detection
#define SAMPLE_STRIDE 8              // 1/8 set sampling for per-line fill PC signatures
#define SAMPLE_SETS (LLC_SETS / SAMPLE_STRIDE)
#define DECAY_PERIOD 1024            // periodic decay cadence (fast to be phase-safe)

// ---------------- Per-line metadata (packed conceptually to 10 bits) ----------------
struct LineMeta {
    uint8_t lru;      // 0..15 (recency bucket; 0=MRU)
    uint8_t ttl;      // 0..3 (survival priority)
    uint8_t mhits;    // 0..3 (demand-hit count; multi-hit gating)
    uint8_t pf;       // 0/1 (filled by prefetch)
    uint8_t stream;   // 0/1 (stream-clamped: never promote)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Sampled per-line fill PC signatures (10-bit) ----------------
static uint16_t fill_sig[SAMPLE_SETS][LLC_WAYS]; // only for sampled sets (10-bit indices)
static inline int32_t sample_index(int32_t set) {
    return ((set & (SAMPLE_STRIDE - 1)) == 0) ? (set / SAMPLE_STRIDE) : -1;
}

// ---------------- PC coldness predictor (3-bit per 10-bit PC signature) ----------------
static uint8_t pc_cold[PC_COLD_SIZE]; // 0=hot ... 7=strong cold

// ---------------- Per-PC streaming sentinel ----------------
static uint16_t pc_last_line[STREAM_PC_SIZE]; // LAST_LINE_BITS of line#
static uint8_t  pc_run[STREAM_PC_SIZE];       // 0..3 forward +1 stride run

// ---------------- Per-set stream pressure (0..3) ----------------
static uint8_t set_pressure[LLC_SETS];

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); } // 64B lines

static inline uint32_t pc_sig10(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 17) ^ (pc << 13);
    x ^= (x >> 7);
    return static_cast<uint32_t>(x) & (PC_COLD_SIZE - 1);
}
static inline uint32_t stream_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 9) ^ (pc << 3);
    return static_cast<uint32_t>(x) & (STREAM_PC_SIZE - 1);
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

// Victim scoring (higher is better victim)
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                                        // older preferred
    s += (3 - m.ttl) * 8;                              // low TTL preferred
    s += (m.mhits == 0) ? 6 : ((m.mhits == 1) ? 3 : 0); // zero/single-hit preferred
    s += m.pf ? 6 : 0;                                 // prefetch quarantined
    s += m.stream ? 8 : 0;                             // stream-clamped preferred
    return s;
}

// periodic light decay for pc_cold, stream runs, and set pressure
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of pc_cold
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr + i) & (PC_COLD_SIZE - 1);
        if (pc_cold[idx] > 0) pc_cold[idx]--;
    }
    // Bleed stream runs sparsely
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (decay_ptr + (i * 3)) & (STREAM_PC_SIZE - 1);
        if (pc_run[idx] > 0) pc_run[idx]--;
    }
    // Bleed set pressure sparsely
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t s = (decay_ptr + (i * 7)) & (LLC_SETS - 1);
        if (set_pressure[s] > 0) set_pressure[s]--;
    }
    decay_ptr++;
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
        }
        set_pressure[s] = 0;
    }
    for (uint32_t i = 0; i < PC_COLD_SIZE; i++) pc_cold[i] = 3; // neutral start
    for (uint32_t i = 0; i < STREAM_PC_SIZE; i++) {
        pc_last_line[i] = 0;
        pc_run[i] = 0;
    }
    for (uint32_t s = 0; s < SAMPLE_SETS; s++)
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            fill_sig[s][w] = 0;
    event_ctr = 0;
    decay_ptr = 0;
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

    // Choose highest-scoring victim
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (w == 0 || sc > best_score) {
            best_score = sc;
            best_way = w;
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
    (void)cpu;
    event_ctr++;
    periodic_decay();

    // Update per-PC stream sentinel with current access (except writeback)
    if (type != WRITEBACK) {
        uint32_t sidx = stream_index(PC);
        uint16_t mask = (1u << LAST_LINE_BITS) - 1u;
        uint16_t line_low = static_cast<uint16_t>(line_number(paddr) & mask);
        uint16_t prev = pc_last_line[sidx];
        uint16_t delta = static_cast<uint16_t>((line_low - prev) & mask);
        uint8_t run = pc_run[sidx];
        if (delta == 1) {
            if (run < 3) run++;
        } else if (line_low != prev) {
            if (run > 0) run--;
        }
        pc_last_line[sidx] = line_low;
        pc_run[sidx] = run;

        // Adjust set stream pressure quickly when sequential
        if (run >= 2) {
            if (set_pressure[set] < 3) set_pressure[set]++;
        } else if (set_pressure[set] > 0) {
            set_pressure[set]--;
        }
    }

    int32_t samp = sample_index(set);

    if (hit) {
        // On hit: gated promotion. Only promote on second demand hit and not stream-clamped.
        LineMeta &m = meta[set][way];

        if (is_demand(type)) {
            if (m.mhits < 3) m.mhits++;
            if (!m.stream) {
                if (m.mhits >= 2) {
                    // confirmed reusable: promote and extend TTL
                    lru_make_mru(set, way);
                    if (m.ttl < 3) m.ttl++;
                } // else: do not promote on first demand hit
            } // stream-clamped never promote
        } else {
            // Prefetch hit or others: never promote on non-demand
            // Keep quarantine for prefetches
        }
        return;
    }

    // Miss path (fill): train predictor using the evicted line (victim) before overwrite
    if (samp >= 0) {
        // Victim's PC signature and outcomes
        uint16_t vsig = fill_sig[samp][way];
        if (vsig < PC_COLD_SIZE) {
            LineMeta &vm = meta[set][way];
            // Zero-hit strongly penalized, single-hit mildly penalized, multi-hit rewarded
            if (vm.mhits == 0) {
                pc_cold[vsig] = (pc_cold[vsig] + 2 > 7) ? 7 : (pc_cold[vsig] + 2);
            } else if (vm.mhits == 1) {
                pc_cold[vsig] = (pc_cold[vsig] + 1 > 7) ? 7 : (pc_cold[vsig] + 1);
            } else {
                pc_cold[vsig] = (pc_cold[vsig] > 1) ? (pc_cold[vsig] - 2) : 0;
            }
        }
    }

    // Decide insertion policy for the incoming fill
    LineMeta &mnew = meta[set][way];
    uint8_t new_pf = (type == PREFETCH) ? 1 : 0;

    // Streaming classification
    uint8_t run_now = 0;
    if (type != WRITEBACK) {
        run_now = pc_run[stream_index(PC)];
    }

    // PC cold prediction
    uint32_t sig = pc_sig10(PC);
    uint8_t cold_ctr = pc_cold[sig];
    bool strong_cold = (cold_ctr >= 5); // strong coldness threshold

    // Default initialization
    mnew.lru = LLC_WAYS - 1;
    mnew.ttl = 1;
    mnew.mhits = 0;
    mnew.pf = new_pf;
    mnew.stream = 0;

    // WRITEBACKs: never bypass; place modestly (mid-stack), not stream
    if (type == WRITEBACK) {
        mnew.ttl = 2;
        mnew.stream = 0;
        mnew.pf = 0;
        lru_insert_pos(set, way, 4);
    } else {
        // Streaming controls
        bool clamp_stream = (run_now >= 2);
        bool hard_stream = (run_now >= 3) || (set_pressure[set] >= 3);

        if (hard_stream) {
            // Approximate full bypass: LRU insert, no promotion ever
            mnew.ttl = 0;
            mnew.stream = 1;
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else if (clamp_stream) {
            // Quarantine stream: LRU insert, no promotion
            mnew.ttl = 0;
            mnew.stream = 1;
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else if (new_pf) {
            // Prefetch quarantine regardless of coldness
            mnew.ttl = 0;
            mnew.stream = 0;
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else if (strong_cold) {
            // Cold PC gating: tail insert with no immediate priority
            mnew.ttl = 0;
            mnew.stream = 0;
            lru_insert_pos(set, way, LLC_WAYS - 1);
        } else {
            // Normal demand insertion: modest priority
            mnew.ttl = 2;
            mnew.stream = 0;
            lru_insert_pos(set, way, 6);
        }
    }

    // Record fill PC signature for future eviction training (sampled sets only)
    if (samp >= 0) {
        fill_sig[samp][way] = static_cast<uint16_t>(sig);
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