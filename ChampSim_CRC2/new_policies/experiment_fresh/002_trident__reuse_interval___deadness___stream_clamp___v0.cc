#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Access types (CRC2 defines if not present) ----------------
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
#define SAMPLE_RATE 32                 // 1-of-32 sampled sets
#define PC_IDX_SIZE 2048               // 11-bit PC index
#define OBJ_SIG_SIZE 1024              // 10-bit PC⊕page signature
#define STREAM_RUN_THR 2               // >=2 consecutive +1 strides => streaming
#define DECAY_PERIOD 4096              // periodic decay cadence

// ---------------- Per-line metadata (no RRIP) ----------------
struct LineMeta {
    uint8_t ttl;     // 0..3 survival priority
    uint8_t mhits;   // 0..3 demand hit count
    uint8_t pf;      // 0/1 prefetched origin
    uint8_t stream;  // 0/1 stream-clamped (no promotion)
    uint8_t lru;     // 0..15 recency bucket (0=MRU)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3 (scan/stream pressure)

// ---------------- PC-indexed predictors ----------------
static uint8_t  PC_UTIL[PC_IDX_SIZE];      // 0..7 utility (higher => more reuse)
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low bits of last line number
static uint8_t  PC_RUN[PC_IDX_SIZE];       // 0..3 stride(+1) run length

// ---------------- Object (PC⊕page) lifetime ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];     // 0..3 (longer life => higher)

// ---------------- Sampled shadow tags (Hawkeye-like last-touch trainer) -----
struct ShadowEntry {
    uint8_t  valid;
    uint16_t pc_idx;     // 11-bit effective
    uint16_t line_sig;   // low 16 bits of line number
    uint16_t obj_sig;    // 10-bit effective
};
static ShadowEntry shadow_tbl[LLC_SETS / SAMPLE_RATE][LLC_WAYS];
static uint8_t shadow_rr[LLC_SETS / SAMPLE_RATE];

// ---------------- Epoch counter ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 7) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t obj_sig_from(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12);
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7);
    return static_cast<uint32_t>(mix) & (OBJ_SIG_SIZE - 1);
}
static inline bool is_sampled(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
static inline uint32_t sample_set_idx(uint32_t set) { return set / SAMPLE_RATE; }

static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }

static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }
static inline bool is_demandish(uint32_t type) { return (type == LOAD) || (type == RFO) || (type == PREFETCH); }

// periodic, bounded decay to avoid runaway confidence
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of PC_UTIL and PC_RUN
    uint32_t base = static_cast<uint32_t>(event_ctr) & (PC_IDX_SIZE - 1);
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (base + i) & (PC_IDX_SIZE - 1);
        if (PC_UTIL[idx] > 0) PC_UTIL[idx]--;
        if (PC_RUN[idx] > 0) PC_RUN[idx]--;
    }

    // Bleed scan debt sparsely across sets
    for (uint32_t s = 0; s < LLC_SETS; s += 64) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }

    // Age object life lightly (helps phase changes)
    uint32_t obase = (static_cast<uint32_t>(event_ctr >> 1)) & (OBJ_SIG_SIZE - 1);
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t o = (obase + i) & (OBJ_SIG_SIZE - 1);
        if (OBJ_LIFE[o] > 0) OBJ_LIFE[o]--;
    }
}

// Update stream sentinel; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (is_demandish(type)) {
        uint16_t cur = static_cast<uint16_t>(line & 0xFFFF);
        uint16_t prev = PC_LAST_LINE[pc_idx];
        int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(prev);
        if (stride == 1) sat_inc_u8(PC_RUN[pc_idx], 3);
        else if (PC_RUN[pc_idx] > 0) PC_RUN[pc_idx]--;
        PC_LAST_LINE[pc_idx] = cur;

        if (PC_RUN[pc_idx] >= STREAM_RUN_THR) sat_inc_u8(set_scan_debt[set], 3);
        else if ((event_ctr & 0xF) == 0) { // bleed quickly after scans
            if (set_scan_debt[set] > 0) set_scan_debt[set]--;
        }
    }
    if (type == WRITEBACK) return false;
    return (PC_RUN[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
}

static inline void lru_touch_to_pos(uint32_t set, uint32_t way, uint8_t new_pos) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        uint8_t p = meta[set][w].lru;
        if (p <= new_pos) meta[set][w].lru = p + 1; // push slightly older
    }
    meta[set][way].lru = new_pos;
}

static inline uint32_t score_line(const LineMeta &m) {
    // Higher score => better eviction candidate
    uint32_t s = 0;
    s += m.lru;                         // 0..15 (older is larger)
    s += (3 - m.ttl) * 4;               // favor low TTL
    s += (m.mhits == 0 ? 8 : (m.mhits == 1 ? 4 : 0)); // prefer single/no-hit
    if (m.stream) s += 6;               // stream-clamped lines go first
    if (m.pf) s += 3;                   // prefetched lines are weaker
    return s;
}

static inline void sampler_on_access(uint32_t set, uint64_t line, uint32_t pc_idx, uint32_t obj_sig) {
    if (!is_sampled(set)) return;
    uint16_t sig = static_cast<uint16_t>(line & 0xFFFF);
    uint32_t ss = sample_set_idx(set);

    // Reuse before eviction?
    for (uint32_t i = 0; i < LLC_WAYS; i++) {
        ShadowEntry &e = shadow_tbl[ss][i];
        if (e.valid && e.line_sig == sig) {
            // Reuse observed
            sat_inc_u8(PC_UTIL[e.pc_idx], 7);
            sat_inc_u8(OBJ_LIFE[e.obj_sig], 3);
            e.valid = 0;
            break;
        }
    }
}

static inline void sampler_on_fill(uint32_t set, uint64_t line, uint32_t pc_idx, uint32_t obj_sig) {
    if (!is_sampled(set)) return;
    uint32_t ss = sample_set_idx(set);
    uint8_t idx = shadow_rr[ss] % LLC_WAYS;
    shadow_rr[ss] = (shadow_rr[ss] + 1) % LLC_WAYS;

    ShadowEntry &vict = shadow_tbl[ss][idx];
    if (vict.valid) {
        // Died without reuse
        sat_dec_u8(PC_UTIL[vict.pc_idx]);
        sat_dec_u8(OBJ_LIFE[vict.obj_sig]);
    }
    vict.valid = 1;
    vict.pc_idx = static_cast<uint16_t>(pc_idx & (PC_IDX_SIZE - 1));
    vict.line_sig = static_cast<uint16_t>(line & 0xFFFF);
    vict.obj_sig = static_cast<uint16_t>(obj_sig & (OBJ_SIG_SIZE - 1));
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].ttl = 1;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].lru = static_cast<uint8_t>(w); // initial stack
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_UTIL[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 0;
    }
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        shadow_rr[ss] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            shadow_tbl[ss][w].valid = 0;
            shadow_tbl[ss][w].pc_idx = 0;
            shadow_tbl[ss][w].line_sig = 0;
            shadow_tbl[ss][w].obj_sig = 0;
        }
    }
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

    // Score-based victim selection
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t s = score_line(meta[set][w]);
        if (w == 0 || s > best_score) {
            best_score = s;
            best_way = w;
        }
    }
    return best_way; // replaced block index
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
    event_ctr++;
    periodic_decay();

    uint64_t line = (paddr >> 6);
    uint32_t pc_idx = pc_index(PC);
    uint32_t obj_sig = obj_sig_from(pc_idx, paddr);

    // Streaming sentinel update and sampling on every access
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);
    sampler_on_access(set, line, pc_idx, obj_sig);

    LineMeta &m = meta[set][way];

    if (hit) {
        // Demand hits control promotion; first hit only warms (no promotion)
        if (is_demand(type)) {
            // Do not promote stream-clamped lines; keep them weak
            if (m.stream) {
                if (m.mhits < 3) m.mhits++;
                if (m.ttl > 0) m.ttl--; // bias to die quickly
            } else {
                if (m.pf && (m.mhits == 0)) {
                    // First demand hit of a prefetched line => no promotion
                    m.mhits = 1;
                    // Slight warm but not MRU
                    if (m.ttl < 2) m.ttl++;
                } else {
                    if (m.mhits == 0) {
                        m.mhits = 1;           // multi-hit gating
                        if (m.ttl < 2) m.ttl++; // light warm
                        // leave LRU position unchanged on first hit
                    } else {
                        if (m.mhits < 3) m.mhits++;
                        m.ttl = 3;            // fully protect after second hit
                        m.pf = 0;             // graduated to demand-resident
                        lru_touch_to_pos(set, way, 0); // MRU
                    }
                }
            }
            // reinforce object life on observed reuse
            sat_inc_u8(OBJ_LIFE[obj_sig], 3);
        } else {
            // Non-demand hits: keep quarantine
            if (m.pf) {
                if (m.ttl > 0) m.ttl--;
            }
        }
    } else {
        // Miss fill: initialize metadata according to predictions
        sampler_on_fill(set, line, pc_idx, obj_sig);

        bool dead_pred = (PC_UTIL[pc_idx] <= 1);
        bool is_pf = (type == PREFETCH);
        bool is_wb = (type == WRITEBACK);

        uint8_t ins_pos;
        uint8_t ins_ttl;

        if (is_wb) {
            // Never bypass writebacks; moderate insertion
            ins_pos = 7;
            ins_ttl = 2;
        } else if (is_pf) {
            // Quarantine prefetches strongly
            ins_pos = 15;
            ins_ttl = 0;
        } else if (stream_pred) {
            // Aggressive soft-bypass for detected streams
            ins_pos = 15;
            ins_ttl = 0;
        } else if (dead_pred) {
            // Predicted-dead PC => tail insertion with no promotion
            ins_pos = 14;
            ins_ttl = 0;
        } else {
            // Normal demand insertion with phase-aware TTL
            uint8_t olife = OBJ_LIFE[obj_sig];
            ins_pos = 4;                           // not MRU; allows exploration
            ins_ttl = (olife >= 2) ? 2 : 1;        // stabilize object-friendly phases
        }

        m.ttl = ins_ttl;
        m.mhits = 0;
        m.pf = is_pf ? 1 : 0;
        m.stream = (stream_pred && !is_wb) ? 1 : 0;
        lru_touch_to_pos(set, way, ins_pos);
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