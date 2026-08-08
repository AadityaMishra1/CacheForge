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
#define SAMPLE_RATE 32                 // 1-of-32 sampled sets (64 sample sets total)
#define PC_IDX_SIZE 2048               // 11-bit PC index
#define OBJ_SIG_SIZE 1024              // 10-bit PC⊕page signature
#define STREAM_RUN_THR 2               // >=2 consecutive +1 strides => streaming
#define DECAY_PERIOD 4096              // periodic decay cadence
#define DEAD_THR 1                     // SHiP 2-bit: <=1 considered cold/dead

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t ttl;     // 0..3 survival priority
    uint8_t mhits;   // 0..3 demand hit count
    uint8_t pf;      // 0/1 prefetched origin
    uint8_t stream;  // 0/1 stream-clamped (no promotion)
    uint8_t lru;     // 0..15 recency (0=MRU)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3 (scan/stream pressure)

// ---------------- PC-indexed predictors ----------------
static uint8_t  PC_UTIL[PC_IDX_SIZE];      // 2-bit SHiP usefulness (0..3)
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low 16 bits of last line number
static uint8_t  PC_RUN[PC_IDX_SIZE];       // 0..3 stride(+1) run length

// ---------------- Object (PC⊕page) lifetime ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];     // 0..3 (longer life => higher)

// ---------------- Sampled shadow tags (Hawkeye-like last-touch trainer) -----
struct ShadowEntry {
    uint8_t  valid;     // 1
    uint8_t  used;      // 1 (reused before eviction)
    uint16_t line_sig;  // 16-bit block signature
    uint16_t pc_idx;    // 11-bit effective (packed in 16)
};
static ShadowEntry shadow_tbl[LLC_SETS / SAMPLE_RATE][LLC_WAYS];
static uint8_t shadow_rr[LLC_SETS / SAMPLE_RATE]; // 0..15 RR pointer

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

// LRU helpers
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru < old)
            meta[set][w].lru++;
        if (meta[set][w].lru > (LLC_WAYS - 1))
            meta[set][w].lru = (LLC_WAYS - 1);
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
static inline void lru_make_lru(uint32_t set, uint32_t way) {
    lru_insert_pos(set, way, LLC_WAYS - 1);
}

// periodic, bounded decay to avoid runaway confidence
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of PC predictors
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
    if (type == WRITEBACK) return false;

    uint16_t cur = static_cast<uint16_t>(line & 0xFFFF);
    uint16_t prev = PC_LAST_LINE[pc_idx];
    int32_t stride = static_cast<int32_t>(cur) - static_cast<int32_t>(prev);
    if (stride == 1) {
        sat_inc_u8(PC_RUN[pc_idx], 3);
    } else {
        if (PC_RUN[pc_idx] > 0) PC_RUN[pc_idx]--;
    }
    PC_LAST_LINE[pc_idx] = cur;

    if (PC_RUN[pc_idx] >= STREAM_RUN_THR) sat_inc_u8(set_scan_debt[set], 3);
    else if ((event_ctr & 0xF) == 0) { // bleed quickly after scans
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }

    return (PC_RUN[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
}

// Victim scoring: higher is better candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += m.lru;                                 // oldness
    s += (3 - m.ttl) * 4;                       // low TTL first
    s += (m.mhits == 0 ? 6 : (m.mhits == 1 ? 3 : 0)); // discourage multi-hit
    if (m.stream) s += 8;                       // stream-clamped first
    if (m.pf) s += 2;                           // prefetch is weaker
    return s;
}

// Sampler: train on hit and on new sample insertion
static inline void sampler_on_hit(uint32_t set, uint64_t line) {
    if (!is_sampled(set)) return;
    uint32_t ss = sample_set_idx(set);
    uint16_t sig = static_cast<uint16_t>(line & 0xFFFF);
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        ShadowEntry &e = shadow_tbl[ss][w];
        if (e.valid && e.line_sig == sig && e.used == 0) {
            e.used = 1;
            uint16_t idx = e.pc_idx & (PC_IDX_SIZE - 1);
            if (PC_UTIL[idx] < 3) PC_UTIL[idx]++;
            break;
        }
    }
}
static inline void sampler_on_fill(uint32_t set, uint64_t line, uint32_t pc_idx) {
    if (!is_sampled(set)) return;
    uint32_t ss = sample_set_idx(set);
    uint8_t &rr = shadow_rr[ss];
    // evict old sample
    ShadowEntry &old = shadow_tbl[ss][rr];
    if (old.valid && old.used == 0) {
        uint16_t idx = old.pc_idx & (PC_IDX_SIZE - 1);
        if (PC_UTIL[idx] > 0) PC_UTIL[idx]--;
    }
    // insert new
    ShadowEntry &ne = shadow_tbl[ss][rr];
    ne.valid = 1;
    ne.used = 0;
    ne.line_sig = static_cast<uint16_t>(line & 0xFFFF);
    ne.pc_idx = static_cast<uint16_t>(pc_idx & (PC_IDX_SIZE - 1));
    rr = (uint8_t)((rr + 1) & (LLC_WAYS - 1)); // 0..15
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w] = {1, 0, 0, 0, (uint8_t)w}; // mild TTL, cold, ordered LRU
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
            shadow_tbl[ss][w] = {0, 0, 0, 0};
        }
    }
    event_ctr = 0;
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
    // Score-based victim among valid lines
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
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    event_ctr++;
    periodic_decay();

    uint64_t line = (paddr >> 6);
    uint32_t pci = pc_index(PC);
    uint32_t obj = obj_sig_from(pci, paddr);

    // Update stream sentinel and get prediction
    bool stream_pred = update_stream_sentinel(set, pci, line, type);

    // Train sampler on LLC hit before any metadata promotion
    if (hit) {
        sampler_on_hit(set, line);
    }

    LineMeta &m = meta[set][way];

    if (hit) {
        // On hit: handle multi-hit gating and promotions
        if (type != WRITEBACK) {
            if (!m.stream) {
                // Update hits only for demand hits; prefetch-first-hit is gated
                if (is_demand(type)) {
                    if (m.mhits < 3) m.mhits++;
                    if (m.mhits >= 2) {
                        // Confirmed multi-hit: full protection and MRU
                        m.ttl = 3;
                        lru_make_mru(set, way);
                        if (OBJ_LIFE[obj] < 3) OBJ_LIFE[obj]++;
                    } else {
                        // First demand hit: warm only, modest TTL bump, no MRU thrash
                        if (m.ttl < 2) m.ttl = 2;
                        // gentle recency nudge: place near MRU but not stack-top
                        lru_insert_pos(set, way, 2);
                    }
                } else {
                    // Prefetch hit: do not promote on first touch
                    // leave mhits as-is to require later demand confirmation
                }
            } else {
                // Stream-clamped lines never promote; keep them easy victims
                // but still count hits to possibly inform object life elsewhere
                if (is_demand(type) && OBJ_LIFE[obj] < 3) OBJ_LIFE[obj]++;
            }
        }
    } else {
        // Miss: new insertion (never bypass WRITEBACK)
        bool pf = (type == PREFETCH);
        bool is_wb = (type == WRITEBACK);

        // SHiP usefulness for this PC
        uint8_t util = PC_UTIL[pci];

        // Default insertion: guarded (older half)
        uint8_t ins_ttl = 1;
        uint8_t ins_pos = 12;
        uint8_t ins_stream = 0;

        if (!is_wb) {
            if (stream_pred) {
                // Hard stream clamp: emulate bypass via immediate-evictable tail insert
                ins_ttl = 0;
                ins_pos = 15;
                ins_stream = 1;
            } else if (util <= DEAD_THR) {
                // Predicted-dead PC: tail insert with low TTL
                ins_ttl = 0;
                ins_pos = 15;
                ins_stream = 0;
            } else {
                // Warm PC: use object life to slightly bias TTL
                uint8_t life = OBJ_LIFE[obj];
                if (life >= 2 && ins_ttl < 2) ins_ttl = 2;
                ins_pos = 12;
                ins_stream = 0;
            }
        } else {
            // Writeback insert: low priority but not stream
            ins_ttl = 1;
            ins_pos = 14;
            ins_stream = 0;
        }

        // Program per-line metadata
        m.ttl = ins_ttl;
        m.mhits = 0;
        m.pf = pf ? 1 : 0;
        m.stream = ins_stream ? 1 : 0;
        lru_insert_pos(set, way, ins_pos);

        // Sampler learns last-touch utility (LOAD/RFO/PREFETCH only)
        if (!is_wb) {
            sampler_on_fill(set, line, pci);
        }
    }

    // TTL aging on accesses: lines with ttl>0 slowly decay with recency aging
    // To avoid infinite aging loops, just clamp on accesses; victim scoring prefers low TTL.

    (void)set; (void)way; (void)paddr; (void)PC; (void)type; (void)hit;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}