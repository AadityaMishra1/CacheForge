#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32                 // 1-of-32 sampled sets
#define PC_IDX_SIZE 2048               // 11-bit index
#define OBJ_SIG_SIZE 1024              // 10-bit signature
#define DECAY_PERIOD 4096              // periodic decay of predictors
#define STREAM_WEAK_THR 2              // >=2 consecutive +1 lines -> weak stream
#define STREAM_STRONG_THR 3            // >=3 consecutive +1 lines -> strong stream
#define SCAN_DEBT_MAX 3                // per-set scan debt saturation
#define UTIL_COLD_THR 1                // PC utility <=1 considered cold

// ---------------- Access types ----------------
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

// ---------------- Per-line metadata ----------------
struct LineMeta {
    uint8_t ttl   : 2;  // 0..3
    uint8_t mhits : 2;  // 0..3 (demand-hit count)
    uint8_t pf    : 1;  // filled by prefetch
    uint8_t stream: 1;  // streaming clamp (no promotion)
    uint8_t lru   : 4;  // 0..15, lower is newer
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed state ----------------
static uint8_t  PC_UTIL[PC_IDX_SIZE];     // 3-bit utility (0..7)
static uint16_t PC_LAST_LINE[PC_IDX_SIZE];// last line index (low 16)
static uint8_t  PC_RUN[PC_IDX_SIZE];      // 0..3 (+1-line run length)

// ---------------- Object (PC⊕page) lifetime ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];    // 0..3

// ---------------- Sampled-set training entries ----------------
struct SampleEntry {
    uint16_t pc_idx;  // 11 bits effective
    uint16_t obj_sig; // 10 bits effective
    uint8_t  valid;   // 1 bit
    uint8_t  reused;  // 1 bit
};
static SampleEntry sample_tbl[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

// ---------------- Leader dueling for default insertion TTL ----------------
static int16_t zcnt_ins1 = 0; // score for TTL1 leaders
static int16_t zcnt_ins2 = 0; // score for TTL2 leaders
static uint8_t xdip_mode = 2; // 1 or 2 (default insertion TTL)
static uint64_t event_ctr = 0;
static uint64_t last_flip_evt = 0;

// ---------------- Helpers ----------------
static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_s16(int16_t &x, int16_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_s16(int16_t &x, int16_t minv) { if (x > minv) x--; }

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

static inline bool is_leader_ins1(uint32_t set) {
    uint32_t m = set & 63;
    return (m == 1) || (m == 9) || (m == 17) || (m == 25);
}
static inline bool is_leader_ins2(uint32_t set) {
    uint32_t m = set & 63;
    return (m == 33) || (m == 41) || (m == 49) || (m == 57);
}

static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a small stripe of PC predictors
    uint32_t base = static_cast<uint32_t>(event_ctr) & (PC_IDX_SIZE - 1);
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (base + i) & (PC_IDX_SIZE - 1);
        if (PC_UTIL[idx] > 0) PC_UTIL[idx]--;
        if (PC_RUN[idx] > 0) PC_RUN[idx]--;
    }
    // Bleed scan debt sparsely to recover after scans
    for (uint32_t s = 0; s < LLC_SETS; s += 256) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    // Leader dueling hysteresis
    int32_t margin = static_cast<int32_t>(zcnt_ins2) - static_cast<int32_t>(zcnt_ins1);
    uint8_t desired = (margin >= 8 ? 2 : (margin <= -8 ? 1 : xdip_mode));
    if (desired != xdip_mode && (event_ctr - last_flip_evt) >= (DECAY_PERIOD << 2)) {
        xdip_mode = desired;
        last_flip_evt = event_ctr;
        zcnt_ins1 >>= 1;
        zcnt_ins2 >>= 1;
    }
}

static inline void lru_to_mru(uint32_t set, uint32_t way) {
    uint8_t pos = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].lru < pos) meta[set][w].lru++;
    }
    meta[set][way].lru = 0;
}

static inline uint32_t score_line(const LineMeta &m) {
    // Older + low TTL + single/no-hit + prefetch/stream get evicted first
    uint32_t s = 0;
    s += m.lru;                            // 0..15
    s += (3 - m.ttl) << 4;                 // 0,16,32,48
    s += (m.mhits == 0 ? 8 : (m.mhits == 1 ? 4 : 0)); // 0/4/8
    s += (m.pf ? 6 : 0);
    s += (m.stream ? 4 : 0);
    return s;
}

static inline void train_on_eviction(uint32_t set, uint32_t way) {
    if (!is_sampled(set)) return;
    SampleEntry &e = sample_tbl[sample_set_idx(set)][way];
    if (e.valid) {
        // Train PC utility and object life
        if (e.reused) {
            sat_inc_u8(PC_UTIL[e.pc_idx], 7);
            sat_inc_u8(OBJ_LIFE[e.obj_sig], 3);
            if (is_leader_ins1(set)) sat_inc_s16(zcnt_ins1, 32767);
            if (is_leader_ins2(set)) sat_inc_s16(zcnt_ins2, 32767);
        } else {
            if (PC_UTIL[e.pc_idx] > 0) PC_UTIL[e.pc_idx]--;
            if (OBJ_LIFE[e.obj_sig] > 0) OBJ_LIFE[e.obj_sig]--;
            if (is_leader_ins1(set)) sat_dec_s16(zcnt_ins1, -32768);
            if (is_leader_ins2(set)) sat_dec_s16(zcnt_ins2, -32768);
        }
    }
    e.valid = 0; e.reused = 0;
}

static inline void update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return; // don't classify writebacks
    int32_t stride = static_cast<int32_t>((line & 0xFFFFull) - PC_LAST_LINE[pc_idx]);
    if (stride == 1) {
        sat_inc_u8(PC_RUN[pc_idx], 3);
        // escalate scan debt when we observe a run
        sat_inc_u8(set_scan_debt[set], SCAN_DEBT_MAX);
    } else if (stride != 0) {
        // reset on non-repeating stride
        PC_RUN[pc_idx] = 0;
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
    PC_LAST_LINE[pc_idx] = static_cast<uint16_t>(line & 0xFFFFull);
}

static inline uint8_t stream_level(uint32_t set, uint32_t pc_idx) {
    // 0: none, 1: weak, 2: strong
    if (PC_RUN[pc_idx] >= STREAM_STRONG_THR || set_scan_debt[set] >= SCAN_DEBT_MAX) return 2;
    if (PC_RUN[pc_idx] >= STREAM_WEAK_THR || set_scan_debt[set] >= 2) return 1;
    return 0;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w] = {0, 0, 0, 0, static_cast<uint8_t>(w)};
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_UTIL[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) OBJ_LIFE[i] = 0;
    for (uint32_t s = 0; s < (LLC_SETS / SAMPLE_RATE); s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_tbl[s][w] = {0, 0, 0, 0};
        }
    }
    zcnt_ins1 = zcnt_ins2 = 0;
    xdip_mode = 2;
    event_ctr = 0;
    last_flip_evt = 0;
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Choose the highest score (best eviction candidate)
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

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = (paddr >> 6);
    update_stream_sentinel(set, pc_idx, line, type);
    uint8_t slevel = stream_level(set, pc_idx);
    bool demand = (type == LOAD) || (type == RFO);
    bool prefetch = (type == PREFETCH);
    bool wb = (type == WRITEBACK);

    if (hit) {
        // On hit: recency + demand hit accounting
        if (demand) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;
            // Unlock promotion on 2nd/3rd demand hit if not streaming-clamped
            if (!meta[set][way].stream) {
                if (meta[set][way].mhits >= 3) meta[set][way].ttl = 3;
                else if (meta[set][way].mhits >= 2 && meta[set][way].ttl < 2) meta[set][way].ttl = 2;
            }
            // Reinforce object life on true reuse
            uint32_t osig = obj_sig_from(pc_idx, paddr);
            sat_inc_u8(OBJ_LIFE[osig], 3);
        }
        // Never promote on prefetch hits (they don't increment mhits)
        lru_to_mru(set, way);

        // Mark reused in sampled sets
        if (is_sampled(set)) {
            SampleEntry &e = sample_tbl[sample_set_idx(set)][way];
            if (e.valid) e.reused = 1;
        }
        return;
    }

    // Miss: train evicted line if sampled
    train_on_eviction(set, way);

    // Install new line metadata
    LineMeta &m = meta[set][way];
    m.pf = prefetch ? 1 : 0;
    m.stream = 0;
    m.mhits = 0;
    // Decide insertion TTL
    uint8_t ttl = 1; // base
    if (wb) {
        // never bypass writebacks; modest retention
        ttl = 1;
        m.stream = 0;
    } else if (prefetch) {
        // quarantine prefetches at tail, never promote
        ttl = 0;
        m.stream = 1;
    } else {
        // Demand fill
        bool pc_cold = (PC_UTIL[pc_idx] <= UTIL_COLD_THR);
        if (slevel >= 2) {
            // strong stream -> practical bypass: dead-on-arrival
            ttl = 0;
            m.stream = 1;
        } else if (slevel == 1) {
            // weak stream -> tail insert and clamp
            ttl = 0;
            m.stream = 1;
        } else if (pc_cold) {
            // dead-block prediction -> cold insert
            ttl = 0;
            m.stream = 0;
        } else {
            // default guided by leader dueling + object life
            uint8_t base = (xdip_mode == 2 ? 2 : 1);
            uint32_t osig = obj_sig_from(pc_idx, paddr);
            uint8_t boost = (OBJ_LIFE[osig] >= 2) ? 1 : 0;
            ttl = static_cast<uint8_t>(base + boost);
            if (ttl > 3) ttl = 3;
            m.stream = 0;
        }
    }
    m.ttl = ttl;
    // Writebacks get a head-start on reuse; others start at 0
    m.mhits = wb ? 1 : 0;

    // Recency placement: new line is MRU (still tail-biased by TTL in victim selection)
    lru_to_mru(set, way);

    // Record sample info for future training
    if (is_sampled(set)) {
        SampleEntry &e = sample_tbl[sample_set_idx(set)][way];
        e.pc_idx = static_cast<uint16_t>(pc_idx);
        e.obj_sig = static_cast<uint16_t>(obj_sig_from(pc_idx, paddr));
        e.valid = 1;
        e.reused = 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}