#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32              // 1-of-32 sampled sets for utility/object training
#define PC_IDX_SIZE 2048            // 11-bit PC index
#define OBJ_SIG_SIZE 1024           // 10-bit object signature (PC_idx ⊕ page)
#define DECAY_PERIOD 4096
#define STREAM_RUN_THR 2            // >=2 consecutive +1 strides => streaming
#define SCAN_DEBT_MAX 3

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

// ---------------- Per-line metadata (compact; non-RRIP) ----------------
struct LineMeta {
    uint8_t ttl : 2;     // 0..3: survival priority (0=dead soon, 3=long)
    uint8_t mhits : 2;   // demand hit count (capped at 3)
    uint8_t pf : 1;      // filled by prefetch
    uint8_t stream : 1;  // streaming-clamped (no promotion)
    uint8_t lru : 4;     // 0..15: recency bucket (lower is newer)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS]; // 0..3

// ---------------- PC-indexed state ----------------
static uint8_t PC_UTIL[PC_IDX_SIZE];       // 0..7 utility (saturating)
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // last line# low bits
static uint8_t PC_LAST_STRIDE[PC_IDX_SIZE];// 0:other, 1:+1 stride
static uint8_t PC_RUN[PC_IDX_SIZE];        // 0..3 consecutive +1 strides

// ---------------- Object (PC⊕page) lifetime ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];     // 0..3 (2-bit)

// ---------------- Sampled-set training entries ----------------
struct SampleEntry {
    uint16_t pc_idx;   // 11 bits effective
    uint16_t obj_sig;  // 10 bits effective
    uint8_t  mode;     // 0:none, 1:ins1, 2:ins2 (leader annotation)
    uint8_t  valid;
};
static SampleEntry sample_tbl[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

// ---------------- Leader dueling for default TTL depth ----------------
static int16_t zcnt_ins1 = 0;
static int16_t zcnt_ins2 = 0;
static uint8_t xdip_mode = 2; // 1 or 2
static uint64_t event_ctr = 0;
static uint64_t last_flip_evt = 0;

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

static inline bool is_leader_ins1(uint32_t set) {
    uint32_t m = set & 63;
    return (m == 1) || (m == 9) || (m == 17) || (m == 25);
}
static inline bool is_leader_ins2(uint32_t set) {
    uint32_t m = set & 63;
    return (m == 33) || (m == 41) || (m == 49) || (m == 57);
}

static inline void sat_inc_u8(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_s16(int16_t &x, int16_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_s16(int16_t &x, int16_t minv) { if (x > minv) x--; }

// bounded periodic decay
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;
    // decay a stripe of PC_UTIL and PC_RUN
    uint32_t base = static_cast<uint32_t>(event_ctr) & (PC_IDX_SIZE - 1);
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t idx = (base + i) & (PC_IDX_SIZE - 1);
        if (PC_UTIL[idx] > 0) PC_UTIL[idx]--;
        if (PC_RUN[idx] > 0) PC_RUN[idx]--;
    }
    // bleed scan debt sparsely to recover from scans
    for (uint32_t s = 0; s < LLC_SETS; s += 256) {
        if (set_scan_debt[s] > 0) set_scan_debt[s]--;
    }
    // light hysteresis update for dueling
    int32_t margin = static_cast<int32_t>(zcnt_ins2) - static_cast<int32_t>(zcnt_ins1);
    uint8_t desired = (margin >= 4 ? 2 : (margin <= -4 ? 1 : xdip_mode));
    if (desired != xdip_mode && (event_ctr - last_flip_evt) >= (DECAY_PERIOD << 2)) {
        xdip_mode = desired;
        last_flip_evt = event_ctr;
        // dampen counters after flips
        zcnt_ins1 >>= 1;
        zcnt_ins2 >>= 1;
    }
}

// Streaming sentinel update (per access)
static inline bool update_stream_sentinel(uint32_t set, uint32_t pc_idx, uint64_t line, uint32_t type) {
    // never classify writebacks as streaming
    bool is_demandish = (type == LOAD) || (type == RFO) || (type == PREFETCH);
    bool seqp1 = false;
    if (is_demandish) {
        int32_t stride = static_cast<int32_t>((line & 0xFFFF) - PC_LAST_LINE[pc_idx]);
        if (stride == 1) {
            seqp1 = true;
            PC_LAST_STRIDE[pc_idx] = 1;
            sat_inc_u8(PC_RUN[pc_idx], 3);
        } else {
            PC_LAST_STRIDE[pc_idx] = 0;
            if (PC_RUN[pc_idx] > 0) PC_RUN[pc_idx]--;
        }
        PC_LAST_LINE[pc_idx] = static_cast<uint16_t>(line & 0xFFFF);
        if (PC_RUN[pc_idx] >= STREAM_RUN_THR) {
            sat_inc_u8(set_scan_debt[set], SCAN_DEBT_MAX);
        } else if ((event_ctr & 0xF) == 0) {
            if (set_scan_debt[set] > 0) set_scan_debt[set]--;
        }
    }
    bool stream_pred = (PC_RUN[pc_idx] >= STREAM_RUN_THR) || (set_scan_debt[set] >= 2);
    if (type == WRITEBACK) stream_pred = false;
    return stream_pred;
}

static inline uint32_t score_line(const LineMeta &m) {
    // Higher score => better eviction candidate
    // old (lru high) + low TTL + single/no-hit lines are favored for eviction
    uint32_t ttl_pen = (3 - m.ttl) * 16;      // 0,16,32,48
    uint32_t hit_pen = (m.mhits == 0) ? 8 : 0;// 8 if never hit
    return static_cast<uint32_t>(m.lru) + ttl_pen + hit_pen;
}

// ---------------- API ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w] = {2, 0, 0, 0, static_cast<uint8_t>(w)}; // ttl=2 seed, increasing lru
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        PC_UTIL[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 0;
    }
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_tbl[ss][w] = {0, 0, 0, 0};
        }
    }
    zcnt_ins1 = 0; zcnt_ins2 = 0; xdip_mode = 2; event_ctr = 0; last_flip_evt = 0;
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
    // Otherwise choose max score candidate
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    bool best_inited = false;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (!best_inited || sc > best_score) {
            best_inited = true;
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

    // streaming sentinel update (per access)
    uint32_t pc_idx = pc_index(PC);
    uint64_t line = (paddr >> 6);
    bool stream_pred = update_stream_sentinel(set, pc_idx, line, type);

    // decay housekeeping
    periodic_decay();

    // On hit: multi-hit promotion with clamps
    if (hit) {
        // Demand vs prefetch
        bool is_demand = (type == LOAD) || (type == RFO);
        // age others (bounded)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (w != way && meta[set][w].lru < 15) meta[set][w].lru++;
        }
        meta[set][way].lru = 0;

        if (is_demand) {
            if (meta[set][way].mhits < 3) meta[set][way].mhits++;
            // No promotion on first demand hit or for streaming-clamped lines
            if (!meta[set][way].stream && meta[set][way].mhits >= 2 && set_scan_debt[set] < 2) {
                if (meta[set][way].ttl < 3) meta[set][way].ttl++;
            }
        }
        // Prefetch hits do not promote
        return;
    }

    // Miss (install/replace): train on the victim (old content at 'way')
    // Train only on sampled sets to save storage
    if (is_sampled(set) && sample_tbl[sample_set_idx(set)][way].valid) {
        const SampleEntry &se = sample_tbl[sample_set_idx(set)][way];
        bool zero_reuse = (meta[set][way].mhits == 0);
        // PC utility: +1 on reuse, -1 on dead
        if (zero_reuse) {
            if (PC_UTIL[se.pc_idx] > 0) PC_UTIL[se.pc_idx]--;
        } else {
            sat_inc_u8(PC_UTIL[se.pc_idx], 7);
        }
        // Object life
        if (zero_reuse) {
            if (OBJ_LIFE[se.obj_sig] > 0) OBJ_LIFE[se.obj_sig]--;
        } else if (meta[set][way].mhits >= 2) {
            sat_inc_u8(OBJ_LIFE[se.obj_sig], 3);
        }
        // Leader dueling feedback
        if (se.mode == 1) {
            if (zero_reuse) sat_inc_s16(zcnt_ins1, 1024);
            else sat_dec_s16(zcnt_ins1, -1024);
        } else if (se.mode == 2) {
            if (zero_reuse) sat_inc_s16(zcnt_ins2, 1024);
            else sat_dec_s16(zcnt_ins2, -1024);
        }
    }

    // Install new line metadata
    uint32_t obj = obj_sig_from(pc_idx, paddr);
    uint8_t base_ttl;

    // Default insertion depth chosen by dueling (1 or 2), modulated by object life
    uint8_t duel_depth = (xdip_mode == 2) ? 2 : 1;

    // Streaming "soft-bypass": tail insert, never promote
    bool soft_bypass = stream_pred && (type != WRITEBACK);

    if (soft_bypass) {
        base_ttl = 0; // dead soon
    } else {
        // Dead PCs (low utility) get cold insertion
        bool pc_dead = (PC_UTIL[pc_idx] <= 1);
        if (pc_dead && (type != WRITEBACK)) base_ttl = 0;
        else {
            base_ttl = duel_depth;
            if (OBJ_LIFE[obj] >= 2 && base_ttl < 3) base_ttl++; // longer if object shows life
            if (type == PREFETCH && base_ttl > 1) base_ttl = 1; // quarantine prefetches
        }
    }

    // Set initial LRU bucket (older if colder)
    uint8_t ins_lru = 8;
    if (base_ttl >= 3) ins_lru = 4;
    else if (base_ttl == 2) ins_lru = 6;
    else if (base_ttl == 1) ins_lru = 10;
    else ins_lru = 14;

    meta[set][way].ttl = base_ttl & 0x3;
    meta[set][way].mhits = 0;
    meta[set][way].pf = (type == PREFETCH) ? 1 : 0;
    meta[set][way].stream = soft_bypass ? 1 : 0;
    meta[set][way].lru = ins_lru;

    // Record training info only for sampled sets
    if (is_sampled(set)) {
        SampleEntry &se = sample_tbl[sample_set_idx(set)][way];
        se.pc_idx = static_cast<uint16_t>(pc_idx);
        se.obj_sig = static_cast<uint16_t>(obj);
        se.mode = static_cast<uint8_t>(is_leader_ins1(set) ? 1 : (is_leader_ins2(set) ? 2 : 0));
        se.valid = 1;
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