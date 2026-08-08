#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables (refined thresholds) ----------------
#define SAMPLE_RATE 32               // sampled sets for training
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define OBJ_SIG_SIZE 1024            // 10-bit object signature (PC_idx ⊕ page)

#define SHCT_MAX 7                   // 3-bit friendliness
#define LIFE_MAX 3                   // 2-bit lifetime
#define STREAM_MAX 3                 // 2-bit stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define EAF_MAX 3                    // 2-bit eviction-age feedback

#define FRIEND_THR 3                 // SHCT >= 3 => friendly
#define SHORT_THR 2                  // LIFE_SHORT >= 2 => short-life
#define LONG_THR 2                   // LIFE_LONG >= 2 => long-life
#define STREAM_THR 2                 // PC_STREAM >= 2 => stream/scan
#define SET_STREAM_THR 2             // per-set streaminess threshold

#define DECAY_PERIOD 2048            // faster periodic decay cadence
#define PARTIAL_DECAY_CHUNK 64       // partial decay breadth

// XDIP hysteresis and dwell
#define XDIP_UP_THR 8                // margin to flip toward ins=2
#define XDIP_DN_THR 8                // margin to flip toward ins=1
#define XDIP_DWELL_EVENTS 8192       // min events between flips

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
    uint8_t rrpv; // 0..3
    uint8_t hits; // 0..3 (we use 0,1,2+)
    uint8_t pf;   // 0/1 (prefetch origin)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set streaminess and scan debt ----------------
static uint8_t set_stream[LLC_SETS];    // 0..3
static uint8_t set_scan_debt[LLC_SETS]; // 0..3 (clamped to 2 in practice)

// ---------------- Sampled-set signature table (valid|pc_idx) ----------------
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // bit15=valid, low11=pc_idx

// ---------------- PC-indexed predictors ----------------
static uint8_t SHCT[PC_IDX_SIZE];       // friendliness (3-bit)
static uint8_t LIFE_SHORT[PC_IDX_SIZE]; // short-life (2-bit)
static uint8_t LIFE_LONG[PC_IDX_SIZE];  // long-life (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];  // stream/scan score (2-bit)
static uint8_t DOA_PC[PC_IDX_SIZE];     // supplemental DOA by PC (2-bit)

// Lightweight per-PC stride detector
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low bits of last line addr
static int8_t   PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];   // 0..3

// ---------------- Object (PC⊕page) predictors ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];  // 2-bit object lifetime
static uint8_t DOA_OBJ[OBJ_SIG_SIZE];   // 2-bit object DOA confidence
static uint8_t EAF_OBJ[OBJ_SIG_SIZE];   // 2-bit eviction-age feedback

// ---------------- XDIP-Dwell state ----------------
static int16_t zcnt_ins1 = 0; // leader group preferring insertion RRPV=1
static int16_t zcnt_ins2 = 0; // leader group preferring insertion RRPV=2
static uint8_t xdip_mode = 2; // current winner (1 or 2)
static uint64_t last_xdip_flip_event = 0;

// ---------------- Epoch tracking ----------------
static uint64_t event_ctr = 0;

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // mix to 11 bits
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 7) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t obj_sig_from(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12); // 4KB page
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7);
    return static_cast<uint32_t>(mix) & (OBJ_SIG_SIZE - 1);
}
static inline bool is_sampled(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
static inline uint32_t sample_set_idx(uint32_t set) { return set / SAMPLE_RATE; }

// Leader sets (symmetric, sparse)
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

static inline void decay_predictors_partial(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = (static_cast<uint32_t>(event_ctr) + i) & (PC_IDX_SIZE - 1);
        SHCT[idx]       >>= 1;
        LIFE_SHORT[idx] >>= 1;
        LIFE_LONG[idx]  >>= 1;
        PC_STREAM[idx]  >>= 1;
        DOA_PC[idx]     >>= 1;
    }
    // Age object tables similarly
    uint32_t base = (static_cast<uint32_t>(event_ctr) + 7) & (OBJ_SIG_SIZE - 1);
    for (uint32_t i = 0; i < (n >> 2); i++) {
        uint32_t o = (base + i) & (OBJ_SIG_SIZE - 1);
        OBJ_LIFE[o] >>= 1;
        DOA_OBJ[o]  >>= 1;
        EAF_OBJ[o]  >>= 1;
    }
}

static inline void update_xdip_mode() {
    int32_t margin = static_cast<int32_t>(zcnt_ins2) - static_cast<int32_t>(zcnt_ins1);
    uint8_t next_mode = xdip_mode;
    if (margin >= XDIP_UP_THR) next_mode = 2;
    else if (margin <= -XDIP_DN_THR) next_mode = 1;
    if (next_mode != xdip_mode) {
        if (event_ctr - last_xdip_flip_event >= XDIP_DWELL_EVENTS) {
            xdip_mode = next_mode;
            last_xdip_flip_event = event_ctr;
            // light decay to avoid runaway after flip
            if (zcnt_ins1 > 0) zcnt_ins1 >>= 1;
            if (zcnt_ins2 > 0) zcnt_ins2 >>= 1;
        }
    }
}

static inline void update_stride_stream(uint32_t pc_idx, uint32_t set, uint64_t line_addr) {
    uint16_t last_line = PC_LAST_LINE[pc_idx];
    int32_t  d = static_cast<int32_t>((line_addr & 0xFFFF) - last_line);
    int8_t stride = static_cast<int8_t>(d);
    if (stride == PC_LAST_STRIDE[pc_idx]) {
        if (PC_STR_RUN[pc_idx] < 3) PC_STR_RUN[pc_idx]++;
    } else {
        PC_STR_RUN[pc_idx] = 1;
        PC_LAST_STRIDE[pc_idx] = stride;
    }
    PC_LAST_LINE[pc_idx] = static_cast<uint16_t>(line_addr & 0xFFFF);

    bool strong_stream = (PC_STR_RUN[pc_idx] >= 3);
    if (strong_stream) {
        sat_inc_u8(PC_STREAM[pc_idx], STREAM_MAX);
        if (set_stream[set] < 3) set_stream[set]++;
        if (set_scan_debt[set] < 3) set_scan_debt[set]++;
        if (set_scan_debt[set] > 2) set_scan_debt[set] = 2; // clamp
    } else {
        // bleed faster to avoid choking reuse
        if (PC_STREAM[pc_idx] > 0) PC_STREAM[pc_idx]--;
        if (set_stream[set] > 0) set_stream[set]--;
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
}

// ---------------- API functions ----------------

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_stream[s] = 0;
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3; // SRRIP default
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
        }
    }
    for (uint32_t i = 0; i < (LLC_SETS / SAMPLE_RATE); i++)
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            sample_sig[i][w] = 0;

    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[i] = 0;
        LIFE_SHORT[i] = 0;
        LIFE_LONG[i] = 0;
        PC_STREAM[i] = 0;
        DOA_PC[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 0;
        DOA_OBJ[i] = 0;
        EAF_OBJ[i] = 0;
    }
    zcnt_ins1 = 0;
    zcnt_ins2 = 0;
    xdip_mode = 2;
    last_xdip_flip_event = 0;
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
    // Return first invalid way if any
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: find a line with RRPV==3; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv >= 3) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // unreachable
    // return 0;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    event_ctr++;

    // Periodic global decay to maintain agility
    if ((event_ctr % DECAY_PERIOD) == 0) {
        decay_predictors_partial(PARTIAL_DECAY_CHUNK);
        update_xdip_mode();
    }

    uint32_t pc_idx = pc_index(PC);
    uint64_t line_addr = (paddr >> 6);
    update_stride_stream(pc_idx, set, line_addr);

    // On hit: multi-hit promotion with scan clamp and prefetch rule
    if (hit) {
        LineMeta &lm = meta[set][way];
        // Only demand accesses are eligible for promotion
        bool is_demand = (type == LOAD) || (type == RFO);
        if (is_demand) {
            // first demand hit on a prefetched line: no promotion
            if (lm.hits == 0 && lm.pf) {
                if (lm.hits < 3) lm.hits++;
                return;
            }

            // Survivor credit: if it hit after aging to >=2, reward friendliness and reduce DOA/EAF
            if (lm.rrpv >= 2) {
                sat_inc_u8(SHCT[pc_idx], SHCT_MAX);
                uint32_t o = obj_sig_from(pc_idx, paddr);
                if (DOA_OBJ[o] > 0) DOA_OBJ[o]--;
                if (EAF_OBJ[o] > 0) EAF_OBJ[o]--;
            }

            // Multi-hit promotion: 1st-><=2 (scan clamp), 2nd->1, 3rd->0
            if (lm.hits == 0) {
                if (set_scan_debt[set] > 0 || PC_STREAM[pc_idx] >= STREAM_THR) {
                    if (lm.rrpv < 2) lm.rrpv = 2;
                } else {
                    if (lm.rrpv > 1) lm.rrpv = 1;
                    else lm.rrpv = 1;
                }
            } else if (lm.hits == 1) {
                lm.rrpv = 1;
            } else {
                lm.rrpv = 0;
            }
            if (lm.hits < 3) lm.hits++;
        }
        return;
    }

    // Miss: train old line (to be evicted) before inserting the new one
    // Read old metadata before overwrite
    LineMeta old = meta[set][way];

    // SHiP/DOA/EAF/OBJ training: only if the set is sampled and we stored a PC_idx
    if (is_sampled(set)) {
        uint16_t &sig = sample_sig[sample_set_idx(set)][way];
        if (sig & 0x8000) {
            uint32_t old_pc_idx = sig & 0x07FF;
            uint32_t old_obj = obj_sig_from(old_pc_idx, victim_addr);
            // Define outcomes
            uint8_t old_hits = old.hits;
            bool reused_once = (old_hits >= 1);
            bool early_death = (old_hits < 2); // before second hit

            // Train friendliness
            if (reused_once) sat_inc_u8(SHCT[old_pc_idx], SHCT_MAX);
            else sat_dec_u8(SHCT[old_pc_idx]);

            // Train lifetime/object
            if (old_hits >= 2) {
                sat_inc_u8(LIFE_LONG[old_pc_idx], LIFE_MAX);
                if (LIFE_SHORT[old_pc_idx] > 0) LIFE_SHORT[old_pc_idx]--;
                if (OBJ_LIFE[old_obj] < LIFE_MAX) OBJ_LIFE[old_obj]++;
            } else {
                sat_inc_u8(LIFE_SHORT[old_pc_idx], LIFE_MAX);
                if (OBJ_LIFE[old_obj] < LIFE_MAX) OBJ_LIFE[old_obj]++;
            }

            // Train DOA/EAF (object-tied) and supplemental PC DOA
            if (early_death) {
                sat_inc_u8(DOA_OBJ[old_obj], DOA_MAX);
                sat_inc_u8(EAF_OBJ[old_obj], EAF_MAX);
                sat_inc_u8(DOA_PC[old_pc_idx], DOA_MAX);
            } else {
                if (DOA_OBJ[old_obj] > 0) DOA_OBJ[old_obj]--;
                if (EAF_OBJ[old_obj] > 0) EAF_OBJ[old_obj]--;
                if (DOA_PC[old_pc_idx] > 0) DOA_PC[old_pc_idx]--;
            }

            // XDIP leader feedback with magnitude weighting
            int8_t weight = 0;
            if (old_hits == 0) weight = 2;          // dead
            else if (old_hits == 1) weight = 1;     // early-death
            else weight = -1;                       // good reuse (>=2)

            bool friendly = SHCT[old_pc_idx] >= FRIEND_THR;
            bool longish  = (LIFE_LONG[old_pc_idx] >= LONG_THR) || (OBJ_LIFE[old_obj] >= LONG_THR);

            if (friendly && longish) {
                if (is_leader_ins1(set)) {
                    if (weight > 0) { for (int i = 0; i < weight; i++) sat_inc_s16(zcnt_ins1, 1024); }
                    else { for (int i = 0; i < (-weight); i++) sat_dec_s16(zcnt_ins1, -1024); }
                } else if (is_leader_ins2(set)) {
                    if (weight > 0) { for (int i = 0; i < weight; i++) sat_inc_s16(zcnt_ins2, 1024); }
                    else { for (int i = 0; i < (-weight); i++) sat_dec_s16(zcnt_ins2, -1024); }
                }
            }

            // Invalidate sampled tag (will be set on new fill)
            sig = 0;
        }
    }

    // Compute insertion state for the new line
    LineMeta &lm = meta[set][way];
    lm.hits = 0;
    lm.pf = (type == PREFETCH) ? 1 : 0;

    uint32_t o_sig = obj_sig_from(pc_idx, paddr);
    bool friendly = (SHCT[pc_idx] >= FRIEND_THR);
    bool likely_long = (LIFE_LONG[pc_idx] >= LONG_THR) || (OBJ_LIFE[o_sig] >= LONG_THR);
    bool likely_short = (LIFE_SHORT[pc_idx] >= SHORT_THR) || (OBJ_LIFE[o_sig] >= SHORT_THR);
    bool set_streamy = (set_stream[set] >= SET_STREAM_THR);
    bool pc_streamy  = (PC_STREAM[pc_idx] >= STREAM_THR);
    bool streamish   = set_streamy || pc_streamy;

    bool doa_cold = (DOA_OBJ[o_sig] >= 2);           // require two consecutive early-deaths
    bool eaf_strong = (EAF_OBJ[o_sig] >= 2) || (DOA_PC[pc_idx] >= DOA_MAX);

    uint8_t ins_rrpv = 2; // default neutral

    if (type == WRITEBACK) {
        ins_rrpv = 1; // never bypass writebacks
    } else if (streamish) {
        ins_rrpv = 3; // scan-resistant cold insertion
    } else if (doa_cold && !friendly && likely_short) {
        ins_rrpv = 3; // guarded cold-bypass for dead objects
    } else if (friendly && likely_long) {
        // XDIP-controlled warm insertion, but respect EAF colding and scan clamp
        ins_rrpv = (xdip_mode == 1) ? 1 : 2;
        if (eaf_strong && ins_rrpv < 2) ins_rrpv = 2;      // colder if EAF is strong
        if (set_scan_debt[set] > 0 && ins_rrpv < 2) ins_rrpv = 2; // clamp under scan
    } else {
        ins_rrpv = likely_short ? 3 : 2;
    }

    // Install new line
    lm.rrpv = ins_rrpv;

    // Record signature in sampled sets for future training
    if (is_sampled(set)) {
        sample_sig[sample_set_idx(set)][way] = static_cast<uint16_t>(0x8000 | (pc_idx & 0x07FF));
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