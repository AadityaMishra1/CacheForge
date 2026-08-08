#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32               // sampled sets for SHiP training
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define OBJ_SIG_SIZE 1024            // 10-bit object signature (PC_idx ⊕ page)

// Predictor ranges
#define SHCT_MAX 7                   // 3-bit friendliness
#define LIFE_MAX 3                   // 2-bit lifetime
#define STREAM_MAX 3                 // 2-bit stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define EAF_MAX 3                    // 2-bit eviction-age feedback

// Thresholds
#define FRIEND_THR 3
#define SHORT_THR 2
#define LONG_THR 2
#define STREAM_THR 2
#define SET_STREAM_THR 2

// Decay
#define DECAY_PERIOD 2048
#define PARTIAL_DECAY_CHUNK 64

// XDIP hysteresis and dwell
#define XDIP_UP_THR 8
#define XDIP_DN_THR 8
#define XDIP_DWELL_EVENTS 8192

// Phase banking
#define PHASE_EPOCH 131072          // coarse dwell to toggle banks

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
    uint8_t hits; // 0..3
    uint8_t pf;   // 0/1 (prefetch origin)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set streaminess and scan debt ----------------
static uint8_t set_stream[LLC_SETS];      // 0..3
static uint8_t set_scan_debt[LLC_SETS];   // 0..2 normally; effective clamp may reach 3 via EAF
static uint8_t alloc_was_invalid[LLC_SETS]; // flag latched in GetVictimInSet

// ---------------- Sampled-set signature table (valid|bank|pc_idx) ----------------
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // bit15=valid, bit14=bank, low11=pc_idx

// ---------------- PC-indexed predictors (banked SHCT/LIFE) ----------------
static uint8_t SHCT[2][PC_IDX_SIZE];        // friendliness (3-bit)
static uint8_t LIFE_SHORT[2][PC_IDX_SIZE];  // short-life (2-bit)
static uint8_t LIFE_LONG[2][PC_IDX_SIZE];   // long-life (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];      // stream/scan score (2-bit)
static uint8_t DOA_NEG[PC_IDX_SIZE];        // PC-negative shortlist (2-bit)

// Lightweight per-PC stride detector
static uint16_t PC_LAST_LINE[PC_IDX_SIZE];  // low bits of last line addr
static int8_t   PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];    // 0..3

// ---------------- Object (PC⊕page) predictors ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];  // 2-bit object lifetime
static uint8_t DOA_OBJ[OBJ_SIG_SIZE];   // 2-bit object DOA confidence
static uint8_t EAF_OBJ[OBJ_SIG_SIZE];   // 2-bit eviction-age feedback

// ---------------- XDIP-Dwell state ----------------
static int16_t zcnt_ins1 = 0; // leader group preferring insertion RRPV=1
static int16_t zcnt_ins2 = 0; // leader group preferring insertion RRPV=2
static uint8_t xdip_mode = 2; // current winner (1 or 2)
static uint64_t last_xdip_flip_event = 0;

// ---------------- Phase bank state ----------------
static uint8_t phase_bank = 0;              // 0/1
static uint64_t last_phase_flip_event = 0;

// ---------------- Epoch tracking ----------------
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
        SHCT[0][idx]       >>= 1;
        SHCT[1][idx]       >>= 1;
        LIFE_SHORT[0][idx] >>= 1;
        LIFE_SHORT[1][idx] >>= 1;
        LIFE_LONG[0][idx]  >>= 1;
        LIFE_LONG[1][idx]  >>= 1;
        PC_STREAM[idx]     >>= 1;
        DOA_NEG[idx]       >>= 1;
    }
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
            if (zcnt_ins1 > 0) zcnt_ins1 >>= 1;
            if (zcnt_ins2 > 0) zcnt_ins2 >>= 1;
        }
    }
}

static inline void update_phase_bank() {
    if (event_ctr - last_phase_flip_event >= PHASE_EPOCH) {
        phase_bank ^= 1u;
        last_phase_flip_event = event_ctr;
    }
}

static inline void stride_stream_update(uint32_t set, uint32_t pc_idx, uint64_t paddr) {
    uint16_t line = static_cast<uint16_t>((paddr >> 6) & 0xFFFF);
    int16_t delta = static_cast<int16_t>(line) - static_cast<int16_t>(PC_LAST_LINE[pc_idx]);
    int8_t stride = 0;
    if (delta > -64 && delta < 64) stride = static_cast<int8_t>(delta);

    if (stride != 0 && stride == PC_LAST_STRIDE[pc_idx]) {
        sat_inc_u8(PC_STR_RUN[pc_idx], 3);
    } else {
        // decay run if pattern breaks
        sat_dec_u8(PC_STR_RUN[pc_idx]);
        PC_LAST_STRIDE[pc_idx] = stride;
    }
    PC_LAST_LINE[pc_idx] = line;

    if (PC_STR_RUN[pc_idx] >= 2) {
        sat_inc_u8(PC_STREAM[pc_idx], STREAM_MAX);
        sat_inc_u8(set_stream[set], STREAM_MAX);
        if (set_scan_debt[set] < 2) sat_inc_u8(set_scan_debt[set], 2);
    } else {
        sat_dec_u8(PC_STREAM[pc_idx]);
        if (set_stream[set] > 0) sat_dec_u8(set_stream[set]);
        // bleed scan debt quickly when streaminess subsides
        if (set_scan_debt[set] > 0 && PC_STR_RUN[pc_idx] == 0) sat_dec_u8(set_scan_debt[set]);
    }
}

static inline uint8_t effective_scan_clamp(uint32_t set, uint32_t obj_sig) {
    uint8_t clamp = set_scan_debt[set];           // 0..2
    if (EAF_OBJ[obj_sig] >= 2) {                  // EAF-driven escalation
        if (clamp < 3) clamp++;
    }
    if (clamp > 3) clamp = 3;
    return clamp; // 0..3
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_stream[s] = 0;
        set_scan_debt[s] = 0;
        alloc_was_invalid[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3;
            meta[s][w].hits = 0;
            meta[s][w].pf = 0;
        }
    }
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) sample_sig[ss][w] = 0;
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[0][i] = SHCT_MAX / 2;
        SHCT[1][i] = SHCT_MAX / 2;
        LIFE_SHORT[0][i] = 0;
        LIFE_SHORT[1][i] = 0;
        LIFE_LONG[0][i] = 0;
        LIFE_LONG[1][i] = 0;
        PC_STREAM[i] = 0;
        DOA_NEG[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t o = 0; o < OBJ_SIG_SIZE; o++) {
        OBJ_LIFE[o] = 0;
        DOA_OBJ[o] = 0;
        EAF_OBJ[o] = 0;
    }
    zcnt_ins1 = 0; zcnt_ins2 = 0; xdip_mode = 2; last_xdip_flip_event = 0;
    phase_bank = 0; last_phase_flip_event = 0;
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
    // If any invalid way exists, return it immediately and latch flag
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            alloc_was_invalid[set] = 1;
            return w;
        }
    }
    alloc_was_invalid[set] = 0;

    // SRRIP victim: seek RRPV==3; if none, age once and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
        // loop will terminate because all counters saturate at 3
    }
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

    // Periodic predictor decay, XDIP and phase-bank updates
    if ((event_ctr % DECAY_PERIOD) == 0) decay_predictors_partial(PARTIAL_DECAY_CHUNK);
    update_xdip_mode();
    update_phase_bank();

    uint32_t pc_idx = pc_index(PC);
    stride_stream_update(set, pc_idx, paddr);
    uint32_t obj_sig = obj_sig_from(pc_idx, paddr);
    uint8_t clamp = effective_scan_clamp(set, obj_sig);

    if (hit) {
        // Multi-hit promotion with scan clamp and prefetch-first-hit suppression
        uint8_t &rrpv = meta[set][way].rrpv;
        uint8_t &hits = meta[set][way].hits;

        bool is_demand = (type == LOAD) || (type == RFO);
        if (meta[set][way].pf && hits == 0 && is_demand) {
            // First demand hit on a prefetched line: no promotion, clear pf
            meta[set][way].pf = 0;
        } else {
            if (hits == 0) {
                if (clamp >= 2) { if (rrpv > 2) rrpv = 2; }
                else rrpv = (rrpv > 1) ? 1 : rrpv;
            } else if (hits == 1) {
                if (clamp >= 3) { if (rrpv > 2) rrpv = 2; }
                else rrpv = (rrpv > 1) ? 1 : rrpv;
            } else {
                if (clamp >= 3) { if (rrpv > 1) rrpv = 1; }
                else rrpv = 0;
            }
        }
        if (hits < 3) hits++;
        // Any reuse clears PC-negative shortcut and softens DOA/EAF
        DOA_NEG[pc_idx] = 0;
        sat_dec_u8(DOA_OBJ[obj_sig]);
        sat_dec_u8(EAF_OBJ[obj_sig]);
        return;
    }

    // Miss path: train on eviction (if any), then insert
    // Train victim (previous line in [set][way]) if it was valid
    if (!alloc_was_invalid[set]) {
        // Train SHiP and LIFE from sampled sets using stored (valid|bank|pc_idx)
        if (is_sampled(set)) {
            uint16_t &sig = sample_sig[sample_set_idx(set)][way];
            if (sig & (1u << 15)) {
                uint8_t vbank = (sig >> 14) & 0x1;
                uint32_t vpc = static_cast<uint32_t>(sig & ((1u << 11) - 1));
                if (meta[set][way].hits == 0) {
                    if (SHCT[vbank][vpc] > 0) SHCT[vbank][vpc]--;
                    if (LIFE_SHORT[vbank][vpc] < LIFE_MAX) LIFE_SHORT[vbank][vpc]++;
                    if (LIFE_LONG[vbank][vpc] > 0) LIFE_LONG[vbank][vpc]--;
                    // PC-negative shortlist tracks consecutive DOA
                    if (DOA_NEG[vpc] < DOA_MAX) DOA_NEG[vpc]++;
                } else {
                    if (SHCT[vbank][vpc] < SHCT_MAX) SHCT[vbank][vpc]++;
                    if (LIFE_LONG[vbank][vpc] < LIFE_MAX) LIFE_LONG[vbank][vpc]++;
                    if (LIFE_SHORT[vbank][vpc] > 0) LIFE_SHORT[vbank][vpc]--;
                    DOA_NEG[vpc] = 0;
                }
                sig = 0; // clear
            }
        }
        // Train object DOA/EAF and XDIP
        uint32_t vpc_for_obj = pc_idx; // victim_addr's PC not known; object-only updates use addr
        (void)vpc_for_obj;
        uint32_t vobj = obj_sig_from(pc_index(PC), victim_addr);
        if (meta[set][way].hits == 0) {
            sat_inc_u8(DOA_OBJ[vobj], DOA_MAX);
            sat_inc_u8(EAF_OBJ[vobj], EAF_MAX);
            sat_inc_u8(OBJ_LIFE[vobj], LIFE_MAX);
            if (is_leader_ins1(set)) sat_inc_s16(zcnt_ins1, 32767);
            if (is_leader_ins2(set)) sat_inc_s16(zcnt_ins2, 32767);
        } else {
            sat_dec_u8(DOA_OBJ[vobj]);
            sat_dec_u8(EAF_OBJ[vobj]);
            if (OBJ_LIFE[vobj] > 0) OBJ_LIFE[vobj]--;
            if (is_leader_ins1(set)) sat_dec_s16(zcnt_ins1, -32768);
            if (is_leader_ins2(set)) sat_dec_s16(zcnt_ins2, -32768);
        }
    }

    // Choose insertion RRPV
    bool is_demand = (type == LOAD) || (type == RFO);
    bool is_pf = (type == PREFETCH);
    bool cold_bypass = false;

    // PC-/Object-based DOA gating: immediate cold insertion after two quick DOAs
    if (type != WRITEBACK) {
        if (DOA_NEG[pc_idx] >= 2 || DOA_OBJ[obj_sig] >= 2) cold_bypass = true;
    }

    bool pc_streamy = (PC_STREAM[pc_idx] >= STREAM_THR);
    bool set_streamy = (set_stream[set] >= SET_STREAM_THR);
    bool friendly = (SHCT[phase_bank][pc_idx] >= FRIEND_THR);
    bool longlife = (LIFE_LONG[phase_bank][pc_idx] >= LONG_THR);
    bool shortlist_short = (LIFE_SHORT[phase_bank][pc_idx] >= SHORT_THR) || (OBJ_LIFE[obj_sig] >= SHORT_THR);

    uint8_t ins_rrpv = 2; // default
    if (type == WRITEBACK) {
        ins_rrpv = 2; // never bypass on writeback
    } else if (cold_bypass) {
        ins_rrpv = 3;
    } else if (pc_streamy || set_streamy || shortlist_short) {
        ins_rrpv = 3;
    } else if (friendly && longlife) {
        ins_rrpv = (xdip_mode == 1) ? 1 : 2;
        if (clamp >= 2 && ins_rrpv < 2) ins_rrpv = 2; // stabilize under scans
    } else {
        ins_rrpv = 2;
    }

    // Install new line
    meta[set][way].rrpv = ins_rrpv;
    meta[set][way].hits = 0;
    meta[set][way].pf = is_pf ? 1 : 0;

    // Record signature for sampled sets (with bank)
    if (is_sampled(set)) {
        uint16_t sig = 0;
        sig |= (1u << 15);                       // valid
        sig |= (static_cast<uint16_t>(phase_bank & 0x1) << 14); // bank at install
        sig |= static_cast<uint16_t>(pc_idx & ((1u << 11) - 1)); // pc_idx
        sample_sig[sample_set_idx(set)][way] = sig;
    }

    // Done
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}