#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
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
#define LIFE_MAX 3                   // 2-bit lifetime (short/long banks)
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

// Phase blending epoch
#define PHASE_EPOCH 131072          // coarse dwell to adjust blend weight

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
    uint8_t hits; // 0..3 (demand hits counted, regardless of prefetch)
    uint8_t pf;   // 0/1 (prefetch origin)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set scan debt (scan clamp driver) ----------------
static uint8_t set_scan_debt[LLC_SETS];   // 0..3

// ---------------- Sampled-set signature table (valid|bank|pc_idx) ----------------
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // bit15=valid, bit14=bank, low11=pc_idx

// ---------------- PC-indexed predictors (banked SHCT/LIFE) ----------------
static uint8_t SHCT[2][PC_IDX_SIZE];        // friendliness (3-bit)
static uint8_t LIFE_SHORT[2][PC_IDX_SIZE];  // short-life (2-bit)
static uint8_t LIFE_LONG[2][PC_IDX_SIZE];   // long-life (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];      // stream/scan score (2-bit)
static uint8_t DOA_NEG[PC_IDX_SIZE];        // PC-negative shortlist (2-bit)

// Lightweight per-PC stride detector
static uint16_t PC_LAST_LINE[PC_IDX_SIZE];  // low 16b of last line addr
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

// ---------------- Phase blending state ----------------
static uint8_t blend_w = 4;            // 0..8 weight for bank1 (bank0 gets 8-blend_w)
static int16_t phase_score[2] = {0,0}; // correctness margin per bank
static uint64_t last_phase_event = 0;

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

static inline void update_phase_blend_epoch() {
    if (event_ctr - last_phase_event >= PHASE_EPOCH) {
        int32_t margin = static_cast<int32_t>(phase_score[1]) - static_cast<int32_t>(phase_score[0]);
        if (margin > 4 && blend_w < 8) blend_w++;
        else if (margin < -4 && blend_w > 0) blend_w--;
        // hysteresis: damp scores
        phase_score[0] >>= 1;
        phase_score[1] >>= 1;
        last_phase_event = event_ctr;
    }
}

static inline uint8_t blended_friend_score(uint32_t pc_idx) {
    // Weighted average of SHCT banks in 3-bit domain
    uint16_t s0 = SHCT[0][pc_idx];
    uint16_t s1 = SHCT[1][pc_idx];
    uint16_t num = static_cast<uint16_t>(s0 * (8 - blend_w) + s1 * blend_w);
    return static_cast<uint8_t>((num + 4) >> 3);
}
static inline uint8_t blended_life_short(uint32_t pc_idx) {
    uint16_t s0 = LIFE_SHORT[0][pc_idx];
    uint16_t s1 = LIFE_SHORT[1][pc_idx];
    uint16_t num = static_cast<uint16_t>(s0 * (8 - blend_w) + s1 * blend_w);
    return static_cast<uint8_t>((num + 4) >> 3);
}
static inline uint8_t blended_life_long(uint32_t pc_idx) {
    uint16_t s0 = LIFE_LONG[0][pc_idx];
    uint16_t s1 = LIFE_LONG[1][pc_idx];
    uint16_t num = static_cast<uint16_t>(s0 * (8 - blend_w) + s1 * blend_w);
    return static_cast<uint8_t>((num + 4) >> 3);
}

static inline void train_phase_scores(uint32_t pc_idx, uint8_t actual_reuse) {
    uint8_t pred0 = (SHCT[0][pc_idx] >= FRIEND_THR) ? 1 : 0;
    uint8_t pred1 = (SHCT[1][pc_idx] >= FRIEND_THR) ? 1 : 0;
    if (pred0 == actual_reuse) sat_inc_s16(phase_score[0], 63);
    else sat_dec_s16(phase_score[0], -63);
    if (pred1 == actual_reuse) sat_inc_s16(phase_score[1], 63);
    else sat_dec_s16(phase_score[1], -63);
}

static inline void update_stride_and_stream(uint32_t pc_idx, uint64_t line, uint32_t set) {
    uint16_t cur = static_cast<uint16_t>(line & 0xFFFF);
    int16_t delta = static_cast<int16_t>(cur - PC_LAST_LINE[pc_idx]); // wrap-safe
    int8_t stride = static_cast<int8_t>(delta);
    bool cont = (PC_STR_RUN[pc_idx] > 0) && (stride == PC_LAST_STRIDE[pc_idx]);
    if (cont) {
        if (PC_STR_RUN[pc_idx] < 3) PC_STR_RUN[pc_idx]++;
    } else {
        PC_STR_RUN[pc_idx] = 1;
        PC_LAST_STRIDE[pc_idx] = stride;
    }
    PC_LAST_LINE[pc_idx] = cur;

    bool small_stride = (stride == 1 || stride == -1 || stride == 2 || stride == -2);
    if (small_stride && PC_STR_RUN[pc_idx] >= 3) {
        sat_inc_u8(PC_STREAM[pc_idx], STREAM_MAX);
        sat_inc_u8(set_scan_debt[set], 3);
    } else {
        sat_dec_u8(PC_STREAM[pc_idx]);
        sat_dec_u8(set_scan_debt[set]);
    }
}

static inline uint8_t count_rrpv3(const LineMeta* lines) {
    uint8_t c = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (lines[w].rrpv == 3) c++;
    return c;
}

static inline void age_rrpv(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3;
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
        }
    }
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) sample_sig[ss][w] = 0; // invalid
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[0][i] = SHCT[1][i] = 0;
        LIFE_SHORT[0][i] = LIFE_SHORT[1][i] = 0;
        LIFE_LONG[0][i]  = LIFE_LONG[1][i]  = 0;
        PC_STREAM[i] = 0;
        DOA_NEG[i] = 0;
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 0;
        DOA_OBJ[i] = 0;
        EAF_OBJ[i] = 0;
    }
    zcnt_ins1 = zcnt_ins2 = 0;
    xdip_mode = 2;
    last_xdip_flip_event = 0;

    blend_w = 4;
    phase_score[0] = phase_score[1] = 0;
    last_phase_event = 0;

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

    // SRRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        age_rrpv(set); // saturating to 3 ensures termination
    }
    // Should never reach
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

    // periodic predictor decay and phase blend update
    if ((event_ctr % DECAY_PERIOD) == 0) decay_predictors_partial(PARTIAL_DECAY_CHUNK);
    update_xdip_mode();
    update_phase_blend_epoch();

    uint32_t pc_idx = pc_index(PC);
    uint64_t line = (paddr >> 6);
    update_stride_and_stream(pc_idx, line, set);

    // Scan clamp level driven by set debt and object EAF on this access
    uint32_t obj_idx_cur = obj_sig_from(pc_idx, paddr);
    bool scanning = (PC_STREAM[pc_idx] >= STREAM_THR) || (set_scan_debt[set] >= SET_STREAM_THR);
    uint8_t clamp = 0;
    if (scanning) {
        clamp = 2;
        if (EAF_OBJ[obj_idx_cur] >= 2) clamp = 3; // raise clamp only with EAF evidence
    }

    if (hit) {
        // Multi-hit promotion with clamp; defer for prefetch-shadow (2 demand hits required)
        LineMeta &lm = meta[set][way];
        if (type != PREFETCH) {
            if (lm.hits < 3) lm.hits++;
        }
        // Effective hits consider prefetch-shadow: need two demand hits before any promotion
        uint8_t eff_hits = lm.hits;
        if (lm.pf) {
            if (eff_hits < 2) return; // no promotion yet
            eff_hits -= 2;
        }
        // Compute target RRPV based on effective hits
        uint8_t target = lm.rrpv; // default no change
        if (eff_hits == 0) {
            // still no promotion (already handled above)
            target = lm.rrpv;
        } else if (eff_hits == 1) {
            target = 1;
            if (clamp >= 2) target = std::max<uint8_t>(target, 2);
        } else if (eff_hits >= 2) {
            target = 1;
            if (eff_hits >= 3) target = 0;
            if (clamp == 2) {
                // allow second hit to reach 1; third+ to 0
            } else if (clamp == 3) {
                // stronger clamp: second hit cannot go below 2; third+ cannot go below 1
                if (eff_hits == 2) target = std::max<uint8_t>(target, 2);
                if (eff_hits >= 3) target = std::max<uint8_t>(target, 1);
            }
        }
        if (target < lm.rrpv) lm.rrpv = target;
        return;
    }

    // Miss/Fill path: train on eviction for sampled sets
    if (is_sampled(set)) {
        uint32_t sidx = sample_set_idx(set);
        uint16_t entry = sample_sig[sidx][way];
        if (entry & 0x8000) { // valid
            uint32_t ev_pc_idx = (entry & (PC_IDX_SIZE - 1));
            uint8_t ev_bank = (entry >> 14) & 0x1;
            uint32_t ev_obj_idx = obj_sig_from(ev_pc_idx, victim_addr);
            uint8_t reused = (meta[set][way].hits > 0) ? 1 : 0;

            // Train SHCT on the recorded bank
            if (reused) sat_inc_u8(SHCT[ev_bank][ev_pc_idx], SHCT_MAX);
            else        sat_dec_u8(SHCT[ev_bank][ev_pc_idx]);

            // Train DOA shortlist (PC) and OBJ DOA/EAF
            if (reused) {
                DOA_NEG[ev_pc_idx] = 0;
                sat_dec_u8(DOA_OBJ[ev_obj_idx]);
                sat_dec_u8(EAF_OBJ[ev_obj_idx]);
            } else {
                sat_inc_u8(DOA_NEG[ev_pc_idx], DOA_MAX);
                sat_inc_u8(DOA_OBJ[ev_obj_idx], DOA_MAX);
                sat_inc_u8(EAF_OBJ[ev_obj_idx], EAF_MAX);
            }

            // Lifetime signals
            if (meta[set][way].hits >= 2) {
                sat_inc_u8(LIFE_LONG[ev_bank][ev_pc_idx], LIFE_MAX);
                sat_dec_u8(LIFE_SHORT[ev_bank][ev_pc_idx]);
                sat_inc_u8(OBJ_LIFE[ev_obj_idx], LIFE_MAX);
            } else {
                sat_inc_u8(LIFE_SHORT[ev_bank][ev_pc_idx], LIFE_MAX);
                sat_dec_u8(LIFE_LONG[ev_bank][ev_pc_idx]);
                sat_dec_u8(OBJ_LIFE[ev_obj_idx]);
            }

            // Phase-bank correctness score (for blend weight)
            train_phase_scores(ev_pc_idx, reused);

            // XDIP leaders: count zero-reuse evictions
            if (!reused) {
                if (is_leader_ins1(set)) sat_inc_s16(zcnt_ins1, 32760);
                if (is_leader_ins2(set)) sat_inc_s16(zcnt_ins2, 32760);
            } else {
                if (is_leader_ins1(set)) sat_dec_s16(zcnt_ins1, -32760);
                if (is_leader_ins2(set)) sat_dec_s16(zcnt_ins2, -32760);
            }

            sample_sig[sidx][way] = 0; // clear
        }
    }

    // Decide insertion policy for the arriving block
    LineMeta &lm = meta[set][way];
    lm.hits = 0;
    lm.pf = (type == PREFETCH) ? 1 : 0;

    // Predictions
    uint8_t friend_score = blended_friend_score(pc_idx);
    bool friendly = (friend_score >= FRIEND_THR);
    bool short_life = (blended_life_short(pc_idx) >= SHORT_THR) && (OBJ_LIFE[obj_idx_cur] < LONG_THR);

    // Per-set pressure: few rrpv==3 victims => high pressure
    uint8_t easy = count_rrpv3(meta[set]);
    bool pressure_high = (easy <= 2);

    // Obj-gated PC-negative immediate-cold (soft-bypass) unless writeback
    bool obj_bad = (DOA_OBJ[obj_idx_cur] >= 2) || (EAF_OBJ[obj_idx_cur] >= 2);
    bool pc_neg = (DOA_NEG[pc_idx] >= 2);
    bool allow_immediate_cold = (type != WRITEBACK) && pc_neg && obj_bad && pressure_high;

    uint8_t ins_rrpv = 2; // default conservative cold
    if (allow_immediate_cold) {
        ins_rrpv = 3;
    } else if (scanning || !friendly || short_life) {
        // stream/scan or unfriendly/short: conservative or coldest
        ins_rrpv = scanning ? 3 : 2;
    } else {
        // friendly-long: choose via global XDIP winner (never MRU on insert)
        ins_rrpv = (xdip_mode == 1) ? 1 : 2;
    }
    if (type == WRITEBACK) {
        // never bypass writebacks: neutral insertion
        ins_rrpv = std::min<uint8_t>(ins_rrpv, 2);
    }

    lm.rrpv = ins_rrpv;

    // Record signature for sampled training; bank chosen by current blend tendency
    if (is_sampled(set)) {
        uint32_t sidx = sample_set_idx(set);
        uint16_t bank = (blend_w >= 4) ? 1 : 0;
        uint16_t rec = static_cast<uint16_t>(0x8000u | (bank << 14) | (pc_idx & (PC_IDX_SIZE - 1)));
        sample_sig[sidx][way] = rec;
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