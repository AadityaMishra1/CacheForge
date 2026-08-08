#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32               // sampled sets for SHiP training (1 out of 32)
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define OBJ_SIG_SIZE 1024            // 10-bit object signature (PC_idx ⊕ page)

// Predictor ranges (saturating)
#define SHCT_MAX 7                   // 3-bit friendliness
#define LIFE_MAX 3                   // 2-bit lifetime (short/long banks)
#define STREAM_MAX 3                 // 2-bit per-PC stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define EAF_MAX 3                    // 2-bit eviction-age feedback

// Thresholds
#define FRIEND_THR 3
#define SHORT_THR 2
#define LONG_THR 2
#define STREAM_THR 2

// Decay
#define DECAY_PERIOD 2048
#define PARTIAL_DECAY_CHUNK 64

// XDIP hysteresis and dwell
#define XDIP_UP_THR 8
#define XDIP_DN_THR 8
#define XDIP_DWELL_EVENTS 8192

// Phase blending epoch and disagreement control
#define PHASE_EPOCH 131072
#define DISAGREE_FREEZE_THR 512
#define FREEZE_DWELL_EPOCHS 1

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
    uint8_t hits; // 0..3 demand hits
    uint8_t pf;   // 0/1 (prefetch origin)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS];   // 0..3 (scan clamp driver)
static uint8_t set_tail_dead[LLC_SETS];   // 0..3 (tail-dead ratio tracker)

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
static uint32_t disagree_ctr = 0;
static bool blend_frozen = false;
static uint32_t freeze_epochs_left = 0;
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
    // round-robin decay over PC-indexed and object-indexed predictors
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

// Blend control with freeze/unfreeze
static inline void update_blend_control(bool disagree, bool bank0_correct, bool bank1_correct) {
    if (disagree) {
        if (disagree_ctr < 0x7fffffff) disagree_ctr++;
    }
    // accumulate correctness margins
    if (bank0_correct && !bank1_correct) sat_inc_s16(phase_score[0], 32760);
    else if (!bank0_correct && bank1_correct) sat_inc_s16(phase_score[1], 32760);
    // periodic adjustments
    if ((event_ctr - last_phase_event) >= PHASE_EPOCH) {
        last_phase_event = event_ctr;
        if (blend_frozen) {
            // decay toward neutral while frozen
            if (blend_w > 4) blend_w--;
            else if (blend_w < 4) blend_w++;
            if (freeze_epochs_left > 0) freeze_epochs_left--;
            if (freeze_epochs_left == 0) blend_frozen = false;
        } else {
            // adjust weight based on which bank accumulated more margin
            int32_t m0 = static_cast<int32_t>(phase_score[0]);
            int32_t m1 = static_cast<int32_t>(phase_score[1]);
            if (m1 > m0 && blend_w < 8) blend_w++;
            else if (m0 > m1 && blend_w > 0) blend_w--;
            // freeze if disagreements were high
            if (disagree_ctr >= DISAGREE_FREEZE_THR) {
                blend_frozen = true;
                freeze_epochs_left = FREEZE_DWELL_EPOCHS;
            }
        }
        // decay scores and disagreement
        phase_score[0] >>= 1;
        phase_score[1] >>= 1;
        disagree_ctr >>= 1;
    }
}

static inline uint8_t count_tail_pressure(uint32_t set) {
    uint8_t c = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (meta[set][w].rrpv == 3) c++;
    return c;
}

static inline bool is_demand(uint32_t type) { return (type == LOAD) || (type == RFO); }

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        set_tail_dead[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3;
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
        }
    }
    for (uint32_t ss = 0; ss < (LLC_SETS / SAMPLE_RATE); ss++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) sample_sig[ss][w] = 0;
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
        OBJ_LIFE[i] = DOA_OBJ[i] = EAF_OBJ[i] = 0;
    }
    zcnt_ins1 = zcnt_ins2 = 0;
    xdip_mode = 2;
    last_xdip_flip_event = 0;

    blend_w = 4;
    phase_score[0] = phase_score[1] = 0;
    disagree_ctr = 0;
    blend_frozen = false;
    freeze_epochs_left = 0;
    last_phase_event = 0;

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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim selection
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) {
                return w;
            }
        }
        // Age all lines (saturate at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
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

    // Light per-PC stride/stream tracking on every access
    {
        uint32_t idx = pc_index(PC);
        uint16_t line = static_cast<uint16_t>((paddr >> 6) & 0xFFFF);
        int16_t stride = static_cast<int16_t>(line) - static_cast<int16_t>(PC_LAST_LINE[idx]);
        if (stride == PC_LAST_STRIDE[idx] && (stride == 1 || stride == -1)) {
            sat_inc_u8(PC_STR_RUN[idx], 3);
            if (PC_STR_RUN[idx] >= 2) sat_inc_u8(PC_STREAM[idx], STREAM_MAX);
        } else {
            PC_STR_RUN[idx] = 0;
            sat_dec_u8(PC_STREAM[idx]);
        }
        PC_LAST_STRIDE[idx] = static_cast<int8_t>(std::max<int16_t>(-8, std::min<int16_t>(7, stride)));
        PC_LAST_LINE[idx] = line;
        // update per-set scan clamp gently
        if (PC_STREAM[idx] >= STREAM_THR) sat_inc_u8(set_scan_debt[set], 3);
        else sat_dec_u8(set_scan_debt[set]);
    }

    if (hit) {
        // Demand hits contribute to multi-hit promotion; prefetch hits don't
        if (is_demand(type)) {
            if (meta[set][way].hits < 3) meta[set][way].hits++;
            // Multi-hit promotion: require >=2 demand hits; quarantine prefetched lines
            if (meta[set][way].hits >= 2) {
                uint8_t clamp = set_scan_debt[set];
                uint8_t target = (clamp > 0) ? clamp : 0; // respect scan clamp
                meta[set][way].rrpv = target;
            }
        }
    } else {
        // Miss fill: train on the evicted line (previous content of [set][way])
        uint8_t old_hits = meta[set][way].hits;
        uint8_t old_pf   = meta[set][way].pf;
        // Tail-dead ratio update based on evicted line outcome
        if (old_hits == 0) sat_inc_u8(set_tail_dead[set], 3);
        else sat_dec_u8(set_tail_dead[set]);

        if (is_sampled(set)) {
            uint16_t &sig = sample_sig[sample_set_idx(set)][way];
            if (sig & 0x8000) {
                // Valid previous signature -> train
                uint8_t bank = (sig & 0x4000) ? 1 : 0;
                uint32_t pc_idx_old = static_cast<uint32_t>(sig & 0x07FF);
                bool friendly = (old_hits > 0);
                bool longlife = (old_hits > 1);

                // Train SHCT and LIFE (current bank)
                if (friendly) sat_inc_u8(SHCT[bank][pc_idx_old], SHCT_MAX);
                else sat_dec_u8(SHCT[bank][pc_idx_old]);

                if (longlife) {
                    sat_inc_u8(LIFE_LONG[bank][pc_idx_old], LIFE_MAX);
                    sat_dec_u8(LIFE_SHORT[bank][pc_idx_old]);
                } else {
                    sat_inc_u8(LIFE_SHORT[bank][pc_idx_old], LIFE_MAX);
                    sat_dec_u8(LIFE_LONG[bank][pc_idx_old]);
                }

                // DOA/OBJ/EAF feedback
                uint32_t o = obj_sig_from(pc_idx_old, victim_addr);
                if (old_hits == 0) {
                    sat_inc_u8(DOA_NEG[pc_idx_old], DOA_MAX);
                    sat_inc_u8(DOA_OBJ[o], DOA_MAX);
                    sat_inc_u8(EAF_OBJ[o], EAF_MAX);
                } else {
                    sat_dec_u8(DOA_NEG[pc_idx_old]);
                    sat_dec_u8(DOA_OBJ[o]);
                    sat_dec_u8(EAF_OBJ[o]);
                }

                // Phase correctness and disagreement for blending
                bool b0_friend = (SHCT[0][pc_idx_old] >= FRIEND_THR);
                bool b1_friend = (SHCT[1][pc_idx_old] >= FRIEND_THR);
                bool b0_long   = (LIFE_LONG[0][pc_idx_old] >= LONG_THR) && (LIFE_SHORT[0][pc_idx_old] < SHORT_THR);
                bool b1_long   = (LIFE_LONG[1][pc_idx_old] >= LONG_THR) && (LIFE_SHORT[1][pc_idx_old] < SHORT_THR);
                bool disagree = (b0_friend != b1_friend) || (b0_long != b1_long);
                bool bank0_correct = ((b0_friend == friendly) && (b0_long == longlife));
                bool bank1_correct = ((b1_friend == friendly) && (b1_long == longlife));
                update_blend_control(disagree, bank0_correct, bank1_correct);
            }
            // Record new signature for the incoming line
            uint32_t pc_idx_new = pc_index(PC);
            uint16_t rec = 0x8000; // valid
            // bank bit reflects current tilt toward bank1 if weight >= half
            if (blend_w >= 4) rec |= 0x4000;
            rec |= static_cast<uint16_t>(pc_idx_new & 0x07FF);
            sample_sig[sample_set_idx(set)][way] = rec;
        }

        // Leader-dueling accounting (coarse): reward insertion-1 on friendly, insertion-2 otherwise
        if (is_leader_ins1(set) || is_leader_ins2(set)) {
            if (old_hits > 0) { // friendly
                sat_inc_s16(zcnt_ins1, 32760);
                sat_dec_s16(zcnt_ins2, -32760);
            } else {
                sat_inc_s16(zcnt_ins2, 32760);
                sat_dec_s16(zcnt_ins1, -32760);
            }
            update_xdip_mode();
        }

        // Periodic decays to keep predictors fresh
        if ((event_ctr % DECAY_PERIOD) == 0) {
            decay_predictors_partial(PARTIAL_DECAY_CHUNK);
        }

        // Decide insertion for the new line (Tail-Guarded DOA, FreezeBlend, MCF-Boost, quarantine)
        uint32_t pc_idx = pc_index(PC);
        uint32_t o = obj_sig_from(pc_idx, paddr);

        // Blended friendliness and lifetime
        uint32_t w1 = blend_w, w0 = 8 - w1;
        uint32_t f_blend = (SHCT[0][pc_idx] * w0 + SHCT[1][pc_idx] * w1 + 4) >> 3; // 0..7
        uint32_t s_blend = (LIFE_SHORT[0][pc_idx] * w0 + LIFE_SHORT[1][pc_idx] * w1 + 4) >> 3; // 0..3
        uint32_t l_blend = (LIFE_LONG[0][pc_idx]  * w0 + LIFE_LONG[1][pc_idx]  * w1 + 4) >> 3; // 0..3

        bool pred_friendly = (f_blend >= FRIEND_THR);
        bool pred_long     = (l_blend >= LONG_THR) && (s_blend < SHORT_THR);

        // Scan clamp (per-set)
        uint8_t clamp = set_scan_debt[set];

        // MCF-Boost (PC-negative ∧ object DOA/EAF), gated by tail pressure or tail-dead ratio; never for WRITEBACK
        bool pc_neg  = (DOA_NEG[pc_idx] >= 2);
        bool obj_doa = (DOA_OBJ[o] >= 2) || (EAF_OBJ[o] >= 2);
        uint8_t tail_pressure = count_tail_pressure(set);
        bool high_pressure = (tail_pressure >= 8) || (set_tail_dead[set] >= 2);
        bool mcf_boost = (pc_neg && obj_doa && (type != WRITEBACK));

        uint8_t ins_rrpv = 2; // conservative-cold default
        if (mcf_boost && high_pressure && is_demand(type)) {
            ins_rrpv = 3; // soft-bypass under pressure
        } else {
            // Friendly-long gets XDIP depth; others conservative or deep
            if (pred_friendly && pred_long) {
                ins_rrpv = (xdip_mode == 1) ? 1 : 2;
            } else if (pred_friendly) {
                ins_rrpv = 2;
            } else {
                ins_rrpv = high_pressure ? 3 : 2;
            }
        }

        // Prefetch quarantine: insert at tail; demand must earn promotion
        uint8_t pf_flag = 0;
        if (type == PREFETCH) {
            ins_rrpv = 3;
            pf_flag = 1;
        }
        // Never bypass on writeback; keep conservative insertion
        if (type == WRITEBACK) {
            pf_flag = 0;
            if (ins_rrpv > 2) ins_rrpv = 2;
        }

        // Respect scan clamp
        if (ins_rrpv < clamp) ins_rrpv = clamp;

        // Install new metadata
        meta[set][way].rrpv = ins_rrpv;
        meta[set][way].hits = 0;
        meta[set][way].pf   = pf_flag;
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