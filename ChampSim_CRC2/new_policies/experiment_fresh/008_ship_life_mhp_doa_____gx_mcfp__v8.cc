#include <vector>
#include <cstdint>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Tunables ----------------
#define SAMPLE_RATE 32               // 1 out of 32 sets are sampled
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

// Phase blending epoch and disagreement control (FreezeBlend v2)
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
    uint8_t rrpv; // 0..3 (2 bits)
    uint8_t hits; // 0..3 demand hits (2 bits)
    uint8_t pf;   // 0/1 (1 bit)
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t set_scan_debt[LLC_SETS];     // 0..3 (scan clamp)
static uint8_t set_tail_dead[LLC_SETS];     // 0..3 (dead-on-eviction EWMA)
static uint8_t set_tail_pressure[LLC_SETS]; // 0..3 (tail occupancy EWMA)

// ---------------- Sampled-set signature table (valid|bank|pc_idx) ----------------
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // bit15=valid, bit14=bank, low11=pc_idx

// ---------------- PC-indexed predictors (banked SHCT/LIFE) ----------------
static uint8_t SHCT[2][PC_IDX_SIZE];        // friendliness (3-bit)
static uint8_t LIFE_SHORT[2][PC_IDX_SIZE];  // short-life (2-bit)
static uint8_t LIFE_LONG[2][PC_IDX_SIZE];   // long-life (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];      // stream/scan score (2-bit)
static uint8_t DOA_NEG[PC_IDX_SIZE];        // PC-negative shortlist (2-bit)

// Lightweight per-PC stride detector (MCF pointer-chase gate)
static uint16_t PC_LAST_LINE[PC_IDX_SIZE];  // low 16b of last line addr
static int8_t   PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];    // 0..3

// ---------------- Object (PC⊕page) predictors ----------------
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];  // 2-bit object short-life score
static uint8_t DOA_OBJ[OBJ_SIG_SIZE];   // 2-bit object DOA confidence
static uint8_t EAF_OBJ[OBJ_SIG_SIZE];   // 2-bit eviction-age feedback

// ---------------- XDIP-Dwell state ----------------
static int16_t zcnt_ins1 = 0; // leader group preferring insertion RRPV=1
static int16_t zcnt_ins2 = 0; // leader group preferring insertion RRPV=2
static uint8_t xdip_mode = 2; // current winner (1 or 2)
static uint64_t last_xdip_flip_event = 0;

// ---------------- FreezeBlend v2 state ----------------
static uint8_t  blend_w = 4;            // 0..8 weight for bank1 (bank0 gets 8-blend_w)
static int16_t  phase_score[2] = {0,0}; // correctness margin per bank
static uint32_t disagree_ctr = 0;
static bool     blend_frozen = false;
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

// Update XDIP winner with hysteresis and dwell
static inline void update_xdip_mode() {
    int32_t margin = static_cast<int32_t>(zcnt_ins2) - static_cast<int32_t>(zcnt_ins1);
    uint8_t next_mode = xdip_mode;
    if (margin >= XDIP_UP_THR) next_mode = 2;
    else if (margin <= -XDIP_DN_THR) next_mode = 1;

    if (next_mode != xdip_mode) {
        if (event_ctr - last_xdip_flip_event >= XDIP_DWELL_EVENTS) {
            xdip_mode = next_mode;
            last_xdip_flip_event = event_ctr;
            // soften counters to avoid flip-flop
            if (zcnt_ins1 > 0) zcnt_ins1 >>= 1; else if (zcnt_ins1 < 0) zcnt_ins1 = (zcnt_ins1 + 1) >> 1;
            if (zcnt_ins2 > 0) zcnt_ins2 >>= 1; else if (zcnt_ins2 < 0) zcnt_ins2 = (zcnt_ins2 + 1) >> 1;
        }
    }
}

// ---------------- Initialization ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_scan_debt[s] = 0;
        set_tail_dead[s] = 0;
        set_tail_pressure[s] = 0;
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
        DOA_NEG[i]   = 0;
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 0;
        DOA_OBJ[i]  = 0;
        EAF_OBJ[i]  = 0;
    }
    zcnt_ins1 = zcnt_ins2 = 0;
    xdip_mode = 2;
    blend_w = 4;
    phase_score[0] = phase_score[1] = 0;
    disagree_ctr = 0;
    blend_frozen = false;
    freeze_epochs_left = 0;
    last_phase_event = 0;
    event_ctr = 0;
    last_xdip_flip_event = 0;
}

// ---------------- Victim selection ----------------
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

    // Update tail-pressure EWMA based on current tail occupancy
    uint32_t tail_cnt = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (meta[set][w].rrpv == 3) tail_cnt++;
    if (tail_cnt >= (LLC_WAYS / 2)) sat_inc_u8(set_tail_pressure[set], 3);
    else sat_dec_u8(set_tail_pressure[set]);

    // SRRIP: age until a line reaches RRPV=3
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        // Age all
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // Fallback (should not happen)
    return 0;
}

// ---------------- State update ----------------
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

    // Per-PC stride + stream detector (MCF pointer-chase gate + scan evidence)
    uint32_t pidx = pc_index(PC);
    uint32_t cur_line = static_cast<uint32_t>(paddr >> 6); // line granularity
    int32_t stride = static_cast<int32_t>(static_cast<int32_t>(cur_line & 0xFFFF) - static_cast<int32_t>(PC_LAST_LINE[pidx]));
    if (stride > 127) stride = 127;
    if (stride < -127) stride = -127;
    if (PC_LAST_STRIDE[pidx] == static_cast<int8_t>(stride) && (stride >= -1 && stride <= 1)) {
        if (PC_STR_RUN[pidx] < 3) PC_STR_RUN[pidx]++;
    } else {
        PC_STR_RUN[pidx] = 0;
        PC_LAST_STRIDE[pidx] = static_cast<int8_t>(stride);
    }
    PC_LAST_LINE[pidx] = static_cast<uint16_t>(cur_line & 0xFFFF);
    bool streamish = (stride == 1 || stride == -1);
    if (streamish) sat_inc_u8(PC_STREAM[pidx], STREAM_MAX);
    else sat_dec_u8(PC_STREAM[pidx]);
    if (streamish) sat_inc_u8(set_scan_debt[set], 3);
    else sat_dec_u8(set_scan_debt[set]);

    // Hit path: multi-hit promotion with quarantine for prefetched lines
    if (hit) {
        LineMeta &lm = meta[set][way];
        if (type == PREFETCH) {
            // never promote on prefetch hit
        } else {
            if (lm.pf) {
                // Require two demand hits before any promotion
                if (lm.hits < 2) {
                    lm.hits++;
                    if (lm.rrpv > 2) lm.rrpv = 2; // keep cold while quarantined
                } else {
                    if (lm.rrpv > 0) lm.rrpv = 0;
                    if (lm.hits < 3) lm.hits++;
                    lm.pf = 0; // graduated from quarantine
                }
            } else {
                if (lm.hits == 0) {
                    lm.hits = 1;
                    if (lm.rrpv > 1) lm.rrpv = 1; // no first-hit MRU
                } else {
                    if (lm.rrpv > 0) lm.rrpv = 0; // MRU only after ≥2 demand hits
                    if (lm.hits < 3) lm.hits++;
                }
            }
        }
        // periodic decay and exit
        if ((event_ctr % DECAY_PERIOD) == 0) decay_predictors_partial(PARTIAL_DECAY_CHUNK);
        return;
    }

    // Miss/fill path: train on the evicted line first using old metadata
    LineMeta old = meta[set][way];

    // Sampled-set bookkeeping for leader-dueling and phase blending
    bool sampled = is_sampled(set);
    uint32_t ss = sampled ? sample_set_idx(set) : 0;
    uint16_t sig_entry = sampled ? sample_sig[ss][way] : 0;
    bool sig_valid = sampled && ((sig_entry >> 15) & 1);
    uint32_t pidx_old = sig_valid ? (sig_entry & ((1u << 11) - 1)) : pidx; // fallback to current pidx
    uint8_t sh0_prev = SHCT[0][pidx_old];
    uint8_t sh1_prev = SHCT[1][pidx_old];
    bool pred_friend0 = (sh0_prev >= FRIEND_THR);
    bool pred_friend1 = (sh1_prev >= FRIEND_THR);

    // Outcome
    bool evicted_dead = (old.hits == 0);

    // Phase scoring on disagreement (before updating tables)
    if (sig_valid) {
        bool reused = !evicted_dead;
        // bank 0
        if (pred_friend0 == reused) { if (phase_score[0] < 127) phase_score[0]++; }
        else { if (phase_score[0] > -128) phase_score[0]--; }
        // bank 1
        if (pred_friend1 == reused) { if (phase_score[1] < 127) phase_score[1]++; }
        else { if (phase_score[1] > -128) phase_score[1]--; }
        // disagreement tracker
        if (pred_friend0 != pred_friend1) {
            if (disagree_ctr < 0xFFFFFFFFu) disagree_ctr++;
        } else if (disagree_ctr > 0) {
            disagree_ctr--;
        }
        // FreezeBlend v2: freeze weight on persistent disagreement; drift toward neutral during dwell
        if (!blend_frozen && disagree_ctr > DISAGREE_FREEZE_THR) {
            blend_frozen = true;
            freeze_epochs_left = FREEZE_DWELL_EPOCHS + 1;
            last_phase_event = event_ctr;
            disagree_ctr = 0;
        }
        if (blend_frozen) {
            if ((event_ctr - last_phase_event) >= PHASE_EPOCH) {
                last_phase_event = event_ctr;
                if (freeze_epochs_left > 0) freeze_epochs_left--;
                else blend_frozen = false;
            }
            // drift toward neutral while frozen
            if (blend_w > 4) blend_w--;
            else if (blend_w < 4) blend_w++;
        } else {
            // adjust weight toward better-scoring bank with gentle nudges
            int32_t diff = static_cast<int32_t>(phase_score[1]) - static_cast<int32_t>(phase_score[0]);
            if (diff > 8 && blend_w < 8) blend_w++;
            else if (diff < -8 && blend_w > 0) blend_w--;
        }
    }

    // SHiP + DOA/EAF training on eviction
    if (evicted_dead) {
        if (SHCT[0][pidx_old] > 0) SHCT[0][pidx_old]--;
        if (SHCT[1][pidx_old] > 0) SHCT[1][pidx_old]--;
        sat_inc_u8(DOA_NEG[pidx_old], DOA_MAX);
        uint32_t os = obj_sig_from(pidx_old, victim_addr);
        sat_inc_u8(DOA_OBJ[os], DOA_MAX);
        sat_inc_u8(EAF_OBJ[os], EAF_MAX);
        // object lifetime learns "short"
        sat_inc_u8(OBJ_LIFE[os], LIFE_MAX);
        sat_inc_u8(set_tail_dead[set], 3);
    } else {
        sat_inc_u8(SHCT[0][pidx_old], SHCT_MAX);
        sat_inc_u8(SHCT[1][pidx_old], SHCT_MAX);
        sat_dec_u8(DOA_NEG[pidx_old]);
        uint32_t os = obj_sig_from(pidx_old, victim_addr);
        sat_dec_u8(DOA_OBJ[os]);
        sat_dec_u8(EAF_OBJ[os]);
        // lifetime training by hit count
        if (old.hits >= 2) {
            sat_inc_u8(LIFE_LONG[0][pidx_old], LIFE_MAX);
            sat_inc_u8(LIFE_LONG[1][pidx_old], LIFE_MAX);
            sat_dec_u8(LIFE_SHORT[0][pidx_old]);
            sat_dec_u8(LIFE_SHORT[1][pidx_old]);
            if (OBJ_LIFE[os] > 0) OBJ_LIFE[os]--; // object learned long
        } else {
            sat_inc_u8(LIFE_SHORT[0][pidx_old], LIFE_MAX);
            sat_inc_u8(LIFE_SHORT[1][pidx_old], LIFE_MAX);
            sat_dec_u8(LIFE_LONG[0][pidx_old]);
            sat_dec_u8(LIFE_LONG[1][pidx_old]);
            sat_inc_u8(OBJ_LIFE[os], LIFE_MAX);   // object short
        }
    }

    // XDIP leader-dueling update on sampled leader sets
    if (sig_valid) {
        bool bank = ((sig_entry >> 14) & 1);
        if (evicted_dead) {
            if (!bank) sat_dec_s16(zcnt_ins1, -128);
            else       sat_dec_s16(zcnt_ins2, -128);
        } else {
            if (!bank) sat_inc_s16(zcnt_ins1, 127);
            else       sat_inc_s16(zcnt_ins2, 127);
        }
        sample_sig[ss][way] = 0; // clear
        update_xdip_mode();
    }

    // Decide insertion for the new line
    uint32_t osig = obj_sig_from(pidx, paddr);
    // Blend banks
    uint32_t w0 = 8 - blend_w;
    uint32_t w1 = blend_w;
    uint32_t fr_blend = (w0 * SHCT[0][pidx] + w1 * SHCT[1][pidx] + 4) >> 3;
    bool friendly = (fr_blend >= FRIEND_THR);

    uint32_t sh_blend = (w0 * LIFE_SHORT[0][pidx] + w1 * LIFE_SHORT[1][pidx] + 4) >> 3;
    uint32_t lg_blend = (w0 * LIFE_LONG[0][pidx]  + w1 * LIFE_LONG[1][pidx]  + 4) >> 3;
    bool short_like = (sh_blend >= SHORT_THR) || (OBJ_LIFE[osig] >= SHORT_THR);
    bool long_like  = (lg_blend >= LONG_THR) && (OBJ_LIFE[osig] < SHORT_THR);

    bool doa_pc  = (DOA_NEG[pidx] >= 2);
    bool doa_obj = (DOA_OBJ[osig] >= 2);
    bool eaf_hi  = (EAF_OBJ[osig] >= 2);

    bool ptr_chase = (PC_STR_RUN[pidx] >= 2) && (PC_LAST_STRIDE[pidx] >= -1) && (PC_LAST_STRIDE[pidx] <= 1);
    bool mcf_boost = ptr_chase && (type == LOAD || type == RFO);

    bool allow_soft_bypass = (set_tail_pressure[set] >= 2) || (set_tail_dead[set] >= 2);

    uint8_t new_rrpv = 2;
    uint8_t new_hits = 0;
    uint8_t new_pf   = 0;

    if (type == WRITEBACK) {
        // never bypass writebacks; conservative insertion
        new_rrpv = 2;
    } else if (type == PREFETCH) {
        // quarantine prefetches deep; require two demand hits before promotion
        new_rrpv = 3;
        new_pf = 1;
    } else {
        // demand (LOAD/RFO)
        if (streamish || PC_STREAM[pidx] >= STREAM_THR) {
            new_rrpv = 3; // scan resistance
        } else {
            bool pred_dead = (doa_pc && (doa_obj || eaf_hi)) || (mcf_boost && (doa_pc || doa_obj));
            if (pred_dead) {
                new_rrpv = allow_soft_bypass ? 3 : 2; // Tail-Guard v2
            } else {
                if (friendly && !short_like && long_like) {
                    new_rrpv = (xdip_mode == 1 ? 1 : 2);
                    // ObjLife+EAF inversion: sustained short-life objects avoid shallow insertion
                    if (OBJ_LIFE[osig] >= SHORT_THR && eaf_hi) new_rrpv = 2;
                } else {
                    new_rrpv = 2; // conservative-cold
                }
            }
        }
        // Scan clamp: never allow shallower than 2 under scan debt
        if (set_scan_debt[set] > 0 && new_rrpv < 2) new_rrpv = 2;
    }

    // Install new metadata
    meta[set][way].rrpv = std::min<uint8_t>(3, new_rrpv);
    meta[set][way].hits = new_hits;
    meta[set][way].pf   = new_pf;

    // Sampled-set signature for future training
    if (sampled) {
        uint16_t v = 0;
        v |= (1u << 15); // valid
        uint16_t bankbit = 0;
        bool leader1 = is_leader_ins1(set);
        bool leader2 = is_leader_ins2(set);
        if (leader1) bankbit = 0;
        else if (leader2) bankbit = 1;
        else bankbit = (xdip_mode == 2); // track winner for context
        v |= (bankbit << 14);
        v |= (pidx & ((1u << 11) - 1));
        sample_sig[ss][way] = v;
    }

    // Periodic predictor decay
    if ((event_ctr % DECAY_PERIOD) == 0) decay_predictors_partial(PARTIAL_DECAY_CHUNK);
}

// Print end-of-simulation statistics
void PrintStats() {
    // intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // intentionally blank
}