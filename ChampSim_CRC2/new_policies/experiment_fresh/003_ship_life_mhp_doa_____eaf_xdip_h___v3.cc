#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Tunables (kept compatible with previous design, refined thresholds)
#define SAMPLE_RATE 32               // 3.125% sampled sets
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define OBJ_SIG_SIZE 1024            // 10-bit PC⊕page signature

#define SHCT_MAX 7                   // 3-bit friendliness
#define LIFE_MAX 3                   // 2-bit life scores
#define STREAM_MAX 3                 // 2-bit stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define EAF_MAX 3                    // 2-bit eviction-age score

#define FRIEND_THR 3                 // SHCT >= 3 => friendly
#define SHORT_THR 2                  // LIFE_SHORT >= 2 => short-life
#define LONG_THR 2                   // LIFE_LONG >= 2 => long-life
#define STREAM_THR 2                 // PC_STREAM >= 2 => stream/scan
#define SET_STREAM_THR 2             // per-set streaminess threshold

#define DECAY_PERIOD 4096            // periodic decay cadence
#define PARTIAL_DECAY_CHUNK 64       // number of PC entries to decay on burst

// XDIP hysteresis
#define XDIP_UP_THR 8                // margin to flip toward ins=1
#define XDIP_DN_THR 8                // margin to flip toward ins=2

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

// Per-line metadata (minimal bits: rrpv=2, hits=2, pf=1)
struct LineMeta {
    uint8_t rrpv; // 0..3
    uint8_t hits; // 0..3 (we only need 0,1,2+)
    uint8_t pf;   // 0/1 (prefetch origin)
};

static LineMeta meta[LLC_SETS][LLC_WAYS];

// Per-set lightweight streaminess and scan debt (2 bits each)
static uint8_t set_stream[LLC_SETS];   // 0..3
static uint8_t set_scan_debt[LLC_SETS];// 0..3

// Sampled-set signature table: store PC index with an embedded valid bit
// Use 0x8000 as valid bit, low 11 bits carry PC index
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS];

// Global PC-indexed predictors
static uint8_t SHCT[PC_IDX_SIZE];          // friendliness (3-bit)
static uint8_t LIFE_SHORT[PC_IDX_SIZE];    // short-life score (2-bit)
static uint8_t LIFE_LONG[PC_IDX_SIZE];     // long-life score (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];     // stream/scan score (2-bit)
static uint8_t DOA_CONF[PC_IDX_SIZE];      // DOA confidence (2-bit)
static uint8_t EAF[PC_IDX_SIZE];           // eviction-age feedback (2-bit)

// Lightweight per-PC stride detector
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low bits of last line address
static int8_t   PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];   // 0..3

// Object lifetime by PC⊕page signature (2-bit)
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];

// XDIP-H (zero-reuse–driven insertion preference with hysteresis)
static int16_t zcnt_ins1 = 0; // leader group preferring RRPV=1
static int16_t zcnt_ins2 = 0; // leader group preferring RRPV=2
static uint8_t xdip_mode = 2; // current global winner (1 or 2)

// Epoch tracking
static uint64_t event_ctr = 0;

// Helpers
static inline uint32_t pc_index(uint64_t pc) {
    // mix to 11 bits
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 7) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t obj_sig(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12); // 4KB page
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7);
    return static_cast<uint32_t>(mix) & (OBJ_SIG_SIZE - 1);
}
static inline bool is_sampled(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
// More leader sets with spacing and symmetry
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
    // partial decay: halve confidence for agility
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = (event_ctr + i) & (PC_IDX_SIZE - 1);
        SHCT[idx]       >>= 1;
        LIFE_SHORT[idx] >>= 1;
        LIFE_LONG[idx]  >>= 1;
        PC_STREAM[idx]  >>= 1;
        DOA_CONF[idx]   >>= 1;
        EAF[idx]        >>= 1;
    }
}

static inline void update_xdip_mode() {
    int32_t margin = static_cast<int32_t>(zcnt_ins2) - static_cast<int32_t>(zcnt_ins1);
    if (margin >= XDIP_UP_THR) {
        xdip_mode = 2;
    } else if (margin <= -XDIP_DN_THR) {
        xdip_mode = 1;
    }
}

// Update per-PC stride and per-set streaminess + scan debt
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
        if (set_scan_debt[set] < 3) set_scan_debt[set]++; // raise scan debt
    } else {
        sat_dec_u8(PC_STREAM[pc_idx]);
        sat_dec_u8(set_stream[set]);
        // bleed scan debt slowly
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }
}

// XDIP winner with stream filter: 1 => RRPV=1 insertion, 2 => RRPV=2 insertion
static inline uint8_t xdip_winner_filtered(uint32_t set) {
    if (is_leader_ins1(set)) return 1;
    if (is_leader_ins2(set)) return 2;
    // filter by set streaminess
    if (set_stream[set] >= SET_STREAM_THR) return 2;
    return xdip_mode;
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_stream[s] = 0;
        set_scan_debt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3;
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[i] = FRIEND_THR;   // neutral
        LIFE_SHORT[i] = 1;      // mild short-life prior
        LIFE_LONG[i]  = 1;      // mild long-life prior
        PC_STREAM[i]  = 1;      // slight stream suspicion
        DOA_CONF[i]   = 0;      // no DOA
        EAF[i]        = 0;      // no early-death penalty
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t s = 0; s < (LLC_SETS / SAMPLE_RATE); s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_sig[s][w] = 0; // valid=0
        }
    }
    zcnt_ins1 = 0;
    zcnt_ins2 = 0;
    xdip_mode = 2;
    event_ctr = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        // age all lines (saturate at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    // unreachable
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    event_ctr++;

    uint32_t pc_idx = pc_index(PC);
    uint64_t line_addr = (paddr >> 6);

    // stride/stream tracking (for both hit and miss demand-like events)
    if (type != WRITEBACK) {
        update_stride_stream(pc_idx, set, line_addr);
    } else {
        // bleed scan debt slowly on writebacks
        if (set_scan_debt[set] > 0) set_scan_debt[set]--;
    }

    // Periodic gentle decay to stay agile
    if ((event_ctr & (DECAY_PERIOD - 1)) == 0) {
        decay_predictors_partial(PARTIAL_DECAY_CHUNK);
    }

    if (hit) {
        // Multi-hit promotion with gates
        LineMeta &lm = meta[set][way];

        // First hit behavior
        if (lm.hits == 0) {
            lm.hits = 1;
            // suppress first-hit promotion under scan/stream or if prefetch
            bool suppress = lm.pf || (set_scan_debt[set] > 0) || (PC_STREAM[pc_idx] >= STREAM_THR);
            if (!suppress) {
                if (lm.rrpv > 1) lm.rrpv = 1; // cap at warm
            } else {
                if (lm.rrpv > 2) lm.rrpv = 2; // keep colder
            }
        } else {
            // 2nd+ hit: time-gated MRU promotion
            if (lm.rrpv <= 1) lm.rrpv = 0; else lm.rrpv = 1;
            if (lm.hits < 3) lm.hits++;
            // after a confirming hit, clear prefetch origin to allow normal promotion later
            lm.pf = 0;
        }

        // Any reuse reduces DOA/EAF for this PC
        sat_dec_u8(DOA_CONF[pc_idx]);
        sat_dec_u8(EAF[pc_idx]);
        // long-life reinforcement on multi-hit
        if (meta[set][way].hits >= 2) sat_inc_u8(LIFE_LONG[pc_idx], LIFE_MAX);

        return;
    }

    // Miss / Fill path: train on victim (if any), then initialize new line
    // Train only for sampled sets (SHiP-style)
    if (is_sampled(set)) {
        uint32_t sidx = set / SAMPLE_RATE;
        uint16_t sig = sample_sig[sidx][way];
        if (sig & 0x8000u) {
            uint32_t vpc = sig & 0x07FFu;
            // Victim's local metadata before overwrite
            LineMeta vmeta = meta[set][way];

            // Zero-reuse / died-before-second classification
            bool zero_reuse = (vmeta.hits == 0);
            bool died_before_second = (vmeta.hits < 2);

            if (zero_reuse) {
                if (SHCT[vpc] > 0) SHCT[vpc]--;
                sat_inc_u8(DOA_CONF[vpc], DOA_MAX);
                sat_inc_u8(LIFE_SHORT[vpc], LIFE_MAX);
                // XDIP training
                if (is_leader_ins1(set)) sat_inc_s16(zcnt_ins1, 32767);
                if (is_leader_ins2(set)) sat_inc_s16(zcnt_ins2, 32767);
            } else {
                if (SHCT[vpc] < SHCT_MAX) SHCT[vpc]++;
                sat_dec_u8(DOA_CONF[vpc]);
                if (vmeta.hits >= 2) {
                    sat_inc_u8(LIFE_LONG[vpc], LIFE_MAX);
                } else {
                    sat_inc_u8(LIFE_SHORT[vpc], LIFE_MAX);
                }
                // XDIP training opposite direction
                if (is_leader_ins1(set)) sat_dec_s16(zcnt_ins1, -32768);
                if (is_leader_ins2(set)) sat_dec_s16(zcnt_ins2, -32768);
            }

            // EAF training: early death penalizes PC
            if (died_before_second) sat_inc_u8(EAF[vpc], EAF_MAX);
            else sat_dec_u8(EAF[vpc]);

            // Object lifetime update via PC⊕page signature
            uint32_t os = obj_sig(vpc, victim_addr);
            if (vmeta.hits >= 2) {
                sat_inc_u8(OBJ_LIFE[os], LIFE_MAX);
            } else {
                sat_dec_u8(OBJ_LIFE[os]);
            }

            // Clear signature valid
            sample_sig[sidx][way] = 0;
            // Update XDIP hysteresis after training
            update_xdip_mode();
        }
    }

    // Decide insertion for the incoming line
    LineMeta &nm = meta[set][way];
    nm.hits = 0;
    nm.pf = (type == PREFETCH) ? 1 : 0;

    // Predictions
    bool friendly = (SHCT[pc_idx] >= FRIEND_THR);
    bool short_like = (LIFE_SHORT[pc_idx] >= SHORT_THR) || (PC_STREAM[pc_idx] >= STREAM_THR) || (set_stream[set] >= SET_STREAM_THR);
    bool long_like = (LIFE_LONG[pc_idx] >= LONG_THR);
    bool eaf_high = (EAF[pc_idx] >= 2) || (DOA_CONF[pc_idx] >= 2);

    // Object-based refinement
    uint32_t os_new = obj_sig(pc_idx, paddr);
    if (OBJ_LIFE[os_new] >= LONG_THR) long_like = true;

    uint8_t ins_rrpv = 3;

    if (type == WRITEBACK) {
        // Never bypass on writeback; place moderately warm
        ins_rrpv = 1;
    } else if (nm.pf) {
        // Prefetches default to cold to avoid pollution
        ins_rrpv = 3;
    } else {
        // Demand insertion
        if (friendly && !short_like && !eaf_high) {
            // Friendly and long-lived -> XDIP controlled warmth
            uint8_t pref = xdip_winner_filtered(set);
            ins_rrpv = (pref == 1) ? 1 : 2;
            if (long_like && ins_rrpv > 1) ins_rrpv = 1; // bias long-life warmer
        } else {
            // Likely short/stream/early death -> cold insertion
            ins_rrpv = 3;
        }

        // DOA/EAF hardening: escalate to cold insertion
        if (eaf_high) ins_rrpv = 3;

        // Scan debt guard: suppress warm insertion when set is in scan
        if (set_scan_debt[set] > 0) ins_rrpv = 3;
    }

    nm.rrpv = ins_rrpv;

    // Record signature for future training (sampled sets only)
    if (is_sampled(set)) {
        uint32_t sidx = set / SAMPLE_RATE;
        sample_sig[sidx][way] = static_cast<uint16_t>(0x8000u | (pc_idx & 0x07FFu));
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}