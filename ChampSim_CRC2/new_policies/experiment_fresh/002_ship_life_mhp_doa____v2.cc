#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Tunables
#define SAMPLE_RATE 32               // 3.125% sampled sets
#define PC_IDX_SIZE 2048             // 11-bit PC index
#define OBJ_SIG_SIZE 1024            // 10-bit PC⊕page signature
#define SHCT_MAX 7                   // 3-bit friendliness
#define LIFE_MAX 3                   // 2-bit life scores
#define STREAM_MAX 3                 // 2-bit stream score
#define DOA_MAX 3                    // 2-bit DOA confidence
#define FRIEND_THR 3                 // SHCT >= 3 => friendly
#define SHORT_THR 2                  // LIFE_SHORT >= 2 => short-life
#define LONG_THR 2                   // LIFE_LONG >= 2 => long-life
#define STREAM_THR 2                 // PC_STREAM >= 2 => stream/scan
#define SET_STREAM_THR 2             // per-set streaminess threshold
#define DECAY_PERIOD 4096            // periodic decay cadence
#define MISS_BURST_THR 64            // burst misses trigger partial decay
#define PARTIAL_DECAY_CHUNK 64       // number of PC entries to decay

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

// Per-line metadata
struct LineMeta {
    uint8_t rrpv; // 0..3
    uint8_t hits; // 0..3
    uint8_t pf;   // 0/1
};

static LineMeta meta[LLC_SETS][LLC_WAYS];

// Per-set lightweight streaminess (2-bit)
static uint8_t set_stream[LLC_SETS];

// Sampled-set signature table: PC index for training
static uint16_t sample_sig[LLC_SETS / SAMPLE_RATE][LLC_WAYS]; // 11-bit valid if != 0x7FF

// Global PC-indexed predictors
static uint8_t SHCT[PC_IDX_SIZE];          // friendliness (3-bit)
static uint8_t LIFE_SHORT[PC_IDX_SIZE];    // short-life score (2-bit)
static uint8_t LIFE_LONG[PC_IDX_SIZE];     // long-life score (2-bit)
static uint8_t PC_STREAM[PC_IDX_SIZE];     // stream/scan score (2-bit)
static uint8_t DOA_CONF[PC_IDX_SIZE];      // DOA confidence (2-bit)

// Lightweight per-PC stride detector
static uint16_t PC_LAST_LINE[PC_IDX_SIZE]; // low bits of last line address
static int8_t   PC_LAST_STRIDE[PC_IDX_SIZE];
static uint8_t  PC_STR_RUN[PC_IDX_SIZE];   // 0..3

// Object lifetime by PC⊕page signature (2-bit)
static uint8_t OBJ_LIFE[OBJ_SIG_SIZE];

// XDIP (zero-reuse–driven insertion preference)
static int16_t zcnt_ins1 = 0; // leader group preferring RRPV=1
static int16_t zcnt_ins2 = 0; // leader group preferring RRPV=2

// Epoch and burst tracking
static uint64_t event_ctr = 0;
static uint32_t miss_burst = 0;
static uint32_t decay_ptr = 0;

// Helpers
static inline uint32_t pc_index(uint64_t pc) {
    // simple mixing to 11 bits
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return static_cast<uint32_t>(x) & (PC_IDX_SIZE - 1);
}
static inline uint32_t obj_sig(uint32_t pc_idx, uint64_t paddr) {
    uint64_t page = (paddr >> 12); // 4KB page
    uint64_t mix = (static_cast<uint64_t>(pc_idx) << 5) ^ page ^ (page >> 7);
    return static_cast<uint32_t>(mix) & (OBJ_SIG_SIZE - 1);
}
static inline bool is_sampled(uint32_t set) { return (set % SAMPLE_RATE) == 0; }
static inline bool is_leader_ins1(uint32_t set) { return (set % 64) == 1; }
static inline bool is_leader_ins2(uint32_t set) { return (set % 64) == 33; }

static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u8(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_s16(int16_t &x, int16_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_s16(int16_t &x, int16_t minv) { if (x > minv) x--; }

// Partial decay over PC-indexed tables
static inline void decay_some(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = decay_ptr++ & (PC_IDX_SIZE - 1);
        SHCT[idx]       >>= 1;
        LIFE_SHORT[idx] >>= 1;
        LIFE_LONG[idx]  >>= 1;
        PC_STREAM[idx]  >>= 1;
        DOA_CONF[idx]   >>= 1;
    }
}

// Update per-PC stride and per-set streaminess; return whether a strong stream event occurred
static inline bool update_stride_stream(uint32_t pc_idx, uint32_t set, uint64_t line_addr) {
    uint16_t last_line = PC_LAST_LINE[pc_idx];
    int32_t  d = static_cast<int32_t>((line_addr & 0xFFFF) - last_line);
    // wrap-aware small stride (use int8 range)
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
        sat_inc(PC_STREAM[pc_idx], STREAM_MAX);
        if (set_stream[set] < 3) set_stream[set]++;
    } else {
        sat_dec_u8(PC_STREAM[pc_idx]);
        if (set_stream[set] > 0) set_stream[set]--;
    }
    return strong_stream;
}

// XDIP winner: 1 => RRPV=1 insertion, 2 => RRPV=2 insertion
static inline uint8_t xdip_winner(uint32_t set) {
    if (is_leader_ins1(set)) return 1;
    if (is_leader_ins2(set)) return 2;
    // choose the lower zero-reuse counter (fewer deadlines)
    return (zcnt_ins1 <= zcnt_ins2) ? 1 : 2;
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        set_stream[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].rrpv = 3;
            meta[s][w].hits = 0;
            meta[s][w].pf   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
        SHCT[i] = FRIEND_THR;   // neutral
        LIFE_SHORT[i] = 1;      // slight short-life bias
        LIFE_LONG[i]  = 1;      // slight long-life bias
        PC_STREAM[i]  = 1;      // mild stream suspicion
        DOA_CONF[i]   = 0;      // no DOA confidence
        PC_LAST_LINE[i] = 0;
        PC_LAST_STRIDE[i] = 0;
        PC_STR_RUN[i] = 0;
    }
    for (uint32_t s = 0; s < (LLC_SETS / SAMPLE_RATE); s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            sample_sig[s][w] = 0x7FF; // invalid
        }
    }
    for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
        OBJ_LIFE[i] = 1; // neutral
    }
    event_ctr = 0;
    miss_burst = 0;
    decay_ptr = 0;
    zcnt_ins1 = 0;
    zcnt_ins2 = 0;
}

// Victim selection: SRRIP with safe aging
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Prefer any invalid way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Find RRPV==3, else age and retry (bounded)
    for (int iter = 0; iter < 4; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv == 3) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (meta[set][w].rrpv < 3) meta[set][w].rrpv++;
        }
    }
    return 0; // fallback
}

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

    // periodic global decay for agility
    if ((event_ctr % DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_IDX_SIZE; i++) {
            SHCT[i]       >>= 1;
            LIFE_SHORT[i] >>= 1;
            LIFE_LONG[i]  >>= 1;
            PC_STREAM[i]  >>= 1;
            DOA_CONF[i]   >>= 1;
        }
        for (uint32_t i = 0; i < OBJ_SIG_SIZE; i++) {
            OBJ_LIFE[i] >>= 1;
        }
    }

    uint32_t pc_idx = pc_index(PC);
    uint64_t line_addr = (paddr >> 6);

    if (hit) {
        miss_burst = 0;

        // Any reuse quickly clears DOA confidence
        DOA_CONF[pc_idx] = 0;

        // Update stride/stream tracking on hit
        bool stream_evt = update_stride_stream(pc_idx, set, line_addr);
        (void)stream_evt; // not needed directly here

        LineMeta &m = meta[set][way];

        bool is_demand = (type == LOAD) || (type == RFO);
        if (is_demand) {
            // Prefetch first demand hit: clear pf, do not promote
            if (m.pf && m.hits == 0) {
                m.pf = 0;
            }
            if (m.hits < 3) m.hits++;

            // Time-gated multi-hit promotion: only on 2nd+ hit while still young
            if (m.hits >= 2) {
                if (m.rrpv <= 1) {
                    m.rrpv = 0; // MRU on confirmed quick reuse
                } else {
                    // Encourage but avoid MRU if already aged
                    if (m.rrpv > 0) m.rrpv--;
                }
                // Reinforce life predictors
                sat_dec_u8(LIFE_SHORT[pc_idx]);
                sat_inc(LIFE_LONG[pc_idx], LIFE_MAX);
                // Object life reinforcement
                uint32_t os = obj_sig(pc_idx, paddr);
                sat_inc(OBJ_LIFE[os], 3);
                // Friendliness reinforcement
                sat_inc(SHCT[pc_idx], SHCT_MAX);
            } else {
                // First-hit hesitation: cap at RRPV=1
                if (m.rrpv > 1) m.rrpv = 1;
            }
        } else if (type == PREFETCH) {
            // prefetch hits: keep mild promotion only
            if (m.hits < 3) m.hits++;
            if (m.rrpv > 1) m.rrpv = 1;
        } else {
            // writebacks shouldn't influence much
            if (m.rrpv > 1) m.rrpv--;
        }
        return;
    }

    // Miss path (fill): victim was at [set][way] => train victim then insert new
    miss_burst++;
    if ((miss_burst % MISS_BURST_THR) == 0) {
        decay_some(PARTIAL_DECAY_CHUNK);
    }

    // Train with victim (sampled sets only)
    if (is_sampled(set)) {
        uint16_t vsig = sample_sig[set / SAMPLE_RATE][way];
        if (vsig != 0x7FF) {
            uint8_t vhits = meta[set][way].hits;
            if (vhits >= 1) sat_inc(SHCT[vsig], SHCT_MAX);
            else            sat_dec_u8(SHCT[vsig]);

            if (vhits >= 2) {
                sat_dec_u8(LIFE_SHORT[vsig]);
                sat_inc(LIFE_LONG[vsig], LIFE_MAX);
            } else {
                sat_inc(LIFE_SHORT[vsig], LIFE_MAX);
                sat_dec_u8(LIFE_LONG[vsig]);
            }
            // DOA escalation only on zero-hit evictions
            if (vhits == 0) sat_inc(DOA_CONF[vsig], DOA_MAX);

            // XDIP zero-reuse accounting on leaders
            if (is_leader_ins1(set)) {
                if (vhits == 0) sat_inc_s16(zcnt_ins1, 127);
                else            sat_dec_s16(zcnt_ins1, -128);
            } else if (is_leader_ins2(set)) {
                if (vhits == 0) sat_inc_s16(zcnt_ins2, 127);
                else            sat_dec_s16(zcnt_ins2, -128);
            }
        }
    }

    // Compute predictors for the incoming line
    bool stream_evt = update_stride_stream(pc_idx, set, line_addr);
    bool pc_streamy = (PC_STREAM[pc_idx] >= STREAM_THR) || stream_evt;
    bool set_streamy = (set_stream[set] >= SET_STREAM_THR);

    bool friendly = (SHCT[pc_idx] >= FRIEND_THR);
    bool short_life = (LIFE_SHORT[pc_idx] >= SHORT_THR);
    bool long_life  = (LIFE_LONG[pc_idx]  >= LONG_THR);

    uint32_t os = obj_sig(pc_idx, paddr);
    bool obj_long = (OBJ_LIFE[os] >= 2);

    bool is_demand = (type == LOAD) || (type == RFO);
    bool is_pf     = (type == PREFETCH);
    bool doa_high  = (DOA_CONF[pc_idx] == DOA_MAX);

    // Decide insertion RRPV
    uint8_t ins_rrpv = 2; // default neutral
    // Guarded hard-bypass (effect): only for non-writeback and high DOA in streamy set
    if (type != WRITEBACK && doa_high && set_streamy) {
        ins_rrpv = 3; // hard/soft bypass effect
    } else if (pc_streamy || short_life) {
        ins_rrpv = 3; // scans/short life insert cold
    } else if (friendly && (long_life || obj_long)) {
        // Friendly long-life: XDIP chooses between 1 and 2
        ins_rrpv = xdip_winner(set);
    } else if (doa_high) {
        ins_rrpv = 3; // soft-bypass when DOA high but set not strongly streamy
    } else {
        ins_rrpv = 2; // conservative
    }

    if (is_pf && ins_rrpv < 3) ins_rrpv++; // prefetches insert one notch colder

    // Initialize new line metadata
    meta[set][way].rrpv = ins_rrpv;
    meta[set][way].hits = 0;
    meta[set][way].pf   = is_pf ? 1 : 0;

    // Record signature for training in sampled sets
    if (is_sampled(set)) {
        sample_sig[set / SAMPLE_RATE][way] = static_cast<uint16_t>(pc_idx);
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}