#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 sets) -------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for sampled

// ---------------- Tunables (StreamGuard+ refined) -----------------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COOL_DEPTH   = 5;   // moderate tail
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;   // hard tail
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream: short-reuse override on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH = 1;   // demote stream-tag on any touch
static constexpr uint8_t  PC_USE_HOT_THRESH   = 9;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH   = 2;   // 2b coldness (>=2 => cold)
static constexpr uint32_t LFU_DECAY_PERIOD    = 4096;// slower decay for gcc/omnetpp stability
static constexpr uint32_t BANDIT_EPOCH        = 8192;// selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 2;   // require two B-wins before enabling
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set bandit threshold

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b (demand hits), stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set/global selectors -----------------------
static int8_t   bandit_score[LLC_SETS]; // [-8..7]
static uint8_t  GSEL = 0;               // global gate
static uint64_t access_count = 0;
static int32_t  leaderA_score = 0;      // hits - misses in leader A
static int32_t  leaderB_score = 0;      // hits - misses in leader B

// ---------------- Mode A (Hawkeye-like via tiny SHCT) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static constexpr uint8_t SHCT_THRES_ALIVE = 16; // >= alive; else dead

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader A per-line buffers for training (only 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

// ---------------- Mode B (Stream + TinyLFU) ----------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B -> 10b

// Per-PC state: 10b last_line, 2b stream_conf, 4b use, 2b coldness
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE];
static uint8_t  pc_use4[PC_TBL_SIZE];
static uint8_t  pc_cold2[PC_TBL_SIZE];

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_u5(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// +1/+2 forward-run detector with 2-step confidence; reset on non-forward
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else         pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_A(set)) return false;
    if (LEADER_B(set)) return true;
    // Followers: require both global and per-set gates
    if (GSEL >= GSEL_THRES && bandit_score[set] >= MODEB_ENABLE_THRESH)
        return true;
    return false;
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand,   0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
        pc_cold2[i] = 0;
    }
    GSEL = 0;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// ---------------- Victim selection (SRRIP) -----------------------
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
    // SRRIP victim search with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// ---------------- State update -----------------------------------
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
    (void)cpu; (void)victim_addr;

    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    bool useB    = modeB_enabled(set);

    // Per-access TinyLFU update and stream observation (skip writebacks)
    bool stream_conf = false;
    if (!is_writeback(type)) {
        stream_conf = detect_and_update_stream(PC, paddr);
        if (is_demand(type)) {
            uint32_t pidx = pc_index(PC);
            sat_inc_u4(pc_use4[pidx]);
        }
    }

    // Periodic TinyLFU decay
    access_count++;
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
            sat_dec_u2(pc_cold2[i]);
        }
    }
    // Selector epoch update
    if ((access_count % BANDIT_EPOCH) == 0) {
        if (leaderB_score > leaderA_score) sat_inc_u5(GSEL, GSEL_MAX);
        else                                sat_dec_u5(GSEL);
        leaderA_score = 0;
        leaderB_score = 0;
    }

    if (hit) {
        // Leader accounting on hit
        if (leaderA) leaderA_score++;
        if (leaderB) leaderB_score++;
        if (!useB && !leaderA && !leaderB) {
            // No bandit update for Mode A in followers
        } else if (useB && !leaderB && !leaderA) {
            sat_inc_i8(bandit_score[set]);
        }

        // Demand vs prefetch handling
        bool demand = is_demand(type);
        if (demand) {
            // demand hit increases demand-hit counter up to 3
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            // PC coldness cools down on useful demand hit
            uint32_t pidx = pc_index(PC);
            sat_dec_u2(pc_cold2[pidx]);
        }

        // Stream demote-on-touch unless short-reuse override triggers
        if (stream_tag[set][way]) {
            if (demand && hitcnt[set][way] >= HITS_PROMOTE_STR) {
                // Short-reuse override: clear stream tag
                stream_tag[set][way] = 0;
            } else {
                // quarantine/demote to resist scans
                if (STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
            }
        }

        // Multi-hit promotion: only on 2nd+ demand hit
        if (demand && hitcnt[set][way] >= HITS_PROMOTE_NS) {
            rrpv_promote(set, way);
        }

        return;
    }

    // Miss path: leader accounting on miss
    if (leaderA) leaderA_score--;
    if (leaderB) leaderB_score--;
    if (useB && !leaderA && !leaderB) {
        sat_dec_i8(bandit_score[set]);
    }

    // Hawkeye training for leader A: learn from evicted block (old occupant of 'way')
    if (leaderA) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t vsig = hawk_sig[slot][way];
        uint8_t  vuse = hawk_used[slot][way];
        uint8_t  vpref = hawk_is_pref[slot][way];
        if (vsig != 0) {
            uint32_t idx = shct_idx(vsig);
            if (vpref) {
                if (vuse) shct_inc(shct_prefetch[idx]);
                else      shct_dec(shct_prefetch[idx]);
            } else {
                if (vuse) shct_inc(shct_demand[idx]);
                else      shct_dec(shct_demand[idx]);
            }
        }
        // reset used marker for the new fill (will set below)
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1u : 0u;
        hawk_sig[slot][way] = pc_sig12(PC);
    }

    // PC coldness raises on miss for demands
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u2(pc_cold2[pidx]);
    }

    // Insertion policy
    uint8_t ins_rrpv  = INSERT_COLD_DEPTH;
    uint8_t ins_st    = 0; // stream tag for quarantine/stream
    if (is_writeback(type)) {
        ins_rrpv = INSERT_WARM_DEPTH; // never bypass WBs
        ins_st   = 0;
    } else if (useB || leaderB) {
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            ins_st   = 1;
        } else if (stream_conf) {
            // Strong stream/scan: hard-tail
            ins_rrpv = STREAM_TAIL_DEPTH;
            ins_st   = 1;
        } else {
            uint32_t pidx = pc_index(PC);
            bool cold_pc = (pc_cold2[pidx] >= PC_COLD_STRICT_TH);
            bool hot_pc  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            if (cold_pc)      ins_rrpv = INSERT_COLD_DEPTH;
            else if (hot_pc)  ins_rrpv = INSERT_WARM_DEPTH;
            else              ins_rrpv = INSERT_COOL_DEPTH;
            ins_st = 0;
        }
    } else { // Mode A (Hawkeye)
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t  ctr = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];
        bool alive = (ctr >= SHCT_THRES_ALIVE);
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH; // quarantine prefetch
            ins_st   = 1;
        } else if (alive) {
            ins_rrpv = INSERT_WARM_DEPTH;
            ins_st   = 0;
        } else {
            ins_rrpv = STREAM_TAIL_DEPTH; // predicted dead -> tail
            ins_st   = 0;
        }
    }

    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way]    = 0;
    stream_tag[set][way]= ins_st;
}

// Print end-of-simulation statistics
void PrintStats() {
    // (blank by design)
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // (blank by design)
}