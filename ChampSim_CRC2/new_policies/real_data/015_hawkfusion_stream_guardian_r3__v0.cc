#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamGuardian
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for sampled sets

// ---------------- Tunables (balanced for lbm/mcf/gcc/omnetpp) -----
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail insertion
static constexpr uint8_t  STREAM_TAIL_DEPTH   = 7;   // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_SHALLOW_TAIL = 5;   // slightly shallower for hot PCs
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 forward steps
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH   = 6;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH   = 2;   // 0..3 (>=2 => cold)
static constexpr uint32_t LFU_DECAY_PERIOD    = 2048;// periodic LFU/coldness decay
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch
static constexpr uint8_t  GSEL_MAX            = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES          = 16;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set enable

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set and global selectors -------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B
static uint8_t GSEL = 0;               // global gate (0..31)

// ---------------- Mode A (Hawkeye-like via tiny SHiP) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b PC signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A per-line training buffers (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (Stream-Guardian PC tables) --------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    // +1/+2 forward-run detector with 2-step confidence
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward  = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;   // force Mode B on B leaders
    if (LEADER_A(set)) return false;  // force Mode A on A leaders
    // Followers enable Mode B only if both global and local scores are favorable
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Periodic housekeeping --------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping() {
    op_count++;
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            if (pc_use4[i] > 0) pc_use4[i] >>= 1; // decay TinyLFU
            if (pc_cold2[i] > 0) pc_cold2[i]--;   // relax coldness over time
        }
    }
    if ((op_count & (BANDIT_EPOCH - 1u)) == 0u) {
        if (leaderB_hits_epoch > leaderA_hits_epoch) { if (GSEL < GSEL_MAX) GSEL++; }
        else { if (GSEL > 0) GSEL--; }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;
    }
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
}

// ---------------- Victim selection (RRIP) -------------------------
static inline uint32_t rrip_get_victim(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes (ensure termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: choose the way with largest RRPV
    uint32_t victim = 0, best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_get_victim(set, current_set);
}

// ---------------- Update state on hit/fill ------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    periodic_housekeeping();

    const bool demand = is_demand(type);
    const bool pref   = is_prefetch(type);
    const bool wb     = is_writeback(type);

    // Ignore writebacks for policy decisions (never bypass WBs)
    if (wb) return;

    // Per-PC updates (TinyLFU + stride run) on demand
    if (demand) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
        // Update stream detector; used below on both hit and fill
        (void)detect_and_update_stream(PC, paddr);
    }

    // Mode selection for this access
    const bool modeB = modeB_enabled(set);

    if (hit) {
        // Multi-hit promotion gating
        if (stream_lock[set][way]) {
            // Quarantined stream: demote-on-touch; escape on 2nd demand hit
            if (demand) {
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, 0); // escape to MRU
                } else {
                    rrpv_demote(set, way); // keep near tail
                }
            } else {
                // Prefetch hit inside quarantine: keep demoted
                rrpv_demote(set, way);
            }
        } else {
            if (demand) {
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU on 2nd+ demand hit
                } // else: no early promotion
            }
        }
        // Saturate local hit count
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        // Train leader stats for selector
        if (LEADER_A(set)) leaderA_hits_epoch++;
        if (LEADER_B(set)) leaderB_hits_epoch++;

        // Follower local bandit nudging (approximate credit assignment)
        if (!SEL_SAMPLED(set)) {
            if (modeB) sat_inc_i8(bandit_score[set]);
            else       sat_dec_i8(bandit_score[set]);
        }

        // Mode A positive training on first reuse in leader-A sets
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            if (hawk_used[slot][way] == 0) {
                uint16_t sig = hawk_sig[slot][way];
                if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[shct_idx(sig)]);
                else                         shct_inc(shct_demand[shct_idx(sig)]);
                hawk_used[slot][way] = 1;
            }
        }

        // PC coldness relaxes on reuse
        if (demand) sat_dec_u2(pc_cold2[pc_index(PC)]);

        return;
    }

    // Miss and fill path (no writebacks here)
    // Negative training for prior occupant in leader-A sets (dead blocks)
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (hawk_used[slot][way] == 0) {
            uint16_t oldsig = hawk_sig[slot][way];
            if (hawk_is_pref[slot][way]) shct_dec(shct_prefetch[shct_idx(oldsig)]);
            else                         shct_dec(shct_demand[shct_idx(oldsig)]);
        }
        // Record new signature/flags for training
        uint16_t newsig = pc_sig12(PC);
        hawk_sig[slot][way]     = newsig;
        hawk_is_pref[slot][way] = pref ? 1 : 0;
        hawk_used[slot][way]    = 0;
    }

    // Determine insertion policy
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t lock_stream = 0;

    if (pref) {
        // Prefetches are quarantined at tail
        ins_depth = STREAM_TAIL_DEPTH;
        lock_stream = 1;
    } else if (modeB) {
        // Mode B: Stream-Guardian
        bool is_stream = detect_and_update_stream(PC, paddr);
        uint32_t pcidx = pc_index(PC);
        bool pc_hot  = (pc_use4[pcidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pcidx] == 0);

        if (is_stream) {
            // Logical bypass: hard-tail insert; allow shallow tail for hot PCs
            ins_depth = pc_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            lock_stream = 1; // demote-on-touch; escape on 2nd hit
            // Streaming demand fill is likely dead => increase coldness
            sat_inc_u2(pc_cold2[pcidx]);
        } else {
            // Non-stream: warm only if PC is both hot and not cold
            ins_depth = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock_stream = 0;
            if (!pc_hot) sat_inc_u2(pc_cold2[pcidx]); // reinforce coldness for noisy PCs
        }
    } else {
        // Mode A: Hawkeye-like using tiny SHCT
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        bool friendly = demand ? (shct_demand[idx] >= 16) : (shct_prefetch[idx] >= 16);
        ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        lock_stream = 0;
        if (!friendly && demand) sat_inc_u2(pc_cold2[pc_index(PC)]); // cold PCs insert colder
    }

    rrpv_set(set, way, ins_depth);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock_stream;
}

// ---------------- Printing hooks (silent) -------------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}