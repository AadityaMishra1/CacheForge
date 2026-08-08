#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim-CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// -------------------- RRIP core (shared) --------------------
static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP state (0..7)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// Per-line lightweight metadata
// hitcnt: 2b (0..3), stream_lock:1b
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// -------------------- Set sampling & leadership (64 leaders) --------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 sampled sets: low 6 bits match the next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (SEL_SAMPLED(set) ? (set & 63u) : 0xFFFFu); }

// -------------------- Global selector (conservative) --------------------
static constexpr uint8_t GSEL_MAX   = 31;  // 5-bit
static constexpr uint8_t GSEL_THRES = 16;  // followers enable Mode B only when clearly better
static uint8_t GSEL = 0;                   // defaults to Mode A
static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}
static inline void gsel_inc() { if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec() { if (GSEL > 0) GSEL--; }

// -------------------- Mode A: Hawkeye-like SHiP (leader-A trained) --------------------
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit counters
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit counters

// Leader-only line tags for training (64 leader sets x 16 ways = 1024 lines)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit signature (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // 1 if re-referenced while resident (demand only)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1 if inserted by prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// -------------------- Mode B: StreamSage + TinyLFU --------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10 bits

// Per-PC metadata (packed conceptually)
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10 bits semantically
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2 bits (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4 bits (0..15)

// Tunables (retuned for lbm/zeusmp vs irregular balance)
static constexpr uint8_t STREAM_CONF_THRESH = 2; // need 2 consecutive +1/+2 steps
static constexpr uint8_t HITS_PROMOTE_NS    = 2; // non-stream promote on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 2; // stream escape on 2nd demand hit
static constexpr uint8_t INSERT_WARM_DEPTH  = 2; // near-MRU for hot/friendly
static constexpr uint8_t INSERT_COLD_DEPTH  = 6; // near-tail for cold/quarantine
static constexpr uint8_t PC_USE_HOT_THRESH  = 6; // TinyLFU hot threshold
static constexpr uint8_t STREAM_DEMOTE_TOUCH= 1; // demote stream lines on touch
static constexpr uint32_t LFU_DECAY_PERIOD  = 4096;

static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static uint32_t op_count = 0;
static inline void maybe_decay_lfu() {
    op_count++;
    if ((op_count & (LFU_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1; // halve counts
        }
    }
}

// -------------------- Helpers --------------------
static inline void rrpv_bump_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
}
static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}

// -------------------- Initialization --------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
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
    GSEL = 0;
    op_count = 0;
}

// -------------------- Victim selection (RRIP) --------------------
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
    // RRIP victim selection with bounded aging
    for (int iter = 0; iter < 8; iter++) { // 3-bit RRIP => at most 7 bumps needed
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_bump_all(set);
    }
    // Fallback (should not happen)
    return 0;
}

// -------------------- Update replacement state --------------------
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
    maybe_decay_lfu();
    if (is_demand(type)) {
        uint32_t idx = pc_index(PC);
        sat_inc_u4(pc_use4[idx]);
    }

    uint32_t slot = LEADER_SLOT(set);

    if (hit) {
        // Leader bookkeeping: demand re-reference -> mark used
        if (SEL_SAMPLED(set) && is_demand(type)) {
            hawk_used[slot][way] = 1;
        }

        // Stream-aware handling
        if (stream_lock[set][way]) {
            // Stream locked: demote on touch; escape only on 2nd demand hit
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= (uint8_t)(HITS_PROMOTE_STR - 1)) {
                    stream_lock[set][way] = 0;       // escape stream jail
                    rrpv_set(set, way, 0);           // promote to MRU
                } else {
                    uint8_t newv = rrpv[set][way] + STREAM_DEMOTE_TOUCH;
                    rrpv_set(set, way, newv);
                }
            } else {
                // Prefetch touches do not promote; mild demotion to keep scan resistance
                uint8_t newv = rrpv[set][way] + STREAM_DEMOTE_TOUCH;
                rrpv_set(set, way, newv);
            }
            return;
        }

        // Non-stream: multi-hit promotion (only on 2nd+ demand hit)
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= (uint8_t)(HITS_PROMOTE_NS - 1)) {
                rrpv_set(set, way, 0); // MRU
            } else {
                rrpv_promote(set, way); // slight promotion
            }
        }
        return;
    }

    // -------- Miss path: train leaders on eviction of old line before reinit --------
    if (SEL_SAMPLED(set)) {
        // Use the metadata of the victim way being replaced in this sampled set
        uint16_t old_sig = hawk_sig[slot][way];
        uint8_t  old_used = hawk_used[slot][way];
        uint8_t  old_is_pref = hawk_is_pref[slot][way];

        // Global selector update: dead eviction = penalty for leader's policy
        if (LEADER_A(set)) {
            if (old_used == 0) gsel_inc(); else gsel_dec();
        } else if (LEADER_B(set)) {
            if (old_used == 0) gsel_dec(); else gsel_inc();
        }

        // Train SHiP only with leader-A sets
        if (LEADER_A(set)) {
            uint32_t idx = shct_idx(old_sig);
            if (old_is_pref) {
                if (old_used) shct_inc(shct_prefetch[idx]); else shct_dec(shct_prefetch[idx]);
            } else {
                if (old_used) shct_inc(shct_demand[idx]); else shct_dec(shct_demand[idx]);
            }
        }
    }

    // -------- Insert new line --------
    // Default initialization
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    uint8_t ins_rrpv = INSERT_COLD_DEPTH; // default cold
    bool modeB = use_modeB(set);

    if (type == ACCESS_PREFETCH) {
        // Always quarantine prefetches at tail
        ins_rrpv = maxRRPV;
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert cold
        ins_rrpv = INSERT_COLD_DEPTH;
    } else {
        // Demand fill
        if (modeB) {
            bool is_stream = detect_and_update_stream(PC, paddr);
            uint32_t pidx = pc_index(PC);
            bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

            if (is_stream) {
                // Selective bypass via hard-tail insert; hot PCs get slightly shallower tail
                ins_rrpv = pc_hot ? (uint8_t)(maxRRPV - 1) : maxRRPV;
                stream_lock[set][way] = 1; // never promote until escape on 2nd demand hit
            } else {
                // Non-stream: hot PCs get warm insert, others cold
                ins_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        } else {
            // Mode A: Hawkeye-like SHiP decision (prefetch already handled)
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t ctr = shct_demand[idx];
            // Friendly if counter >= mid; insert warm; otherwise cold
            ins_rrpv = (ctr >= (SHCT_MAX / 2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    rrpv_set(set, way, ins_rrpv);

    // Record leader metadata for the new line (for both leader types; SHiP trains only on A)
    if (SEL_SAMPLED(set)) {
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        hawk_used[slot][way] = 0;
    }
}

// -------------------- Stats (keep blank) --------------------
void PrintStats() {}
void PrintStats_Heartbeat() {}