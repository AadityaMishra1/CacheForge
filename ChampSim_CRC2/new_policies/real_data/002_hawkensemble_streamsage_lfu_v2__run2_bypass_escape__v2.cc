#include <vector>
#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// -------------------- RRIP core (shared) --------------------
static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP (0..7)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];      // 3 bits semantic
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];    // 2 bits semantic (0..3)
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1 bit

// -------------------- Leader set sampling (64 leaders) --------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 sampled sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 valid only if sampled

// -------------------- Global selector (conservative) --------------------
static constexpr uint8_t GSEL_MAX   = 31;  // 5-bit EWMA-like score
static constexpr uint8_t GSEL_THRES = 16;  // Mode B enabled for followers only if >= THRES
static uint8_t GSEL = 0;                   // defaults to Mode A (Hawkeye)

static inline bool use_modeB(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}
static inline void gsel_inc() { if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec() { if (GSEL > 0) GSEL--; }

// -------------------- Mode A: Hawkeye-like SHiP (leader-A trained) --------------------
#define SHCT_SIZE_BITS 10
#define SHCT_SIZE (1u << SHCT_SIZE_BITS)   // 1024 entries
#define SHCT_MAX 31                         // 5-bit counters
static uint8_t shct_demand[SHCT_SIZE];      // friendly/averse by PC signature
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A per-line bookkeeping for SHiP training (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit semantic
static uint8_t  hawk_used[64][LLC_WAYS];    // 1-bit semantic
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1-bit semantic

// -------------------- Mode B: StreamSage + TinyLFU --------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10-bit semantic
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit semantic (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 4-bit (0..15)

// Tunables (Run2 Bypass+Escape)
static constexpr uint8_t STREAM_CONF_THRESH = 2; // need 2 forward (+1/+2) steps
static constexpr uint8_t HITS_PROMOTE_NS    = 2; // non-stream promote at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 2; // stream escape/promo at 2nd demand hit
static constexpr uint8_t INSERT_WARM_DEPTH  = 2; // near-MRU for friendly/hot
static constexpr uint8_t INSERT_COLD_DEPTH  = 6; // deeper tail for cold
static constexpr uint8_t PC_USE_HOT_THRESH  = 5; // TinyLFU hot carveout for streams -> RRIP=5
static constexpr uint32_t LFU_DECAY_PERIOD  = 2048; // halve every N ops

// Leader-B bookkeeping for selector (64 sets x 16 ways)
static uint8_t b_used[64][LLC_WAYS]; // 1-bit semantic

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
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }

// Forward-run detector: returns true if in stream after update
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (fwd) {
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
    if ((op_count & (LFU_DECAY_PERIOD - 1u)) == 0u) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            pc_use4[i] >>= 1; // halve counts
        }
    }
}

// -------------------- Initialization --------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;    // start at tail
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
    std::memset(b_used, 0, sizeof(b_used));
    GSEL = 0; // default to Hawkeye
}

// -------------------- Victim selection --------------------
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Return any invalid way first
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim: look for RRPV==max, otherwise age and retry (bounded)
    for (uint32_t attempt = 0; attempt <= maxRRPV; attempt++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        rrpv_bump_all(set);
    }
    // Fallback (should not happen)
    return 0;
}

// -------------------- Update state --------------------
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

    maybe_decay_lfu();
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        if (pc_use4[pidx] < 15) pc_use4[pidx]++;
    }

    // Leader training on eviction (process previous occupant before overwrite)
    if (SEL_SAMPLED(set) && hit == 0) {
        uint32_t ls = LEADER_SLOT(set);
        if (LEADER_A(set)) {
            uint16_t sig = hawk_sig[ls][way];
            uint32_t idx = shct_idx(sig);
            if (hawk_is_pref[ls][way]) {
                if (hawk_used[ls][way]) shct_inc(shct_prefetch[idx]);
                else                    shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[ls][way]) shct_inc(shct_demand[idx]);
                else                    shct_dec(shct_demand[idx]);
            }
            // Selector learns: dead in A -> favor B (increase)
            if (!hawk_used[ls][way]) gsel_inc();
            // Clear record
            hawk_used[ls][way] = 0;
            hawk_is_pref[ls][way] = 0;
        } else if (LEADER_B(set)) {
            // Selector learns: dead in B -> disfavor B (decrease)
            if (!b_used[ls][way]) gsel_dec();
            b_used[ls][way] = 0;
        }
    }

    // On hit updates
    if (hit) {
        // mark reuse for leaders
        if (SEL_SAMPLED(set)) {
            uint32_t ls = LEADER_SLOT(set);
            if (LEADER_A(set)) {
                hawk_used[ls][way] = hawk_used[ls][way] | (is_demand(type) ? 1u : 0u);
            } else if (LEADER_B(set)) {
                b_used[ls][way] = b_used[ls][way] | (is_demand(type) ? 1u : 0u);
            }
        }

        // Stream demote-on-touch; escape on 2nd demand hit
        if (stream_lock[set][way]) {
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= (HITS_PROMOTE_STR - 1)) {
                    stream_lock[set][way] = 0;        // escape
                    rrpv_set(set, way, 0);            // promote to MRU
                } else {
                    rrpv_demote(set, way);            // demote on first touch
                }
            } else {
                // prefetch touch: demote conservatively
                rrpv_demote(set, way);
            }
            return;
        }

        // Non-stream multi-hit promotion gate
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= (HITS_PROMOTE_NS - 1)) {
                rrpv_set(set, way, 0); // MRU
            }
        }
        return;
    }

    // Miss/fill: decide mode, stream status, and insertion depth
    bool modeB = use_modeB(set);

    // Stream detection (use PC stride +1/+2 run)
    bool is_stream = false;
    if (is_demand(type) || type == ACCESS_PREFETCH) {
        is_stream = detect_and_update_stream(PC, paddr);
    }

    // TinyLFU hotness
    uint32_t pidx = pc_index(PC);
    bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    uint8_t insert_rrpv = maxRRPV; // default tail
    uint8_t new_stream_lock = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass WB: keep at tail, no promotion path
        insert_rrpv = maxRRPV;
        new_stream_lock = 0;
    } else if (modeB) {
        // Mode B: StreamSage + TinyLFU
        if (type == ACCESS_PREFETCH) {
            insert_rrpv = maxRRPV; // quarantine
            new_stream_lock = is_stream ? 1 : 0;
        } else { // demand
            if (is_stream) {
                // Strong quarantine; allow hot carveout to RRIP=5
                insert_rrpv = pc_hot ? 5 : maxRRPV;
                new_stream_lock = 1;

                // If set pressure is high (no current line at max), gently age all
                bool any_at_max = false;
                for (uint32_t w = 0; w < LLC_WAYS; w++) {
                    if (rrpv[set][w] == maxRRPV) { any_at_max = true; break; }
                }
                if (!any_at_max) rrpv_bump_all(set);
            } else {
                // Non-stream: hot gets warm insert; otherwise cold
                insert_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                new_stream_lock = 0;
            }
        }
    } else {
        // Mode A: Hawkeye-like SHiP
        if (type == ACCESS_PREFETCH) {
            insert_rrpv = maxRRPV; // quarantine
            new_stream_lock = is_stream ? 1 : 0;
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            bool friendly = (shct_demand[idx] >= 2);
            insert_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            // If stream detected, override to tail quarantine
            if (is_stream) {
                insert_rrpv = maxRRPV;
                new_stream_lock = 1;
            }
        }
    }

    // Apply insertion
    rrpv_set(set, way, insert_rrpv);
    stream_lock[set][way] = new_stream_lock;
    hitcnt[set][way] = 0; // reset multi-hit counter

    // Record per-leader metadata for training of the new line
    if (SEL_SAMPLED(set)) {
        uint32_t ls = LEADER_SLOT(set);
        if (LEADER_A(set)) {
            hawk_sig[ls][way] = pc_sig12(PC);
            hawk_used[ls][way] = 0;
            hawk_is_pref[ls][way] = (type == ACCESS_PREFETCH) ? 1u : 0u;
        } else if (LEADER_B(set)) {
            b_used[ls][way] = 0;
        }
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // intentionally blank
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // intentionally blank
}