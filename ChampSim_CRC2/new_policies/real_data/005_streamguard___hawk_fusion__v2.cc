#include <vector>
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
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables ---------------------------------------
// RRIP depths (3-bit)
static constexpr uint8_t maxRRPV             = 7;
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for hot-PC streams

// Stream detector (+1/+2 forward run)
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // after 2 fwd steps: arm
static constexpr uint8_t STREAM_LOCK_THRESH  = 3;  // after 3 fwd steps: lock

// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS      = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR     = 2; // stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT = 1; // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH  = 1; // demote quarantined streams on touch

// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Hawkeye

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt(demand hits):2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode A unless confident
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_good = 0;       // hits - misses on leader A sets
static int32_t leaderB_good = 0;       // hits - misses on leader B sets
static uint64_t access_count = 0;

// Keep the chosen victim info for SHiP training on fill
static uint8_t pending_victim_way[LLC_SETS];
static uint8_t pending_victim_valid[LLC_SETS];

// ---------------- Mode A (Hawkeye-lite via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b hash
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-A/B per-line training buffers (only for 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamGuard PC state) -------------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: enable B only if global gate prefers it and local bandit is confident
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Train SHCT on eviction for leader sets
static inline void train_hawkeye_on_eviction(uint32_t set, uint32_t way, uint8_t was_valid) {
    if (!was_valid) return;
    if (!SEL_SAMPLED(set)) return;
    uint32_t slot = LEADER_SLOT(set);
    uint16_t sig = hawk_sig[slot][way] & 0x0FFFu;
    uint32_t idx = shct_idx(sig);
    if (hawk_is_pref[slot][way]) {
        if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
        else                      shct_dec(shct_prefetch[idx]);
    } else {
        if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
        else                      shct_dec(shct_demand[idx]);
    }
    // clear reuse for safety
    hawk_used[slot][way] = 0;
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        pending_victim_way[s] = 0;
        pending_victim_valid[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    prefer_B = false;
    leaderA_good = 0;
    leaderB_good = 0;
    access_count = 0;
}

// ---------------- Victim selection (RRIP) -------------------------
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
        if (!current_set[w].valid) {
            pending_victim_valid[set] = 0; // no training needed
            pending_victim_way[set] = w;
            return w;
        }
    }
    // RRIP: search for maxRRPV; if none, age all and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                pending_victim_valid[set] = 1; // a valid victim will be evicted
                pending_victim_way[set] = w;
                return w;
            }
        }
        // Age all lines (saturate at maxRRPV)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// ---------------- Update replacement state ------------------------
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

    // Leader reward accounting (non-writeback)
    if (!is_writeback(type)) {
        if (LEADER_A(set)) {
            leaderA_good += hit ? 1 : -1;
        } else if (LEADER_B(set)) {
            leaderB_good += hit ? 1 : -1;
        }
    }

    // Global epoch gate
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) {
        prefer_B = ((leaderB_good - leaderA_good) > GATE_MARGIN);
        leaderA_good = 0;
        leaderB_good = 0;
    }

    // TinyLFU: demand uses bump frequency; periodic decay
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        sat_inc_u4(pc_use4[pidx]);
    }
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }

    bool use_modeB = modeB_enabled(set);
    uint32_t pidx = pc_index(PC);
    uint16_t cur_line = line10(paddr);

    // Stream detector update (+1/+2 forward run)
    if (!is_writeback(type)) {
        uint16_t last = pc_last_line10[pidx];
        if (last != 0xFFFFu) {
            int16_t delta = (int16_t)cur_line - (int16_t)last;
            if (delta == 1 || delta == 2) {
                if (pc_stream_conf[pidx] < 3) pc_stream_conf[pidx]++;
            } else {
                pc_stream_conf[pidx] = 0;
            }
        }
        pc_last_line10[pidx] = cur_line;
    }

    // SHCT lookup
    uint16_t sig = pc_sig12(PC);
    uint32_t sidx = shct_idx(sig);
    uint8_t shct_val = is_prefetch(type) ? shct_prefetch[sidx] : shct_demand[sidx];
    bool pc_dead = (shct_val <= 1);
    bool pc_hot_lfu = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    // Mode-based per-set confidence shaping
    if (!is_writeback(type)) {
        if (use_modeB) {
            if (hit) sat_inc_i8(bandit_score[set]);
            else     sat_dec_i8(bandit_score[set]);
        }
    }

    // HIT path
    if (hit) {
        // Leader reuse mark for SHiP
        if (SEL_SAMPLED(set)) {
            uint32_t slot = LEADER_SLOT(set);
            // mark reuse only on demand hits
            if (is_demand(type)) hawk_used[slot][way] = 1;
        }

        // Multi-hit promotion gating
        if (is_demand(type)) {
            // Count only demand hits
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            bool is_stream_q = (stream_tag[set][way] != 0);
            uint8_t thr = is_stream_q ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (is_stream_q && pc_hot_lfu) thr = HITS_PROMOTE_STR_HOT;

            if (hitcnt[set][way] >= thr) {
                rrpv_set(set, way, 0);
                // escape quarantine after promotion
                stream_tag[set][way] = 0;
            } else {
                // Pre-promotion behavior
                if (is_stream_q) {
                    // keep quarantined near tail
                    uint8_t v = rrpv[set][way];
                    for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) {
                        if (v < maxRRPV) v++;
                    }
                    rrpv_set(set, way, v);
                } else {
                    // gentle toward MRU
                    if (rrpv[set][way] > 0) rrpv[set][way]--;
                }
            }
        } else {
            // Prefetch hits: do not promote; keep quarantine if any
            if (stream_tag[set][way]) {
                uint8_t v = std::min<uint8_t>(maxRRPV, (uint8_t)(rrpv[set][way] + STREAM_DEMOTE_TOUCH));
                rrpv_set(set, way, v);
            }
        }
        return;
    }

    // MISS / FILL path (not a writeback bypass)
    if (is_writeback(type)) {
        // Writebacks should not change policy state; insert at MRU-like to preserve data? Do nothing.
        return;
    }

    // Train Hawkeye on eviction for sampled sets before overwriting metadata
    train_hawkeye_on_eviction(set, pending_victim_way[set], pending_victim_valid[set]);

    // Choose insertion policy
    uint8_t ins_rrpv = INSERT_WARM_DEPTH;
    uint8_t ins_stream_tag = 0;

    if (use_modeB) {
        // StreamGuard path
        uint8_t conf = pc_stream_conf[pidx];
        bool locked = (conf >= STREAM_LOCK_THRESH);
        bool armed  = (conf >= STREAM_ARM_THRESH);

        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            ins_stream_tag = 1;
        } else if (locked) {
            ins_rrpv = pc_hot_lfu ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
            ins_stream_tag = 1;
        } else if (armed && pc_dead) {
            // aggressive near-tail for likely scan
            ins_rrpv = STREAM_TAIL_DEPTH;
            ins_stream_tag = 1;
        } else {
            // irregular / reuse-prone under PC gate
            ins_rrpv = pc_dead ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            ins_stream_tag = 0;
        }
    } else {
        // Hawkeye-lite path
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            ins_stream_tag = 1;
        } else {
            ins_rrpv = pc_dead ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            ins_stream_tag = 0;
        }
    }

    // Perform insertion
    rrpv_set(set, way, ins_rrpv);
    stream_tag[set][way] = ins_stream_tag;
    hitcnt[set][way] = 0;

    // Record Hawkeye fill metadata for leaders
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}