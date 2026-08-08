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

// ---------------- Leader-set sampling (64 leaders: 32 A, 32 B) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63 for sampled sets

// ---------------- Tunables ----------------
static constexpr uint8_t  maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;   // near-tail for averse
static constexpr uint8_t  INSERT_TAIL_DEPTH   = 7;   // hard tail (quarantine)
static constexpr uint8_t  STREAM_CONF_THRESH  = 2;   // +1/+2 steps for stream
static constexpr uint8_t  HITS_PROMOTE_NS     = 2;   // non-stream MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;   // stream escape on 2nd demand hit
static constexpr uint32_t BANDIT_EPOCH        = 4096;// selector epoch (ops)
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;   // per-set enable threshold
static constexpr uint8_t  SHCT_FRIENDLY_THRES = 16;  // Mode A friendly cutoff (0..31)

// ---------------- Per-line metadata (conceptual packing) ----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-set selector state ----------------
static int8_t bandit_score[LLC_SETS]; // [-8..7] follower bias to Mode B

// ---------------- Mode A: compact Hawkeye-like SHiP ----------------
#define SHCT_BITS 11
#define SHCT_SIZE (1u << SHCT_BITS) // 2048 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // lightweight 12-bit signature
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Per-line tags for leader-A sets only (64 x 16 lines)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
static uint8_t  hawk_used[64][LLC_WAYS];    // 0/1 reused
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 0/1 filled by prefetch

// ---------------- Mode B: stride-run + PC-Bloom deadness ----------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B -> 10b

// per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b stored in 16b
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_cold_streak[PC_TBL_SIZE]; // 2b (0..3) increments on demand misses
static uint8_t  pc_hot4[PC_TBL_SIZE];        // 4b Tiny-hotness (override Bloom)

// Tiny Bloom filter (1024 bits, 2 hashes)
static constexpr uint32_t BLOOM_BITS = 1024;
static constexpr uint32_t BLOOM_WORDS = BLOOM_BITS / 32;
static uint32_t bloom[BLOOM_WORDS];

static inline uint32_t bloom_h1(uint64_t pc) { return (uint32_t)((pc ^ (pc >> 2) ^ (pc >> 7)) & (BLOOM_BITS - 1u)); }
static inline uint32_t bloom_h2(uint64_t pc) { return (uint32_t)((pc ^ (pc >> 13) ^ (pc << 5) ^ (pc >> 19)) & (BLOOM_BITS - 1u)); }
static inline void bloom_set(uint64_t pc) {
    uint32_t i1 = bloom_h1(pc), i2 = bloom_h2(pc);
    bloom[i1 >> 5] |= (1u << (i1 & 31u));
    bloom[i2 >> 5] |= (1u << (i2 & 31u));
}
static inline bool bloom_test(uint64_t pc) {
    uint32_t i1 = bloom_h1(pc), i2 = bloom_h2(pc);
    bool b1 = (bloom[i1 >> 5] >> (i1 & 31u)) & 1u;
    bool b2 = (bloom[i2 >> 5] >> (i2 & 31u)) & 1u;
    return b1 && b2;
}

// ---------------- Helpers ----------------
static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) { rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val; }
static inline void rrpv_promote(uint32_t set, uint32_t way)         { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)          { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

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

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Global selector bookkeeping ----------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void selector_epoch_end() {
    // Compare leader hits and nudge all follower sets
    if (leaderB_hits_epoch > leaderA_hits_epoch) {
        for (uint32_t s = 0; s < LLC_SETS; s++) if (!SEL_SAMPLED(s)) { if (bandit_score[s] < 7) bandit_score[s]++; }
    } else if (leaderA_hits_epoch > leaderB_hits_epoch) {
        for (uint32_t s = 0; s < LLC_SETS; s++) if (!SEL_SAMPLED(s)) { if (bandit_score[s] > -8) bandit_score[s]--; }
    }
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Initialization ----------------
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
    std::memset(pc_cold_streak, 0, sizeof(pc_cold_streak));
    std::memset(pc_hot4, 0, sizeof(pc_hot4));
    std::memset(bloom, 0, sizeof(bloom));

    op_count = leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ---------------- Victim selection (RRIP with invalid check) ----------------
static inline uint32_t rrip_choose_victim(uint32_t set, const BLOCK* current_set) {
    // 1) If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (!current_set[w].valid) return w;
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] == maxRRPV) return w;
    // 3) Bounded aging passes to ensure termination
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] == maxRRPV) return w;
        for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
    // fallback: pick the oldest (max rrpv) deterministically
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    uint32_t way = rrip_choose_victim(set, current_set);

    // Train Mode A (negative) on A-leader evictions
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        if (current_set[way].valid) {
            uint16_t sig = hawk_sig[slot][way];
            uint8_t used = hawk_used[slot][way];
            uint8_t was_pref = hawk_is_pref[slot][way];
            if (!used) {
                if (was_pref) shct_dec(shct_prefetch[shct_idx(sig)]);
                else          shct_dec(shct_demand[shct_idx(sig)]);
            }
            // reset tag (not required but keeps state clean)
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = 0;
            hawk_sig[slot][way] = 0;
        }
    }
    return way;
}

// ---------------- Update on hit/fill ----------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // periodic selector epoch
    op_count++;
    if ((op_count % BANDIT_EPOCH) == 0) selector_epoch_end();

    if (is_writeback(type)) return; // never bypass or modify on WB

    // Detect stream for this PC (only meaningful for demand)
    bool is_stream = false;
    if (is_demand(type)) {
        is_stream = detect_and_update_stream(PC, paddr);
    }

    bool useB = modeB_enabled(set);

    // Train/select leader hit statistics
    if (hit && is_demand(type)) {
        if (LEADER_A(set)) leaderA_hits_epoch++;
        else if (LEADER_B(set)) leaderB_hits_epoch++;
    }

    // Per-PC cold/hot updates (demand accesses only)
    if (is_demand(type)) {
        uint32_t pidx = pc_index(PC);
        if (hit) {
            pc_cold_streak[pidx] = 0;
            if (pc_hot4[pidx] < 15) pc_hot4[pidx]++;
        } else { // miss
            if (pc_cold_streak[pidx] < 3) pc_cold_streak[pidx]++;
            if (pc_cold_streak[pidx] >= 3) bloom_set(PC);
        }
    }

    if (hit) {
        // ---------------- on hit ----------------
        if (useB) {
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (stream_lock[set][way]) {
                    // demote on first touch to harden quarantine, escape on 2nd+
                    if (hitcnt[set][way] < HITS_PROMOTE_STR) {
                        rrpv_demote(set, way); // gentle demotion on first demand hit
                    } else {
                        stream_lock[set][way] = 0;
                        rrpv_set(set, way, 0); // MRU on 2nd demand hit
                    }
                } else {
                    // non-stream: promote only after enough reuse, else gentle bump
                    if (hitcnt[set][way] >= HITS_PROMOTE_NS) rrpv_set(set, way, 0);
                    else rrpv_promote(set, way);
                }
            }
            // PREFETCH hits are observed as demand hits in ChampSim; type==PREFETCH doesn't "hit"
        } else {
            // Mode A: classic MRU promotion on any demand hit
            if (is_demand(type)) {
                rrpv_set(set, way, 0);
            }
            // Positive training for A-leaders on first reuse
            if (LEADER_A(set)) {
                uint32_t slot = LEADER_SLOT(set);
                if (!hawk_used[slot][way]) {
                    hawk_used[slot][way] = 1;
                    if (hawk_is_pref[slot][way]) shct_inc(shct_prefetch[shct_idx(hawk_sig[slot][way])]);
                    else                         shct_inc(shct_demand [shct_idx(hawk_sig[slot][way])]);
                }
            }
        }
        return;
    }

    // ---------------- on fill (miss) ----------------
    hitcnt[set][way] = 0;
    if (useB) {
        if (is_prefetch(type)) {
            // prefetch quarantine at hard tail
            stream_lock[set][way] = 1;
            rrpv_set(set, way, INSERT_TAIL_DEPTH);
        } else {
            if (is_stream) {
                // stream quarantine (acts as bypass)
                stream_lock[set][way] = 1;
                rrpv_set(set, way, INSERT_TAIL_DEPTH);
            } else {
                stream_lock[set][way] = 0;
                // PC-Bloom deadness with hot override
                bool cold_bloom = bloom_test(PC);
                uint8_t hot = pc_hot4[pc_index(PC)];
                if (cold_bloom && hot < 2) rrpv_set(set, way, INSERT_COLD_DEPTH);
                else                        rrpv_set(set, way, INSERT_WARM_DEPTH);
            }
        }
    } else {
        // Mode A insertion (Hawkeye-like)
        if (is_prefetch(type)) {
            rrpv_set(set, way, INSERT_TAIL_DEPTH);
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            bool friendly = (shct_demand[idx] >= SHCT_FRIENDLY_THRES);
            rrpv_set(set, way, friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
            if (LEADER_A(set)) {
                uint32_t slot = LEADER_SLOT(set);
                hawk_sig[slot][way] = sig;
                hawk_used[slot][way] = 0;
                hawk_is_pref[slot][way] = 0;
            }
        }
        if (LEADER_A(set) && is_prefetch(type)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = 1;
        }
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}