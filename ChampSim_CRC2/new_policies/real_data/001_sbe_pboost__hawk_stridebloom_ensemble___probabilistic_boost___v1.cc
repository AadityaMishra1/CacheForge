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

// ---------------- Tunables (retuned) ----------------
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail for averse
static constexpr uint8_t  INSERT_TAIL_DEPTH   = 7;    // hard tail (quarantine)
static constexpr uint8_t  HITS_PROMOTE_NS     = 3;    // non-stream MRU on 3rd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream/prefetch escape on 2nd demand hit
static constexpr uint32_t BANDIT_EPOCH        = 2048; // selector epoch (ops)
static constexpr int8_t   MODEB_THRESH        = 3;    // enable Mode B for followers when bias >= +3
static constexpr uint8_t  SHCT_FRIENDLY_THRES = 16;   // Mode A friendly cutoff (0..31)

// ---------------- Per-line metadata (conceptual packing) ----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-slot selector state (64 slots) ----------------
static int8_t bandit_bias[64];          // [-8..+7] bias toward Mode B
static uint32_t leader_hits_A[64];      // epoch hit counts for leader A sets
static uint32_t leader_hits_B[64];      // epoch hit counts for leader B sets

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
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3) conf (2 => run_len>=2, 3 saturate)
static uint8_t  pc_stream_extra[PC_TBL_SIZE];// 2b extra run beyond 3 (counts 0..3 as 3..6)
static uint8_t  pc_cold_streak[PC_TBL_SIZE]; // 2b (miss streak up to 3)
static uint8_t  pc_hot4[PC_TBL_SIZE];        // 4b tiny hotness (override Bloom)

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

static inline uint8_t hash_pct(uint64_t pc, uint64_t line10val) {
    // Deterministic 0..99 hash for "probabilistic" gating
    uint64_t x = pc ^ (pc >> 17) ^ ((uint64_t)line10val << 1) ^ (0x9E3779B97F4A7C15ull);
    x ^= (x >> 33); x *= 0xff51afd7ed558ccdull;
    x ^= (x >> 33); x *= 0xc4ceb9fe1a85ec53ull;
    x ^= (x >> 33);
    return (uint8_t)(x % 100u);
}

static inline uint32_t detect_run_len_and_update(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    // compute delta in 10-bit ring
    uint16_t d = (uint16_t)((ln - last) & 0x03FFu);
    bool forward = (d == 1u) || (d == 2u);
    if (forward) {
        if (pc_stream_conf[idx] < 3) {
            pc_stream_conf[idx]++;
        } else {
            // already saturated -> count extra steps (up to +3)
            if (pc_stream_extra[idx] < 3) pc_stream_extra[idx]++;
        }
    } else {
        pc_stream_conf[idx]  = 0;
        pc_stream_extra[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    uint32_t run_len = pc_stream_conf[idx];
    if (run_len == 3) run_len = 3 + pc_stream_extra[idx]; // 3..6
    return run_len;
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_bias[LEADER_SLOT(set)] >= MODEB_THRESH);
}

// ---------------- Global selector bookkeeping ----------------
static uint32_t op_count = 0;

void InitReplacementState() {
    std::memset(rrpv, 0, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start cold
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    // neutral SHCT init around threshold
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = SHCT_FRIENDLY_THRES;
        shct_prefetch[i] = SHCT_FRIENDLY_THRES / 2; // more pessimistic for prefetch
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_stream_extra, 0, sizeof(pc_stream_extra));
    std::memset(pc_cold_streak, 0, sizeof(pc_cold_streak));
    std::memset(pc_hot4, 0, sizeof(pc_hot4));
    std::memset(bloom, 0, sizeof(bloom));

    std::memset(bandit_bias, 0, sizeof(bandit_bias));
    std::memset(leader_hits_A, 0, sizeof(leader_hits_A));
    std::memset(leader_hits_B, 0, sizeof(leader_hits_B));
    op_count = 0;
}

// SRRIP victim selection with invalid check
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP: search for maxRRPV; age if none
    for (uint32_t iter = 0; iter <= maxRRPV; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // increment all RRPVs (age), saturating
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    return 0; // fallback (should not happen)
}

static inline void train_hawkeye_on_eviction(uint32_t set, uint32_t way) {
    if (!LEADER_A(set)) return;
    uint32_t slot = LEADER_SLOT(set);
    uint16_t sig = hawk_sig[slot][way];
    uint32_t idx = shct_idx(sig);
    if (hawk_is_pref[slot][way]) {
        if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
        else                      shct_dec(shct_prefetch[idx]);
    } else {
        if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
        else                      shct_dec(shct_demand[idx]);
    }
    // Clear old record; new fill will overwrite
    hawk_used[slot][way] = 0;
    hawk_is_pref[slot][way] = 0;
    hawk_sig[slot][way] = 0;
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    op_count++;

    // Leader hit accounting for selector
    if (hit) {
        if (LEADER_A(set)) {
            leader_hits_A[LEADER_SLOT(set)]++;
        } else if (LEADER_B(set)) {
            leader_hits_B[LEADER_SLOT(set)]++;
        }
    }

    // Bandit epoch update
    if ((op_count % BANDIT_EPOCH) == 0) {
        for (uint32_t slot = 0; slot < 64; slot++) {
            int32_t diff = (int32_t)leader_hits_B[slot] - (int32_t)leader_hits_A[slot];
            if (diff >= 2) {
                if (bandit_bias[slot] < 7) bandit_bias[slot]++;
            } else if (diff <= -2) {
                if (bandit_bias[slot] > -8) bandit_bias[slot]--;
            } else {
                // minor/ambiguous -> decay toward 0 for hysteresis
                if (bandit_bias[slot] > 0) bandit_bias[slot]--;
                else if (bandit_bias[slot] < 0) bandit_bias[slot]++;
            }
            leader_hits_A[slot] = 0;
            leader_hits_B[slot] = 0;
        }
    }

    // On hits: multi-hit gated promotion, hawk reuse mark
    if (hit) {
        if (is_demand(type)) {
            // Mark reuse for Hawkeye leaders
            if (LEADER_A(set)) {
                hawk_used[LEADER_SLOT(set)][way] = 1;
            }

            // Update PC hotness/coldness on demand hits
            uint32_t pidx = pc_index(PC);
            pc_cold_streak[pidx] = 0;
            if (pc_hot4[pidx] < 15) pc_hot4[pidx]++;

            // Multi-hit promotion gating
            if (stream_lock[set][way]) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    rrpv_set(set, way, 0); // MRU promote
                    stream_lock[set][way] = 0;
                } else {
                    // demote on first touch for stream to encourage eviction
                    if (hitcnt[set][way] == 1) rrpv_demote(set, way);
                }
            } else {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU promote
                }
            }
        } else {
            // Non-demand hits: keep state; tiny gentle aging toward MRU
            rrpv_promote(set, way);
        }
        return;
    }

    // Miss path: train on eviction for leader-A before overwriting metadata
    train_hawkeye_on_eviction(set, way);

    // Demand miss: update PC cold streak/hotness and stream detector state
    uint32_t pidx = pc_index(PC);
    if (is_demand(type)) {
        if (pc_cold_streak[pidx] < 3) pc_cold_streak[pidx]++;
        if (pc_cold_streak[pidx] >= 3) bloom_set(PC);
        if (pc_hot4[pidx] > 0) pc_hot4[pidx]--;
    }

    // Decide which mode governs insertion policy for this set
    bool use_modeB = modeB_enabled(set);

    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t lock_stream = 0;

    if (is_writeback(type)) {
        // Never bypass writebacks; insert moderately warm
        ins_rrpv = INSERT_WARM_DEPTH;
        lock_stream = 0;
    } else if (use_modeB || LEADER_B(set)) {
        // Mode B: Stride-run + PC-Bloom
        if (is_prefetch(type)) {
            ins_rrpv = INSERT_TAIL_DEPTH; // quarantine prefetches
            lock_stream = 1;
        } else if (is_demand(type)) {
            uint32_t run_len = detect_run_len_and_update(PC, paddr);
            bool stream_quarantine = false;
            if (run_len >= 2) {
                // deterministic "probabilistic bypass"
                uint8_t thr = (run_len >= 4) ? 100 : (run_len == 3 ? 85 : 60);
                uint8_t hv = hash_pct(PC, line10(paddr));
                stream_quarantine = (hv < thr);
            }
            if (stream_quarantine) {
                ins_rrpv = INSERT_TAIL_DEPTH;
                lock_stream = 1;
            } else {
                bool cold_pc = bloom_test(PC) && (pc_hot4[pidx] < 3);
                ins_rrpv = cold_pc ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
                lock_stream = 0;
            }
        }
    } else {
        // Mode A (Hawkeye-like SHiP)
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (is_prefetch(type)) {
            ins_rrpv = INSERT_TAIL_DEPTH;
            lock_stream = 1;
        } else {
            uint8_t pred_val = shct_demand[idx];
            bool friendly = (pred_val >= SHCT_FRIENDLY_THRES);
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            lock_stream = 0;
        }
        // Record the new line's sig/reuse flags for leader-A sets only
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    }

    // Install new line metadata
    rrpv_set(set, way, ins_rrpv);
    stream_lock[set][way] = lock_stream ? 1 : 0;
    hitcnt[set][way] = 0;

    // Also record leader-A metadata on Mode B/followers (so evict-train happens only for leader-A)
    if (LEADER_A(set) && !is_writeback(type)) {
        uint16_t sig = pc_sig12(PC);
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way] = sig;
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}