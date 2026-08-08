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

// ---------------- Tunables (retuned for R2) ----------------
static constexpr uint8_t  maxRRPV             = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH   = 2;    // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH   = 6;    // near-tail for cold
static constexpr uint8_t  INSERT_TAIL_DEPTH   = 7;    // hard tail (quarantine)
static constexpr uint8_t  HITS_PROMOTE_NS     = 3;    // non-stream MRU on 3rd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR    = 2;    // stream/prefetch escape on 2nd demand hit
static constexpr uint32_t BANDIT_EPOCH        = 1024; // selector epoch (ops)
static constexpr int8_t   MODEB_THRESH        = 2;    // enable Mode B for followers when bias >= +2
static constexpr uint8_t  SHCT_FRIENDLY_THRES = 16;   // Mode A friendly cutoff (0..31)
static constexpr uint8_t  HOT_OVERRIDE_CUTOFF = 3;    // 3+ hits override Bloom coldness

// ---------------- Per-line metadata (packed conceptually) ----------------
// rrpv:3b, hitcnt:2b, stream_lock:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// ---------------- Per-slot selector state (64 slots) ----------------
static int8_t   bandit_bias[64];          // [-7..+7] bias toward Mode B
static uint16_t leader_hits_A[64];        // epoch hit counts for leader A sets (0..1024)
static uint16_t leader_hits_B[64];        // epoch hit counts for leader B sets (0..1024)
static uint32_t epoch_ctr;

// ---------------- Mode A: compact Hawkeye-like SHiP ----------------
#define SHCT_BITS 11
#define SHCT_SIZE (1u << SHCT_BITS) // 2048 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit

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

// ---------------- Mode B: stride-run + lean PC-coldness ----------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B -> 10b

// per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b stored in 16b
static uint8_t  pc_run_len[PC_TBL_SIZE];     // 3b (0..7)
static uint8_t  pc_cold_streak[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_hot3[PC_TBL_SIZE];        // 3b hot override (0..7)

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

static inline uint8_t update_stream_run(uint64_t PC, uint64_t paddr) {
    // Update +1/+2 forward run length (3-bit, saturating), return new run length
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t prev = pc_last_line10[idx];
    uint16_t d = (uint16_t)((ln + 1024u - prev) & 1023u); // forward delta mod 1024
    if (prev == 0 && pc_run_len[idx] == 0) {
        // initial population: treat as reset
        pc_run_len[idx] = 0;
    } else if (d == 1 || d == 2) {
        if (pc_run_len[idx] < 7) pc_run_len[idx]++;
    } else {
        pc_run_len[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return pc_run_len[idx];
}

static inline bool modeB_enabled_for_set(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    // followers: QuickSelect
    uint32_t slot = LEADER_SLOT(set);
    return (bandit_bias[slot] >= MODEB_THRESH);
}

static inline void selector_epoch_tick() {
    epoch_ctr++;
    if (epoch_ctr < BANDIT_EPOCH) return;
    epoch_ctr = 0;
    for (uint32_t s = 0; s < 64; s++) {
        int32_t diff = (int32_t)leader_hits_B[s] - (int32_t)leader_hits_A[s];
        if (diff > 0) {
            if (bandit_bias[s] < 7) bandit_bias[s]++;
        } else if (diff < 0) {
            if (bandit_bias[s] > -7) bandit_bias[s]--;
        } else {
            // decay toward 0 for hysteresis
            if (bandit_bias[s] > 0) bandit_bias[s]--;
            else if (bandit_bias[s] < 0) bandit_bias[s]++;
        }
        leader_hits_A[s] = leader_hits_B[s] = 0;
    }
}

// ---------------- Init ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = INSERT_COLD_DEPTH; // conservative start
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(bandit_bias, 0, sizeof(bandit_bias));
    std::memset(leader_hits_A, 0, sizeof(leader_hits_A));
    std::memset(leader_hits_B, 0, sizeof(leader_hits_B));
    epoch_ctr = 0;

    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        // neutral leaning
        shct_demand[i] = SHCT_FRIENDLY_THRES;
        shct_prefetch[i] = 0;
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_run_len, 0, sizeof(pc_run_len));
    std::memset(pc_cold_streak, 0, sizeof(pc_cold_streak));
    std::memset(pc_hot3, 0, sizeof(pc_hot3));
    std::memset(bloom, 0, sizeof(bloom));
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
    // SRRIP victim: search for maxRRPV, else age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] >= maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
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
    // Selector epoch maintenance
    selector_epoch_tick();

    // Update per-PC stream run on all non-writeback references
    uint8_t run_len = 0;
    if (!is_writeback(type)) {
        run_len = update_stream_run(PC, paddr);
    }

    // Leader hit accounting (all hit types)
    if (hit) {
        if (LEADER_A(set)) leader_hits_A[LEADER_SLOT(set)]++;
        if (LEADER_B(set)) leader_hits_B[LEADER_SLOT(set)]++;
    }

    // Hawkeye training for leader-A on evictions (performed on fill path)
    if (!hit && LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        // Train old resident at [way]
        if (hawk_sig[slot][way] != 0) {
            uint32_t idx = shct_idx(hawk_sig[slot][way]);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
                else                      shct_dec(shct_prefetch[idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
                else                      shct_dec(shct_demand[idx]);
            }
        }
        // Prepare new record (will be overwritten below after insertion policy is chosen)
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    }

    // On hit: apply multi-hit gated promotions
    if (hit) {
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (stream_lock[set][way]) {
                if (hitcnt[set][way] == 1) {
                    // first demand hit on a stream line: mild demote
                    rrpv_demote(set, way);
                } else if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    rrpv_set(set, way, 0); // MRU
                    stream_lock[set][way] = 0; // escape stream quarantine
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU
                }
            }
        }
        // Mark reuse for Hawkeye training in leader-A
        if (LEADER_A(set)) {
            hawk_used[LEADER_SLOT(set)][way] = 1;
        }
        // PC hotness/coldness tracking
        if (!is_writeback(type)) {
            uint32_t pidx = pc_index(PC);
            if (pc_hot3[pidx] < 7) pc_hot3[pidx]++;
            pc_cold_streak[pidx] = 0;
        }
        return;
    }

    // Miss / Fill path: choose insertion depth and guards
    uint8_t ins = INSERT_WARM_DEPTH;
    uint8_t lock = 0; // stream lock

    // PC coldness tracking on miss
    if (!is_writeback(type)) {
        uint32_t pidx = pc_index(PC);
        if (pc_cold_streak[pidx] < 3) pc_cold_streak[pidx]++;
        if (pc_hot3[pidx] > 0) pc_hot3[pidx]--; // slight decay on miss
        if (pc_cold_streak[pidx] >= 2) {
            bloom_set(PC);
            pc_cold_streak[pidx] = 2; // saturate
        }
    }

    // Prefetch quarantine and Writeback policy
    if (is_prefetch(type)) {
        ins = INSERT_TAIL_DEPTH;
        lock = 1;
    } else if (is_writeback(type)) {
        ins = INSERT_WARM_DEPTH;
        lock = 0;
    } else {
        // Demand miss
        bool use_modeB = modeB_enabled_for_set(set);

        // HardRun stream guard: after 2+ forward steps, hard-tail insert and stream-lock
        if (run_len >= 2) {
            ins = INSERT_TAIL_DEPTH;
            lock = 1;
        } else {
            // Lean PC coldness
            uint32_t pidx = pc_index(PC);
            bool pc_cold = bloom_test(PC) && (pc_hot3[pidx] < HOT_OVERRIDE_CUTOFF);

            if (!use_modeB || LEADER_A(set)) {
                // Mode A: Hawkeye-like insertion (followers default here)
                uint16_t sig = pc_sig12(PC);
                uint32_t idx = shct_idx(sig);
                uint8_t score = shct_demand[idx];
                bool friendly = (score >= SHCT_FRIENDLY_THRES);
                ins = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            } else {
                // Mode B: default warm, cold PCs near-tail
                ins = pc_cold ? INSERT_COLD_DEPTH : INSERT_WARM_DEPTH;
            }
        }
    }

    // Apply insertion
    rrpv_set(set, way, ins);
    stream_lock[set][way] = lock;
    hitcnt[set][way] = 0;

    // Ensure leader-A fills record current PC (already set above on fill)
    if (!LEADER_A(set)) {
        // nothing else to do
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}