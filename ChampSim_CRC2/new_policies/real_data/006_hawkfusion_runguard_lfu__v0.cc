#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t) { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }

// ---------------- Leader-set sampling (64 total) -----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables ----------------
static constexpr uint8_t maxRRPV             = 7;   // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;   // near-MRU for hot PCs
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;   // near-tail for cold PCs
static constexpr uint8_t STREAM_CONF_THRESH  = 2;   // +1/+2 forward steps (2-step)
static constexpr uint8_t HITS_PROMOTE        = 2;   // demand MRU at 2nd hit
static constexpr uint8_t MODEB_ENABLE_THRESH = 2;   // per-set bandit threshold
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;   // demote quarantined on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;   // TinyLFU hot threshold (0..31)
static constexpr uint32_t DECAY_PERIOD       = 8192; // global access period to decay bandit/lfu

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hit_once:1b, stream_flag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hit_once[LLC_SETS][LLC_WAYS];
static uint8_t stream_flag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector ------------------------
static int8_t bandit_score[LLC_SETS]; // followers enable B only if >= threshold

// ---------------- Mode A (Hawkeye, compact) ----------------------
#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* demand_predictor;
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;

#define OPTGEN_VECTOR_SIZE 128
#include "optgen.h"
static OPTgen perset_optgen[LLC_SETS];

// Per-set timers (used for sampled sets)
static uint64_t perset_mytimer[LLC_SETS];

// Signatures + prefetched flag only trained on sampled sets (12b effective sig)
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS];
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS];

// Sampler: reduced size to fit budget (1K entries)
#define SAMPLED_CACHE_SIZE 1024
#define SAMPLER_WAYS 8
#define SAMPLER_SETS (SAMPLED_CACHE_SIZE / SAMPLER_WAYS)
static std::vector<std::map<uint64_t, ADDR_INFO>> addr_history;

// ---------------- Mode B (RunGuard-LFU) --------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); }

// Per-PC state: last line (10b), stream conf (2b), TinyLFU (5b)
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE];
static uint8_t  pc_use5[PC_TBL_SIZE];

// ---------------- Helpers -----------------
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u5(uint8_t& x) { if (x < 31) x++; }
static inline void rrip_decrement(uint8_t& v) { if (v > 0) v--; }
static inline void rrip_increment(uint8_t& v) { if (v < maxRRPV) v++; }

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (fwd) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Global epoch for decay -------------------------
static uint64_t global_access_ctr = 0;

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        perset_mytimer[s] = 0;
        perset_optgen[s].init(LLC_WAYS - 2);
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hit_once[s][w] = 0;
            stream_flag[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    addr_history.clear();
    addr_history.resize(SAMPLER_SETS);
    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use5, 0, sizeof(pc_use5));
    global_access_ctr = 0;
}

// ---------------- Victim selection (RRIP + bandit penalty hook) --
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // 1) Prefer invalid immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // 2) If any at maxRRPV, choose first such way
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }

    // 3) Otherwise pick LRU among highest-RRPV and train Hawkeye negative
    uint32_t max_val = 0;
    int32_t lru_victim = -1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= max_val) {
            max_val = rrpv[set][w];
            lru_victim = (int32_t)w;
        }
    }

    if (lru_victim < 0) lru_victim = 0;

    // Bandit penalty: on leader sets, penalize the active mode if line didn't reach 2nd hit
    if (SEL_SAMPLED(set)) {
        // Approximate: penalize if it saw exactly one demand hit (hit_once==1).
        if (hit_once[set][(uint32_t)lru_victim] == 1) {
            if (LEADER_B(set)) sat_dec_i8(bandit_score[set]);
            if (LEADER_A(set)) sat_inc_i8(bandit_score[set]);
        }
    }

    // Hawkeye negative training on LRU evictions from sampled sets
    if (SEL_SAMPLED(set)) {
        uint16_t sig = hawk_signatures[set][(uint32_t)lru_victim] & ((1u << 12) - 1);
        if (hawk_prefetched[set][(uint32_t)lru_victim])
            prefetch_predictor->decrement(sig);
        else
            demand_predictor->decrement(sig);
    }

    return (uint32_t)lru_victim;
}

// ---------------- Hawkeye update (compact) -----------------------
static inline void hawk_update(uint32_t set, uint64_t paddr, uint64_t PC, uint32_t type) {
    if (type == ACCESS_WRITEBACK) return;

    if (SEL_SAMPLED(set)) {
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;
        uint32_t sampler_set = (uint32_t)((paddr >> 6) % SAMPLER_SETS);
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        auto& ah = addr_history[sampler_set];
        auto it = ah.find(sampler_tag);

        if (it != ah.end() && (type != ACCESS_PREFETCH)) {
            unsigned int curr_timer = (unsigned int)perset_mytimer[set];
            if (curr_timer < it->second.last_quanta)
                curr_timer += 1024;
            bool wrap = ((curr_timer - it->second.last_quanta) > OPTGEN_VECTOR_SIZE);
            uint64_t last_quanta = it->second.last_quanta % OPTGEN_VECTOR_SIZE;

            bool cache_now = (!wrap && perset_optgen[set].should_cache(curr_quanta, last_quanta));
            if (it->second.prefetched)
                (cache_now ? prefetch_predictor->increment(it->second.PC)
                           : prefetch_predictor->decrement(it->second.PC));
            else
                (cache_now ? demand_predictor->increment(it->second.PC)
                           : demand_predictor->decrement(it->second.PC));

            perset_optgen[set].add_access(curr_quanta);
            // Update sampler LRU
            for (auto& kv : ah) {
                if (kv.second.lru < it->second.lru)
                    kv.second.lru++;
            }
            it->second.prefetched = false;
            it->second.PC = PC & ((1u << 12) - 1);
            it->second.last_quanta = (unsigned int)perset_mytimer[set];
            it->second.lru = 0;
        } else if (it == ah.end()) {
            if (ah.size() == SAMPLER_WAYS) {
                // Evict LRU entry
                uint64_t lru_tag = 0;
                uint32_t lru_pos = 0;
                for (auto& kv : ah) {
                    if (kv.second.lru == (SAMPLER_WAYS - 1)) { lru_tag = kv.first; lru_pos = kv.second.lru; break; }
                }
                ah.erase(lru_tag);
            }
            ADDR_INFO info;
            info.PC = PC & ((1u << 12) - 1);
            info.last_quanta = (unsigned int)perset_mytimer[set];
            info.lru = 0;
            info.prefetched = (type == ACCESS_PREFETCH);
            // Age others
            for (auto& kv : ah) {
                if (kv.second.lru < (SAMPLER_WAYS - 1)) kv.second.lru++;
            }
            ah[sampler_tag] = info;
            perset_optgen[set].add_access(curr_quanta);
        }
    }
}

// ---------------- Replacement state update -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    // Global decays (lightweight, amortized)
    global_access_ctr++;
    if ((global_access_ctr % DECAY_PERIOD) == 0) {
        // Decay a single bandit entry toward 0
        uint32_t s = (uint32_t)((global_access_ctr / DECAY_PERIOD) % LLC_SETS);
        if (bandit_score[s] > 0) bandit_score[s]--;
        else if (bandit_score[s] < 0) bandit_score[s]++;

        // LFU decay one PC entry
        uint32_t pi = (uint32_t)((global_access_ctr / DECAY_PERIOD) & (PC_TBL_SIZE - 1));
        sat_dec_u5(pc_use5[pi]);
    }

    // Align to line
    uint64_t line_addr = (paddr >> 6) << 6;

    // Train Hawkeye sampler (safe for all types except WB)
    hawk_update(set, line_addr, PC, type);

    // Prefetch mark for sampled sets (for future negative training on eviction)
    if (type == ACCESS_PREFETCH && !hit) hawk_prefetched[set][way] = 1;
    else if (type != ACCESS_PREFETCH)    hawk_prefetched[set][way] = 0;

    // Ignore writebacks for policy actions (no bypass on WB)
    if (type == ACCESS_WRITEBACK) return;

    // Per-PC stats
    if (is_demand(type)) sat_inc_u5(pc_use5[pc_index(PC)]);

    // On hit: promotion gating and stream quarantine handling
    if (hit) {
        if (is_demand(type)) {
            // If quarantined stream, demote on first touch unless it escapes on 2nd hit
            if (stream_flag[set][way] && STREAM_DEMOTE_TOUCH) rrip_increment(rrpv[set][way]);

            if (hit_once[set][way] == 0) {
                // First demand hit: soften position but do not MRU
                rrip_decrement(rrpv[set][way]);
                hit_once[set][way] = 1;
            } else {
                // Second demand hit: promote to MRU, escape quarantine if any
                rrpv[set][way] = 0;
                stream_flag[set][way] = 0;

                // Bandit reward on leader sets (attribute success)
                if (LEADER_B(set)) sat_inc_i8(bandit_score[set]);
                if (LEADER_A(set)) sat_dec_i8(bandit_score[set]);

                // Keep hit_once at 1 for continued modest protection
                // (alternatively, could clear to re-arm; leave as 1 to reduce thrash)
            }
        } else {
            // Non-demand hit: gently age toward MRU but never full promotion
            rrip_decrement(rrpv[set][way]);
        }
        return;
    }

    // Miss/Fill path: choose mode, decide insertion depth and marks
    bool useB = modeB_enabled(set);

    // Always store a 12b signature for Hawkeye training on (future) eviction
    hawk_signatures[set][way] = (uint16_t)(PC & ((1u << 12) - 1));

    uint8_t ins_depth = INSERT_COLD_DEPTH;
    bool mark_stream = false;

    if (type == ACCESS_PREFETCH) {
        // Quarantine all prefetches at tail
        ins_depth = maxRRPV;
        mark_stream = true;
    } else if (!useB) {
        // Mode A: Hawkeye-guided insertion
        uint32_t sig = (uint32_t)(PC & (SHCT_SIZE - 1));
        // Friendly vs averse based on predictors
        bool friendly = (demand_predictor->get_prediction(sig) > (MAX_SHCT / 2));
        ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        mark_stream = false;
    } else {
        // Mode B: RunGuard-LFU with streaming bypass intent (tail insertion)
        bool is_stream = detect_and_update_stream(PC, line_addr);
        mark_stream = is_stream;
        uint8_t usecnt = pc_use5[pc_index(PC)];
        bool hot_pc = (usecnt >= PC_USE_HOT_THRESH);
        if (is_stream) {
            ins_depth = maxRRPV;
        } else {
            ins_depth = hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    // Apply insertion
    rrpv[set][way] = (ins_depth > maxRRPV) ? maxRRPV : ins_depth;
    stream_flag[set][way] = mark_stream ? 1 : 0;
    hit_once[set][way] = 0;
}

// ---------------- Stats hooks (silent) ---------------------------
void PrintStats_Heartbeat() {}
void PrintStats() {}