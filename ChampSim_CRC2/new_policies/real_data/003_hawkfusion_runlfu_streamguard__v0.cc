#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Leader-set sampling (64 leaders) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // Sample 64 sets: low 6 bits == next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// ---------------- Tunables (balanced for lbm/mcf/gcc/omnetpp/astar) ----
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // +1/+2 forward steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined stream on touch
static constexpr uint8_t PC_USE_HOT_THRESH   = 7;  // TinyLFU hot threshold (0..15)
static constexpr int8_t  MODEB_ENABLE_THRESH = 2;  // followers enable B at >=2

// ---------------- Per-line packed metadata ------------------------
// Conceptually: rrpv:3b, hitcnt:2b (0..3), stream_tag:1b, from_modeB:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];
static uint8_t from_modeB[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector ------------------------
static int8_t bandit_score[LLC_SETS]; // followers use Mode B only if >= MODEB_ENABLE_THRESH

// ---------------- Mode A (Hawkeye-like insertion) -----------------
// Tiny SHCTs (demand & prefetch), trained on leader-A sets only
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Store signature (lower 12 bits of PC) + prefetched flag per line for training (leader-A sets only)
static uint16_t hawk_signatures[LLC_SETS][LLC_WAYS]; // 12b effective in 16b storage
static uint8_t  hawk_prefetched[LLC_SETS][LLC_WAYS]; // 0/1

// ---------------- Mode B (RunGuard + TinyLFU) ---------------------
// 512-entry PC tables
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b value in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b (0..15)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_tag[s][w] = 0;
            from_modeB[s][w] = 0;
            hawk_signatures[s][w] = 0;
            hawk_prefetched[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
}

static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes (guarantee termination)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: return the oldest (max rrpv) way
    uint32_t victim = 0, best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// ---------------- Victim selection -------------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    bool demand = is_demand(type);

    // Demand hit handling: multi-hit gating + TinyLFU learn + stream demote-on-touch
    if (hit) {
        if (demand) {
            // Stream demote-on-touch before promotion (helps scan resistance)
            if (stream_tag[set][way] && hitcnt[set][way] < HITS_PROMOTE_STR) {
                for (uint8_t k = 0; k < STREAM_DEMOTE_TOUCH; k++) {
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                }
            }
            // Track demand hits (cap at 3)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            // TinyLFU: reinforce hot PCs on successful demand hits
            sat_inc_u4(pc_use4[pc_index(PC)]);

            // Promotion only on 2nd demand hit
            uint8_t need = stream_tag[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                rrpv[set][way] = 0; // MRU
                stream_tag[set][way] = 0; // escape quarantine after sufficient reuse
            }
        }
        return;
    }

    // MISS / FILL path
    // 1) Bandit learning on the evicted line from this way (based on its observed reuse)
    {
        uint8_t old_hits = hitcnt[set][way];
        uint8_t old_fromB = from_modeB[set][way];
        // Reward the mode that inserted lines with >=2 demand hits, penalize otherwise
        if (old_fromB) {
            if (old_hits >= 2) sat_inc_i8(bandit_score[set]);
            else               sat_dec_i8(bandit_score[set]);
        } else {
            if (old_hits >= 2) sat_dec_i8(bandit_score[set]);
            else               sat_inc_i8(bandit_score[set]);
        }
        // Mode A training on leader-A sets (use 2+ hits as "friendly")
        if (LEADER_A(set)) {
            uint32_t idx = shct_idx(hawk_signatures[set][way]);
            if (hawk_prefetched[set][way]) {
                if (old_hits >= 2) shct_inc(shct_prefetch[idx]);
                else               shct_dec(shct_prefetch[idx]);
            } else {
                if (old_hits >= 2) shct_inc(shct_demand[idx]);
                else               shct_dec(shct_demand[idx]);
            }
        }
    }

    // 2) Determine active mode for this fill
    bool use_modeB = modeB_enabled(set);

    // 3) Decide insertion policy
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;

    // Prefetches: always quarantine at tail
    if (type == ACCESS_PREFETCH) {
        ins_rrpv = maxRRPV;
        new_stream_tag = 1;
        // Slightly discount PC on prefetch-triggered miss (avoids over-crediting)
        sat_dec_u4(pc_use4[pc_index(PC)]);
    } else if (type == ACCESS_WRITEBACK) {
        // Never bypass writebacks; give them reasonable priority
        ins_rrpv = INSERT_WARM_DEPTH;
    } else {
        // Demand miss
        bool is_stream = detect_and_update_stream(PC, paddr);
        if (use_modeB) {
            // Mode B: RunGuard + TinyLFU
            if (is_stream) {
                ins_rrpv = maxRRPV; // hard-tail quarantine
                new_stream_tag = 1;
            } else {
                uint8_t use = pc_use4[pc_index(PC)];
                ins_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            }
        } else {
            // Mode A: Hawkeye-like (PC-friendly/averse via SHCT trained on leader-A sets)
            uint32_t idx = shct_idx(PC);
            uint8_t conf = (type == ACCESS_PREFETCH) ? shct_prefetch[idx] : shct_demand[idx];
            ins_rrpv = (conf >= (SHCT_MAX / 2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            // Basic stream awareness (safe): do not override Mode A, but update detector state
            (void)is_stream; // already updated by detect during demand misses above
        }

        // TinyLFU: on demand miss, decays cold PCs a bit
        sat_dec_u4(pc_use4[pc_index(PC)]);
    }

    // 4) Install new line metadata
    rrpv[set][way]       = ins_rrpv;
    hitcnt[set][way]     = 0;
    stream_tag[set][way] = new_stream_tag;
    from_modeB[set][way] = use_modeB ? 1 : 0;

    // Record Mode A signature for leader-A sets
    if (LEADER_A(set)) {
        hawk_signatures[set][way] = (uint16_t)(PC & 0x0FFFu); // 12-bit
        hawk_prefetched[set][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}