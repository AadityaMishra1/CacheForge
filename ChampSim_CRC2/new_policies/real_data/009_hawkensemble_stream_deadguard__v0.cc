#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t == ACCESS_PREFETCH); }

// ------------------------ Tunables ------------------------
// 3-bit RRIP
static constexpr uint8_t maxRRPV             = 7;
// Mode A (Hawkeye-like) insertion depths
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
// Mode B (RunShield)
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // need 2 forward steps (+1/+2)
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // quarantine tail
static constexpr uint8_t STREAM_TAIL_SHALLOW = 5;  // shallower tail for very hot PCs
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote on first demand touch
// Multi-hit gating
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream MRU at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream escape at 2nd demand hit
// PC usefulness/coldness thresholds
static constexpr uint8_t PC_USE_HOT_THRESH   = 6;  // TinyLFU >=6 considered hot (0..15)
static constexpr uint8_t PC_COLD_WEAKEN      = 1;  // if coldness >0, bias colder insertion
// Selector
static constexpr uint32_t BANDIT_EPOCH       = 4096; // update global/local gates
static constexpr uint8_t  GSEL_MAX           = 31;   // global gate range
static constexpr uint8_t  GSEL_THRES         = 16;   // followers require >= to allow Mode B
static constexpr int8_t   MODEB_ENABLE_THRES = 1;    // per-set bandit threshold

// ------------------------ Set sampling ------------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set){
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ------------------------ Per-line state (conceptual bit-pack) ------------------------
// rrpv:3b, hitcnt:2b, runlock(stream quarantine):1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t runlock[LLC_SETS][LLC_WAYS]; // 1 if quarantined stream/pref line

// ------------------------ Selector state ------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], but used as small bias per set
static uint8_t GSEL = 0;               // global enable gate
// Per-set stream reuse stats (6-bit saturating conceptual)
static uint8_t stream_obs[LLC_SETS];   // number of stream quarantines observed
static uint8_t stream_reuse[LLC_SETS]; // number of quarantined lines that later hit (escape)

// ------------------------ Mode A (Hawkeye-like) ------------------------
// 1K-entry, 5-bit SHCT (demand only)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x){ if (x > 0) x--; }

// Per-line training buffers for 64 leader sets only
static uint16_t hawk_sig[64][LLC_WAYS];   // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];  // 1b reuse marker
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1b (kept for clarity; demand-only SHCT used)

// ------------------------ Mode B (RunShield + TinyLFU + Coldness) ------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // last line index (10b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

static inline void sat_inc_u2(uint8_t &x){ if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_i4(int8_t &x){ if (x < 7) x++; }
static inline void sat_dec_i4(int8_t &x){ if (x > -8) x--; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr){
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

// ------------------------ Housekeeping ------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void periodic_housekeeping(){
    op_count++;
    if ((op_count % BANDIT_EPOCH) == 0){
        // Global gate via leader hits
        if (leaderB_hits_epoch > leaderA_hits_epoch) { if (GSEL < GSEL_MAX) GSEL++; }
        else if (leaderA_hits_epoch > leaderB_hits_epoch) { if (GSEL > 0) GSEL--; }
        leaderA_hits_epoch = 0;
        leaderB_hits_epoch = 0;

        // Per-set bandit from stream reuse ratio
        for (uint32_t s = 0; s < LLC_SETS; s++){
            // Use 6-bit conceptual saturation; treat >=64 as clamped by code
            uint32_t obs = stream_obs[s];
            uint32_t res = stream_reuse[s];
            if (obs >= 2){ // enough evidence
                if ((res << 1) >= obs) sat_inc_i4(bandit_score[s]); // good reuse under Mode B
                else                   sat_dec_i4(bandit_score[s]); // poor reuse => prefer Mode A
            }
            // Decay stats
            stream_obs[s]   = (uint8_t)(obs >> 1);
            stream_reuse[s] = (uint8_t)(res >> 1);
        }

        // Decay TinyLFU and coldness lazily
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++){
            if (pc_use4[i])  pc_use4[i] >>= 1;
            if (pc_cold2[i]) pc_cold2[i] >>= 1;
        }
    }
}

static inline bool modeB_enabled(uint32_t set){
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRES);
}

// ------------------------ RRIP Victim ------------------------
static inline uint32_t rrip_victim(uint32_t set, const BLOCK* current_set){
    // 1) return invalid immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (!current_set[w].valid) return w;
    }
    // 2) find any line at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) bounded aging to ensure progress
    for (int pass = 0; pass < 8; pass++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback
    return 0;
}

// ------------------------ ChampSim API ------------------------
void InitReplacementState(){
    for (uint32_t s = 0; s < LLC_SETS; s++){
        bandit_score[s] = 0;
        stream_obs[s] = 0;
        stream_reuse[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            runlock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    op_count = 0;
    leaderA_hits_epoch = 0;
    leaderB_hits_epoch = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    periodic_housekeeping();

    // Writebacks are installed but not trained/promoted; keep at tail
    if (type == ACCESS_WRITEBACK){
        if (!hit){
            rrpv[set][way] = maxRRPV;
            hitcnt[set][way] = 0;
            runlock[set][way] = 0; // no runlock on WB lines
        }
        return;
    }

    uint32_t pc_idx = pc_index(PC);
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_idx]);  // online usefulness
        if (!hit) sat_inc_u2(pc_cold2[pc_idx]); // misses increase coldness
    }

    bool modeB = modeB_enabled(set);

    // ---------------- Hit path ----------------
    if (hit){
        // Leader hit accounting for selector
        if (is_demand(type)){
            if (LEADER_A(set)) leaderA_hits_epoch++;
            if (LEADER_B(set)) leaderB_hits_epoch++;
        }

        // Count reuse of quarantined stream lines to guide per-set bandit
        if (runlock[set][way] && is_demand(type)){
            // First touch: demote (if configured), count toward reuse when it hits
            if (STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            // Count as a reuse evidence
            if (stream_reuse[set] < 63) stream_reuse[set]++;
        }

        // Multi-hit gating: promote only on 2nd demand hit
        if (is_demand(type)){
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            bool is_stream_locked = (runlock[set][way] != 0);
            uint8_t need = is_stream_locked ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need){
                rrpv[set][way] = 0;      // MRU
                runlock[set][way] = 0;   // escape quarantine if any
            } else {
                // gentle nudge toward MRU
                if (rrpv[set][way] > 0) rrpv[set][way]--;
            }
            // Demand hit indicates usefulness: reduce PC coldness
            sat_dec_u2(pc_cold2[pc_idx]);
        } else {
            // Prefetch hit: do not aggressively promote
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
        return;
    }

    // ---------------- Miss/Fill path ----------------
    // Train Mode A SHCT on leader-A evictions (use prior occupant's metadata before overwrite)
    if (LEADER_A(set)){
        uint32_t slot = LEADER_SLOT(set);
        // The victim metadata in leader arrays corresponds to this 'way'
        uint16_t esig = hawk_sig[slot][way];
        uint8_t  eused = hawk_used[slot][way];
        // Train demand-only SHCT (prefetch kept for clarity but not used in this design)
        uint32_t idx = shct_idx(esig);
        if (eused) shct_inc(shct_demand[idx]);
        else       shct_dec(shct_demand[idx]);
    }

    // Choose mode for this fill
    if (!modeB){
        // ---------------- Mode A (Hawkeye-like) insertion ----------------
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t depth = (shct_demand[idx] > (SHCT_MAX/2)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

        if (is_prefetch(type)){
            rrpv[set][way] = maxRRPV; // prefetch quarantine at tail
            runlock[set][way] = 1;
            hitcnt[set][way] = 0;
        } else {
            rrpv[set][way] = depth;
            runlock[set][way] = 0;
            hitcnt[set][way] = 0;
        }

        // Record new signature for leader-A training
        if (LEADER_A(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    } else {
        // ---------------- Mode B (RunShield + TinyLFU + Coldness) ----------------
        bool is_stream = detect_and_update_stream(PC, paddr);
        uint8_t use = pc_use4[pc_idx];
        uint8_t cold = pc_cold2[pc_idx];

        if (is_prefetch(type)){
            // Strict quarantine for prefetches
            rrpv[set][way] = STREAM_TAIL_DEPTH;
            runlock[set][way] = 1;
            hitcnt[set][way] = 0;
        } else {
            // Demand fill
            if (is_stream){
                // count stream observation for bandit
                if (stream_obs[set] < 63) stream_obs[set]++;
                // quarantine streams to tail; allow short reuse by shallower tail for very hot PCs
                uint8_t depth = (use >= PC_USE_HOT_THRESH) ? STREAM_TAIL_SHALLOW : STREAM_TAIL_DEPTH;
                rrpv[set][way] = depth;
                runlock[set][way] = 1;   // RunLock: forbid first-hit MRU
                hitcnt[set][way] = 0;
            } else {
                // Non-stream: adaptive insertion using TinyLFU and coldness
                uint8_t depth = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                if (cold >= PC_COLD_WEAKEN && depth < maxRRPV){ // weaken if cold
                    depth = (uint8_t)std::min<uint8_t>(maxRRPV, (uint8_t)(depth + 1));
                }
                rrpv[set][way] = depth;
                runlock[set][way] = 0;
                hitcnt[set][way] = 0;
            }
        }

        // Leader-B bookkeeping (for consistency, we still set leader-A arrays when A; B does not need per-line sig)
        if (LEADER_B(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        }
    }
}

// Called on every access; mark reuse for leader-A lines on hits
// This overload is merged into UpdateReplacementState above by setting hawk_used on hits.
// Ensure hawk_used is set here for leader-A hits.
void PrintStats() {}
void PrintStats_Heartbeat() {}