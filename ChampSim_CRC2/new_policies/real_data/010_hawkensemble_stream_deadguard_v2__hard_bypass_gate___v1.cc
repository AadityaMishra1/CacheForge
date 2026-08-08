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
// RRIP
static constexpr uint8_t maxRRPV               = 7;
// Mode A (Hawkeye-like SHiP-ish)
static constexpr uint8_t INSERT_WARM_DEPTH     = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH     = 6;  // near-tail
static constexpr uint8_t SHCT_FRIENDLY_THRES   = 16; // 5b counter threshold
// Mode B (RunShield v2)
static constexpr uint8_t STREAM_CONF_THRESH    = 2;  // two forward steps (+1/+2)
static constexpr uint8_t STREAM_TAIL_DEPTH     = 7;  // deep quarantine
static constexpr uint8_t STREAM_TAIL_SHALLOW   = 5;  // shallow quarantine for hot-PC
static constexpr uint8_t STREAM_DEMOTE_ON_TOUCH= 1;  // demote quarantined line on first touch
// Multi-hit gating
static constexpr uint8_t HITS_PROMOTE_NS       = 2;  // non-stream -> MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 2;  // stream escape on 2nd demand hit
// PC TinyLFU + coldness
static constexpr uint8_t PC_USE_HOT_THRESH     = 6;  // 0..15 hot threshold
static constexpr uint8_t PC_COLD_WEAKEN        = 1;  // deepen insertion for cold PCs
// Selector
static constexpr uint32_t BANDIT_EPOCH         = 4096; // ops per epoch
static constexpr uint8_t  GSEL_MAX             = 31;
static constexpr uint8_t  GSEL_THRES           = 16;   // need >= to allow Mode B
static constexpr int8_t   MODEB_ENABLE_THRES   = 1;    // per-set bandit threshold
// Local stream reuse gate (reuse*3 >= obs) or low obs
static constexpr uint8_t  SR_LOCAL_MIN_OBS     = 8;

// ------------------------ Helpers ------------------------
static inline void sat_inc_u3(uint8_t &x){ if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u2(uint8_t &x){ if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_i4(int8_t &x){ if (x < 7) x++; }
static inline void sat_dec_i4(int8_t &x){ if (x > -8) x--; }

// ------------------------ Set sampling ------------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set){
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u)); // 64 leaders
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ------------------------ Per-line packed state (conceptual) ------------------------
// rrpv:3b, hitcnt:2b, runlock:1b (quarantine)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t runlock[LLC_SETS][LLC_WAYS]; // 1 if stream/pref quarantine

// ------------------------ Selector state ------------------------
static int8_t  bandit_score[LLC_SETS]; // small per-set bias
static uint8_t GSEL = 0;               // global enable gate
// Per-set stream reuse stats (decayed each epoch)
static uint8_t stream_obs[LLC_SETS];   // observed quarantines
static uint8_t stream_reuse[LLC_SETS]; // quarantined lines that later hit

// ------------------------ Mode A (Hawkeye-like SHiP) ------------------------
// 1K-entry, 5-bit SHCT (demand only)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx_from_pc(uint64_t pc){
    uint16_t sig = pc_sig12(pc);
    return (uint32_t)(sig & (SHCT_SIZE - 1u)); // 10-bit index
}
static inline void shct_inc(uint8_t &x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x){ if (x > 0) x--; }

// Leader training buffers for 64 leader sets only
static uint16_t hawk_sig[64][LLC_WAYS];   // store 10-bit index; 12b slot
static uint8_t  hawk_used[64][LLC_WAYS];  // reuse marker (demand hit)
static uint8_t  hawk_valid[64][LLC_WAYS]; // valid entry for training

// ------------------------ Mode B (RunShield + TinyLFU + Coldness) ------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b last line
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

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
        leaderA_hits_epoch = leaderB_hits_epoch = 0;

        // Light decay of per-set stream stats and bandit scores
        for (uint32_t s = 0; s < LLC_SETS; s++){
            stream_obs[s]   >>= 1;
            stream_reuse[s] >>= 1;
            // drift bandit toward 0
            if (bandit_score[s] > 0) bandit_score[s]--;
            else if (bandit_score[s] < 0) bandit_score[s]++;
        }
    }
}

// ------------------------ Init ------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        stream_obs[s] = stream_reuse[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            runlock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    for (uint32_t i = 0; i < SHCT_SIZE; i++) shct_demand[i] = 1; // mild friendly prior

    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_valid, 0, sizeof(hawk_valid));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    GSEL = 0;
    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
}

// ------------------------ Victim selection ------------------------
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
    // SRRIP victim selection with bounded aging
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Increment all RRPVs that are not at max
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    return 0; // fallback
}

// ------------------------ Update state ------------------------
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
    periodic_housekeeping();

    // Leader membership
    bool lA = LEADER_A(set);
    bool lB = LEADER_B(set);
    uint32_t lslot = LEADER_SLOT(set);

    // Mode B gate (followers require both global and local)
    bool local_ok = (bandit_score[set] >= MODEB_ENABLE_THRES);
    bool global_ok = (GSEL >= GSEL_THRES);
    bool reuse_ok = (stream_obs[set] < SR_LOCAL_MIN_OBS) ||
                    (uint32_t(stream_reuse[set]) * 3u >= uint32_t(stream_obs[set]));
    bool allow_modeB = local_ok && global_ok && reuse_ok;
    bool prefer_modeB = lB || (!lA && allow_modeB);

    uint32_t shct_i = shct_idx_from_pc(PC);
    uint32_t pc_i   = pc_index(PC);

    if (hit) {
        // Train selector with leader hits
        if (lA){ leaderA_hits_epoch++; sat_dec_i4(bandit_score[set]); }
        if (lB){ leaderB_hits_epoch++; sat_inc_i4(bandit_score[set]); }

        // Update TinyLFU and coldness on demand hits
        if (is_demand(type)) {
            sat_inc_u4(pc_use4[pc_i]);
            sat_dec_u2(pc_cold2[pc_i]);
        }

        // Mark reuse for Hawkeye leaders (demand hits only)
        if (lA && is_demand(type)) {
            hawk_used[lslot][way] = 1;
        }

        // Promotion and quarantine handling
        if (runlock[set][way]) {
            // quarantined (stream/pref)
            if (is_demand(type)) {
                if (STREAM_DEMOTE_ON_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    runlock[set][way] = 0;        // escape quarantine
                    hitcnt[set][way] = 0;
                    rrpv[set][way] = 0;           // MRU
                    // count reuse of quarantined lines
                    if (stream_reuse[set] < 63) stream_reuse[set]++;
                }
            }
            // Prefetch-only hits are not specially promoted here (handled via demand path)
        } else {
            // non-quarantined: multi-hit gated promotion
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    hitcnt[set][way] = 0;
                    rrpv[set][way] = 0; // MRU
                }
            }
        }
        return;
    }

    // Miss: choose insertion policy and train as needed
    bool stream_detected = false;
    if (is_demand(type) || is_prefetch(type)) {
        stream_detected = detect_and_update_stream(PC, paddr);
    }

    // Leader A training on eviction (only for demand path)
    if (lA && type != ACCESS_WRITEBACK) {
        if (hawk_valid[lslot][way]) {
            uint32_t old_idx = hawk_sig[lslot][way] & (SHCT_SIZE - 1u);
            if (hawk_used[lslot][way]) shct_inc(shct_demand[old_idx]);
            else                       shct_dec(shct_demand[old_idx]);
        }
        hawk_sig[lslot][way]   = (uint16_t)shct_i;
        hawk_used[lslot][way]  = 0;
        hawk_valid[lslot][way] = 1;
    }

    // Decide insertion depth and quarantine flag
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t quarantine = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback: insert at moderate tail
        ins_rrpv = 6;
        quarantine = 0;
    } else if (prefer_modeB) {
        // Mode B path
        bool hot_pc = (pc_use4[pc_i] >= PC_USE_HOT_THRESH);
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            quarantine = 1;
        } else if (stream_detected) {
            // Hard-tail quarantine for streams; hot-PC gets shallow-tail half the time (PC parity)
            if (hot_pc && (PC & 1ull)) ins_rrpv = STREAM_TAIL_SHALLOW;
            else                       ins_rrpv = STREAM_TAIL_DEPTH;
            quarantine = 1;
        } else {
            // Non-stream in Mode B: adaptive insertion with PC biases
            ins_rrpv = hot_pc ? INSERT_WARM_DEPTH : 4;
            if (pc_cold2[pc_i] > 0) {
                uint8_t bias = PC_COLD_WEAKEN;
                ins_rrpv = (uint8_t)((ins_rrpv + bias > maxRRPV) ? maxRRPV : (ins_rrpv + bias));
            }
        }
    } else {
        // Mode A fallback (Hawkeye-like)
        uint8_t pred = shct_demand[shct_i];
        ins_rrpv = (pred >= SHCT_FRIENDLY_THRES) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        if (is_prefetch(type)) {
            ins_rrpv = STREAM_TAIL_DEPTH;
            quarantine = 1;
        }
        // Weaken noisy PCs
        if (pc_cold2[pc_i] > 0) {
            uint8_t bias = PC_COLD_WEAKEN;
            ins_rrpv = (uint8_t)((ins_rrpv + bias > maxRRPV) ? maxRRPV : (ins_rrpv + bias));
        }
    }

    // Install
    rrpv[set][way]    = (ins_rrpv > maxRRPV) ? maxRRPV : ins_rrpv;
    runlock[set][way] = quarantine ? 1 : 0;
    hitcnt[set][way]  = 0;

    // Stream observation bookkeeping for quarantine
    if (runlock[set][way]) {
        if (stream_obs[set] < 63) stream_obs[set]++;
    }

    // PC coldness update on install for demand/prefetch
    if (is_demand(type) || is_prefetch(type)) {
        sat_inc_u2(pc_cold2[pc_i]);
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