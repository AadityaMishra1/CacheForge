#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
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
static inline bool is_writeback(uint32_t t){ return (t == ACCESS_WRITEBACK); }

// ------------------------ Tunables ------------------------
// RRIP
static constexpr uint8_t maxRRPV               = 7;
// Mode A (Hawkeye-like SHiP-ish)
static constexpr uint8_t INSERT_WARM_DEPTH     = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH     = 6;  // near-tail
static constexpr uint8_t SHCT_FRIENDLY_THRES   = 14; // slightly lowered vs v2
// Mode B (Hysteretic RunShield)
static constexpr uint8_t STREAM_CONF_THRESH    = 2;  // two forward steps (+1/+2) => quarantine
static constexpr uint8_t STREAM_BYPASS_THRESH  = 3;  // at >=3 => hard-tail insert (bypass-like)
static constexpr uint8_t STREAM_TAIL_DEPTH     = 7;  // deep quarantine
static constexpr uint8_t STREAM_TAIL_SHALLOW   = 5;  // shallow quarantine for hot PC (zeusmp)
// Multi-hit gating
static constexpr uint8_t HITS_PROMOTE_NS       = 2;  // non-stream -> MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR      = 2;  // stream escape on 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_ON_TOUCH= 1;  // demote quarantined line on first demand touch
// PC TinyLFU + coldness
static constexpr uint8_t PC_USE_HOT_THRESH     = 6;  // 0..15 hot threshold
static constexpr uint8_t PC_COLD_WEAKEN        = 1;  // deepen insertion for cold PCs
// Selector
static constexpr uint32_t BANDIT_EPOCH         = 4096; // ops per epoch
static constexpr uint8_t  GSEL_MAX             = 31;
static constexpr uint8_t  GSEL_THRES           = 16;   // need >= to allow Mode B
static constexpr int8_t   MODEB_ENABLE_THRES   = 1;    // per-set bandit threshold
static constexpr uint8_t  SR_LOCAL_MIN_OBS     = 8;    // minimum observations before ratio gate

// ------------------------ Helpers ------------------------
static inline void sat_inc_u3(uint8_t &x){ if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t &x){ if (x > 0) x--; }
static inline void sat_inc_u2(uint8_t &x){ if (x < 3) x++; }
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
// rrpv:3b, hitcnt:2b, stream_flag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_flag[LLC_SETS][LLC_WAYS]; // 1 if stream-quarantined (or bypass-like)

// ------------------------ Selector state ------------------------
static int8_t  bandit_score[LLC_SETS]; // small per-set bias toward Mode B
static uint8_t GSEL = GSEL_THRES;      // global enable gate (start neutral)
static uint8_t stream_obs[LLC_SETS];   // observed quarantines (decayed)
static uint8_t stream_reuse[LLC_SETS]; // quarantined lines that later hit (decayed)

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

// Leader training buffers for 64 leader sets only (A/B forced)
static uint16_t hawk_sig[64][LLC_WAYS];   // store 10-bit index
static uint8_t  hawk_used[64][LLC_WAYS];  // reuse marker (demand hit)
static uint8_t  hawk_valid[64][LLC_WAYS]; // valid entry for training

// ------------------------ Mode B (RunShield + TinyLFU + Coldness) ------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines (10 bits)

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b last line
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b confidence (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// Stream run detector (+1/+2). Returns true if in streaming mode (conf >= STREAM_CONF_THRESH).
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
static inline bool stream_bypass_conf(uint64_t PC){
    return pc_stream_conf[pc_index(PC)] >= STREAM_BYPASS_THRESH;
}

// ------------------------ Housekeeping ------------------------
static uint32_t op_count = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;

static inline void end_epoch_maintenance() {
    // Global selector update using leader hits
    if (leaderB_hits_epoch > leaderA_hits_epoch) {
        if (GSEL < GSEL_MAX) GSEL++;
    } else {
        if (GSEL > 0) GSEL--;
    }
    leaderA_hits_epoch = leaderB_hits_epoch = 0;

    // Decay local stats and bandit scores
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        stream_obs[s]   >>= 1;
        stream_reuse[s] >>= 1;
        if (bandit_score[s] > 0) bandit_score[s]--;
        else if (bandit_score[s] < 0) bandit_score[s]++;
    }
    // Decay TinyLFU
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        sat_dec_u4(pc_use4[i]);
    }
}

// ------------------------ API ------------------------
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv)); // init to "old"
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_flag, 0, sizeof(stream_flag));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(stream_obs, 0, sizeof(stream_obs));
    std::memset(stream_reuse, 0, sizeof(stream_reuse));
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_valid, 0, sizeof(hawk_valid));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    op_count = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
    GSEL = GSEL_THRES;
}

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

    // SRRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

static inline bool allow_modeB(uint32_t set) {
    // Local ratio gate: reuse*3 >= obs or too few obs
    bool ratio_ok = (stream_obs[set] < SR_LOCAL_MIN_OBS) || (stream_reuse[set] * 3 >= stream_obs[set]);
    bool bandit_ok = (bandit_score[set] >= MODEB_ENABLE_THRES);
    return (GSEL >= GSEL_THRES) && (ratio_ok || bandit_ok);
}

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

    // Epoch maintenance
    op_count++;
    if ((op_count & (BANDIT_EPOCH - 1)) == 0) end_epoch_maintenance();

    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    uint32_t lslot = LEADER_SLOT(set);

    // Hit handling: promotion/demotion + training
    if (hit) {
        if (is_demand(type)) {
            // TinyLFU reward and coldness relief
            sat_inc_u4(pc_use4[pc_index(PC)]);
            if (pc_cold2[pc_index(PC)] > 0) pc_cold2[pc_index(PC)]--;

            // Multi-hit and stream quarantine handling
            if (stream_flag[set][way]) {
                // First demand touch demotes (scan resistance)
                if (STREAM_DEMOTE_ON_TOUCH && hitcnt[set][way] == 0) {
                    sat_inc_u3(rrpv[set][way]); // demote deeper
                    // record reuse of quarantined stream
                    if (stream_reuse[set] < 15) stream_reuse[set]++;
                    sat_inc_i4(bandit_score[set]);
                }
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_STR) {
                    rrpv[set][way] = 0; // escape quarantine
                    stream_flag[set][way] = 0;
                }
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            } else {
                // Non-stream: MRU only on 2nd demand hit
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_NS) {
                    rrpv[set][way] = 0; // MRU
                }
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            }
        }
        // Prefetch/writeback hits: no special promotion

        // Hawkeye leader reuse mark
        if (SEL_SAMPLED(set)) {
            hawk_used[lslot][way] = 1;
            if (leaderA) leaderA_hits_epoch++;
            if (leaderB) leaderB_hits_epoch++;
        }
        return;
    }

    // Miss path: we are about to install into (set, way). Train leaders on the victim first.
    if (SEL_SAMPLED(set)) {
        if (hawk_valid[lslot][way]) {
            uint32_t old_sig = hawk_sig[lslot][way] & (SHCT_SIZE - 1);
            if (hawk_used[lslot][way]) shct_inc(shct_demand[old_sig]);
            else                       shct_dec(shct_demand[old_sig]);
        }
        hawk_valid[lslot][way] = 0;
        hawk_used[lslot][way]  = 0;
    }

    // Penalize failed quarantines (stream with no reuse) on eviction
    if (stream_flag[set][way] && (hitcnt[set][way] == 0)) {
        sat_dec_i4(bandit_score[set]);
    }

    // Default insertion state
    uint8_t ins_rrpv = INSERT_COLD_DEPTH;
    uint8_t ins_stream = 0;

    // Writeback handling: never bypass; insert modestly old
    if (is_writeback(type)) {
        ins_rrpv = STREAM_TAIL_SHALLOW; // safe, low priority
        ins_stream = 0;
    } else if (is_prefetch(type)) {
        // Prefetch quarantine: deep tail, no stream tag (keeps non-stream promotion rules)
        ins_rrpv = STREAM_TAIL_DEPTH;
        ins_stream = 0;
    } else {
        // Demand miss: decide between Mode A and Mode B
        bool modeB = false;
        if (leaderA) modeB = false;
        else if (leaderB) modeB = true;
        else modeB = allow_modeB(set);

        // Update stream detector regardless to build confidence
        bool is_stream = detect_and_update_stream(PC, paddr);
        bool bypass_like = stream_bypass_conf(PC);

        if (modeB && is_stream) {
            // Strong stream suppression for lbm; protect zeusmp reuse when PC is hot
            ins_stream = 1;
            if (bypass_like) {
                ins_rrpv = STREAM_TAIL_DEPTH; // approximate bypass via deep-tail insert
            } else {
                ins_rrpv = (pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH) ? STREAM_TAIL_SHALLOW : STREAM_TAIL_DEPTH;
            }
        } else {
            // Mode A (Hawkeye-like) with PC coldness bias and TinyLFU shading
            uint32_t sig = shct_idx_from_pc(PC);
            bool friendly = (shct_demand[sig] >= SHCT_FRIENDLY_THRES);
            ins_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;

            // PC coldness deepens insertion slightly
            if (pc_cold2[pc_index(PC)] >= 2) {
                uint8_t add = PC_COLD_WEAKEN;
                uint8_t tmp = ins_rrpv + add;
                ins_rrpv = (tmp > maxRRPV) ? maxRRPV : tmp;
            }
            // TinyLFU shading: slightly deepen if not hot and not friendly
            if (!friendly && pc_use4[pc_index(PC)] < PC_USE_HOT_THRESH) {
                if (ins_rrpv < maxRRPV) ins_rrpv++;
            }
        }
    }

    // Install decision effects
    rrpv[set][way] = (ins_rrpv > maxRRPV) ? maxRRPV : ins_rrpv;
    hitcnt[set][way] = 0;
    stream_flag[set][way] = ins_stream;

    if (ins_stream) {
        if (stream_obs[set] < 15) stream_obs[set]++;
    }

    // SHiP training state for leaders on new install
    if (SEL_SAMPLED(set)) {
        uint32_t sig = shct_idx_from_pc(PC);
        hawk_sig[lslot][way]   = (uint16_t)sig;
        hawk_used[lslot][way]  = 0;
        hawk_valid[lslot][way] = 1;
    }

    // PC stats update on miss
    if (is_demand(type)) {
        // Miss without prior reuse increases coldness
        if (pc_cold2[pc_index(PC)] < 3) pc_cold2[pc_index(PC)]++;
        // Slightly penalize TinyLFU on misses
        sat_dec_u4(pc_use4[pc_index(PC)]);
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}