#include <cstdint>
#include <algorithm>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type helpers (CRC2-conventional)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// Tunables (retuned per lineage feedback)
static constexpr uint8_t  maxRRPV               = 7;    // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;    // near MRU for hot PCs
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;    // far from MRU for cold PCs
static constexpr uint8_t  PREFETCH_TAIL         = 7;    // quarantine depth
static constexpr uint8_t  ZEUS_ASSIST_DEPTH     = 3;    // shallow for hot streams
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;    // run length (+1/+2) to mark stream
static constexpr uint8_t  HITS_PROMOTE_MRU      = 2;    // promote only on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_ON_TOUCH= 1;    // stream-locked lines demote on touch
static constexpr uint8_t  PC_USE_HOT_THRESH     = 7;    // TinyLFU hot cutoff (4-bit)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024; // faster decay to reduce noise
static constexpr uint32_t BANDIT_EPOCH          = 2048; // selector epoch (short)
static constexpr uint8_t  GSEL_MAX              = 31;   // global selector range
static constexpr uint8_t  GSEL_THRES            = 20;   // followers need high global gate
static constexpr int8_t   MODEB_ENABLE_THRESH   = 3;    // per-set bandit threshold

// Leader set sampling (64 sets total)
static constexpr uint32_t LLC_SET_BITS = 11;
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// Per-line packed state (conceptual bits): rrpv:3, hitcnt:2, stream_lock:1
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];

// Per-set selectors/evidence
static int8_t  bandit_score[LLC_SETS];     // saturating -8..+7
static uint8_t stream_evidence[LLC_SETS];  // 2-bit (0..3)

// Global selector and epoch accounting
static uint8_t  GSEL = 0;
static uint32_t leaderA_hits_epoch = 0;
static uint32_t leaderB_hits_epoch = 0;
static uint64_t access_count = 0;

// Mode A: Hawkeye-like SHiP (trained only on leaders)
#define SHCT_BITS 10
#define SHCT_SIZE (1u << SHCT_BITS) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit

// Leader-line metadata (64 leader sets * 16 ways = 1024 lines)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit effective
static uint8_t  hawk_used[64][LLC_WAYS];    // 1-bit reuse seen (demand)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 1-bit was prefetched
static uint8_t  hawk_valid[64][LLC_WAYS];   // 1-bit leader line valid

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 13) ^ (pc >> 27);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Mode B: StreamGuardian Delta (PC stride/run + TinyLFU-lite)
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line id

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10-bit packed in 16-bit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2-bit conf (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4-bit TinyLFU
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2-bit coldness bias

// Small helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t v) {
    rrpv[set][way] = (v > maxRRPV) ? maxRRPV : v;
}

// Stream run detector: +1 / +2 forward steps with short window
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
    if (LEADER_B(set)) return true;     // force B on leader-B
    if (LEADER_A(set)) return false;    // force A on leader-A
    // Followers need global gate, good bandit score, and recent stream evidence
    return (GSEL >= GSEL_THRES) && (bandit_score[set] >= MODEB_ENABLE_THRESH) && (stream_evidence[set] > 0);
}

static inline void epoch_tasks() {
    // Global gate drifts toward the better leader this epoch
    if (leaderB_hits_epoch > leaderA_hits_epoch) { if (GSEL < GSEL_MAX) GSEL++; }
    else if (leaderB_hits_epoch < leaderA_hits_epoch) { if (GSEL > 0) GSEL--; }
    leaderA_hits_epoch = leaderB_hits_epoch = 0;

    // Decay per-set stream evidence
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        if (stream_evidence[s] > 0) stream_evidence[s]--;
        // gently pull bandit score toward 0 to avoid stale wins
        if (bandit_score[s] > 0) bandit_score[s]--;
        else if (bandit_score[s] < 0) bandit_score[s]++;
    }
    // TinyLFU decay + coldness relaxation
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        sat_dec_u4(pc_use4[i]);
        if (pc_cold2[i] > 0) pc_cold2[i]--;
    }
}

// Initialize replacement state
void InitReplacementState() {
    std::memset(rrpv, maxRRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_lock, 0, sizeof(stream_lock));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(stream_evidence, 0, sizeof(stream_evidence));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    for (uint32_t i = 0; i < SHCT_SIZE; i++) {
        shct_demand[i] = 1;   // slightly warm start
        shct_prefetch[i] = 0; // prefetches start cold
    }
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(hawk_valid, 0, sizeof(hawk_valid));
    GSEL = 0;
    leaderA_hits_epoch = leaderB_hits_epoch = 0;
    access_count = 0;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // 1) Choose any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) SRRIP: prefer stream-locked victims among maxRRPV; else any maxRRPV
    for (int iter = 0; iter <= maxRRPV; iter++) {
        // prefer stream-locked at max
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV && stream_lock[set][w]) return w;
        }
        // then any at max
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age all
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    return 0; // fallback (should never happen)
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    // Periodic tasks
    access_count++;
    if ((access_count % BANDIT_EPOCH) == 0) {
        epoch_tasks();
    }

    // Update PC TinyLFU on demand refs
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
    }

    // Stream evidence (skip writebacks)
    if (!is_writeback(type)) {
        if (detect_and_update_stream(PC, paddr)) {
            if (stream_evidence[set] < 3) stream_evidence[set]++;
        }
    }

    // Leader accounting on hits
    if (hit) {
        if (is_demand(type)) {
            if (LEADER_A(set)) { leaderA_hits_epoch++; sat_dec_i8(bandit_score[set]); }
            else if (LEADER_B(set)) { leaderB_hits_epoch++; sat_inc_i8(bandit_score[set]); }
        }
    }

    // Per-access behavior
    if (hit) {
        // Multi-hit gate + stream demotion
        if (stream_lock[set][way]) {
            // demote on any touch
            if (rrpv[set][way] < maxRRPV) rrpv[set][way] = std::min<uint8_t>(maxRRPV, (uint8_t)(rrpv[set][way] + STREAM_DEMOTE_ON_TOUCH));
        }
        // Count only demand hits
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            if (hitcnt[set][way] >= HITS_PROMOTE_MRU) {
                rrpv_set(set, way, 0);     // promote to MRU
                stream_lock[set][way] = 0; // escape from stream quarantine
            }
            // Hawkeye reuse mark on leaders
            if (SEL_SAMPLED(set)) {
                uint32_t slot = LEADER_SLOT(set);
                hawk_used[slot][way] = 1;
            }
            // Coldness bias relax on reuse
            sat_dec_u2(pc_cold2[pc_index(PC)]);
        }
        // No promotion on prefetch hit
        return;
    }

    // MISS path: set initial insertion state
    uint8_t new_rrpv = INSERT_COLD_DEPTH;
    uint8_t new_lock = 0;
    uint8_t new_hitcnt = 0;

    // Mode decision
    bool use_modeB = modeB_enabled(set);

    // Compute SHiP stats
    uint16_t sig = pc_sig12(PC);
    uint32_t sidx = shct_idx(sig);

    // Compute stream/TinyLFU status
    uint32_t pidx = pc_index(PC);
    uint8_t sconf = pc_stream_conf[pidx];
    bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    if (is_prefetch(type)) {
        // Prefetches always quarantine at tail, never immediate promotion
        new_rrpv = PREFETCH_TAIL;
        new_lock = 1;
    } else if (use_modeB) {
        // Strong stream filter with short-reuse escape for hot PCs
        if (sconf >= 3) {
            if (pc_hot) {
                new_rrpv = ZEUS_ASSIST_DEPTH; // admit short reuse
                new_lock = 1;                 // still require 2nd demand hit to fully promote
            } else {
                new_rrpv = PREFETCH_TAIL;
                new_lock = 1;
                sat_inc_u2(pc_cold2[pidx]);   // bias against very streaming PCs
            }
        } else if (sconf >= STREAM_CONF_THRESH) {
            new_rrpv = PREFETCH_TAIL;
            new_lock = 1;
            sat_inc_u2(pc_cold2[pidx]);
        } else {
            // Not a stream: PC coldness + TinyLFU decide depth
            if ((pc_use4[pidx] < 2) || (pc_cold2[pidx] > 1)) {
                new_rrpv = INSERT_COLD_DEPTH;
            } else {
                new_rrpv = INSERT_WARM_DEPTH;
            }
            new_lock = 0;
        }
    } else {
        // Mode A: Hawkeye-like SHiP (demand path)
        if (shct_demand[sidx] > (SHCT_MAX >> 1)) new_rrpv = INSERT_WARM_DEPTH;
        else new_rrpv = INSERT_COLD_DEPTH;
        new_lock = 0;
    }

    // Install the new line
    rrpv_set(set, way, new_rrpv);
    stream_lock[set][way] = new_lock;
    hitcnt[set][way] = new_hitcnt;

    // Leader metadata maintenance (update previous and install new)
    if (SEL_SAMPLED(set)) {
        uint32_t slot = LEADER_SLOT(set);

        // On replacement, update SHCT for the old occupant if valid
        if (hawk_valid[slot][way]) {
            uint16_t prev_sig = hawk_sig[slot][way];
            uint32_t prev_idx = shct_idx(prev_sig);
            if (hawk_is_pref[slot][way]) {
                if (hawk_used[slot][way]) shct_inc(shct_prefetch[prev_idx]);
                else                      shct_dec(shct_prefetch[prev_idx]);
            } else {
                if (hawk_used[slot][way]) shct_inc(shct_demand[prev_idx]);
                else                      shct_dec(shct_demand[prev_idx]);
            }
        }

        // Install new leader-line metadata
        hawk_sig[slot][way] = sig;
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_valid[slot][way] = 1;
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