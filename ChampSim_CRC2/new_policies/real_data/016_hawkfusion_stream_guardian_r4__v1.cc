#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
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
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // Stream-Guardian
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (R4) -----------------------------------
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU insertion
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail insertion
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for streams/prefetch
static constexpr uint8_t  STREAM_SHALLOW_TAIL   = 5;   // shallower for hot-PC streams
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // +1/+2 forward steps required
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 3;   // stream promote on 3rd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;   // hot-PC stream: early escape
static constexpr uint8_t  PC_USE_HOT_THRESH     = 8;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_COLD_STRICT_TH     = 2;   // 0..3 (>=2 => cold)
static constexpr uint32_t LFU_DECAY_PERIOD      = 1024;// faster decay to kill stale PCs
static constexpr uint32_t BANDIT_EPOCH          = 8192;// longer selector epoch
static constexpr uint8_t  GSEL_MAX              = 31;  // global gate range
static constexpr uint8_t  GSEL_THRES            = 18;  // followers prefer Mode B if >=
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;   // per-set bias threshold

// ---------------- Per-line metadata (bit-packed conceptually) ----
// rrpv:3b, hitcnt:2b (0..3), stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];

// ---------------- Per-set and global selectors -------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7] follower bias toward Mode B
static uint8_t GSEL = 0;               // global gate (0..31)
static uint64_t access_count = 0;

// Leader performance accumulators (hysteretic global gate)
static int32_t leaderA_good = 0; // hits - misses
static int32_t leaderB_good = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHiP) ------------
#define SHCT_SIZE (1u << 10) // 1024 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
    // 12b PC signature
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t& x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x) { if (x > 0) x--; }

// Leader-A per-line training buffers (64 sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // saw reuse (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (Stream-Guardian PC tables) --------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B lines -> 10b

// Per-PC state
static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b in 16b storage
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b coldness (0..3)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -8) x--; }
static inline void sat_inc_u5(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) { if (rrpv[set][way] > 0) rrpv[set][way]--; }
static inline void rrpv_demote(uint32_t set, uint32_t way)  { if (rrpv[set][way] < maxRRPV) rrpv[set][way]++; }

// +1/+2 forward-run detector with 2-step confidence
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx  = pc_index(PC);
    uint16_t ln   = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward  = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        pc_stream_conf[idx] = 0;
    }
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;    // force Mode B on B leaders
    if (LEADER_A(set)) return false;   // force Mode A on A leaders
    // Followers: calm global gate plus per-set bias, default to Hawkeye
    if (GSEL < GSEL_THRES) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

void InitReplacementState() {
    std::memset(rrpv, 0, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(stream_tag, 0, sizeof(stream_tag));
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start cold
        }
        bandit_score[s] = 0;
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    access_count = 0;
    leaderA_good = leaderB_good = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // RRIP victim selection with aging
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines, saturating
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
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
    access_count++;

    // Per-PC accounting
    uint32_t pidx = pc_index(PC);
    bool stream_pred = detect_and_update_stream(PC, paddr);

    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pidx]);
    }

    // Periodic TinyLFU/cold decay and selector epoch roll
    if ((access_count % LFU_DECAY_PERIOD) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
            sat_dec_u4(pc_use4[i]);
            if (pc_cold2[i] > 0) pc_cold2[i]--; // mild coldness decay
        }
    }
    if ((access_count % BANDIT_EPOCH) == 0) {
        if (leaderB_good > leaderA_good) {
            if (GSEL < GSEL_MAX) GSEL++;
        } else if (leaderA_good > leaderB_good) {
            if (GSEL > 0) GSEL--;
        }
        leaderA_good = leaderB_good = 0;
    }

    bool useB = modeB_enabled(set);
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);

    // Hit path
    if (hit) {
        // Leader bookkeeping for selector stability
        if (leaderA) leaderA_good++;
        if (leaderB) leaderB_good++;

        // Multi-hit promotion gating
        if (is_demand(type)) {
            // Increment demand-hit counter (prefetch hits do not increase)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            // Hawkeye training: mark reuse for A leaders
            if (leaderA) {
                uint32_t slot = LEADER_SLOT(set);
                hawk_used[slot][way] = 1;
            }

            bool pc_hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pidx] < PC_COLD_STRICT_TH);

            if (stream_tag[set][way]) {
                // Stream demote-on-touch
                rrpv_demote(set, way);
                uint8_t need = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
                if (hitcnt[set][way] >= need) {
                    // escape quarantine
                    stream_tag[set][way] = 0;
                    rrpv_set(set, way, 0);
                }
            } else {
                // Non-stream: promote only on 2nd+ demand hit
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0);
                } else {
                    // gentle promote
                    rrpv_promote(set, way);
                }
            }

            // PC coldness reward on demonstrated reuse (>=2 demand hits)
            if (hitcnt[set][way] >= 2) {
                if (pc_cold2[pidx] > 0) pc_cold2[pidx]--;
            }
        } else {
            // Prefetch or writeback hits: do not promote aggressively
            if (stream_tag[set][way]) rrpv_demote(set, way);
            else rrpv_promote(set, way);
        }

        // Follower bandit update (calm): reward chosen mode on hit
        if (!leaderA && !leaderB) {
            if (useB) sat_inc_i8(bandit_score[set]);
            else      sat_dec_i8(bandit_score[set]);
        }
        return;
    }

    // Miss path (fill). Train leaders first (eviction outcome known here).
    if (leaderA) {
        uint32_t slot = LEADER_SLOT(set);
        // Train SHCT on the victim that is being evicted from this way
        uint16_t vsig = hawk_sig[slot][way];
        uint32_t si = shct_idx(vsig);
        if (hawk_is_pref[slot][way]) {
            if (hawk_used[slot][way]) shct_inc(shct_prefetch[si]);
            else                      shct_dec(shct_prefetch[si]);
        } else {
            if (hawk_used[slot][way]) shct_inc(shct_demand[si]);
            else                      shct_dec(shct_demand[si]);
        }
        // Reset for new occupant
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
        hawk_used[slot][way]    = 0;
    }

    // Selector leader accounting on miss
    if (leaderA) leaderA_good--;
    if (leaderB) leaderB_good--;

    // Decide insertion and tags
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t st_tag = 0;

    bool pc_hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH) && (pc_cold2[pidx] < PC_COLD_STRICT_TH);
    bool pc_cold = (pc_cold2[pidx] >= PC_COLD_STRICT_TH);

    if (is_writeback(type)) {
        ins_depth = INSERT_WARM_DEPTH; // never bypass writebacks
        st_tag = 0;
    } else if (is_prefetch(type)) {
        ins_depth = STREAM_TAIL_DEPTH; // quarantine prefetches
        st_tag = 1;
    } else { // demand
        if (useB) {
            if (stream_pred && !pc_hot) {
                // Hard quarantine/bypass for streams
                ins_depth = STREAM_TAIL_DEPTH;
                st_tag = 1;
            } else if (stream_pred && pc_hot) {
                ins_depth = STREAM_SHALLOW_TAIL;
                st_tag = 1;
            } else {
                // Non-stream: hot warm-insert, otherwise cold
                ins_depth = (pc_hot && !pc_cold) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
                st_tag = 0;
            }
        } else {
            // Mode A (Hawkeye-like SHiP)
            uint32_t si = shct_idx(pc_sig12(PC));
            bool friendly = (shct_demand[si] >= (SHCT_MAX / 2));
            ins_depth = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            st_tag = 0; // SHiP path does not stream-tag
        }

        // Demand miss implies potential deadness: raise coldness slightly
        sat_inc_u2(pc_cold2[pidx]);
    }

    // Install new state
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = st_tag;

    // Follower bandit update (penalize chosen mode on miss)
    if (!leaderA && !leaderB) {
        if (useB) sat_dec_i8(bandit_score[set]);
        else      sat_inc_i8(bandit_score[set]);
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}