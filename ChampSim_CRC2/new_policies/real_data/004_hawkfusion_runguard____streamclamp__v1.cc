#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2 LLC)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK);
}

// RRIP (3-bit)
static constexpr uint8_t maxRRPV = 7;

// Tunables (StreamClamp variant)
static constexpr uint8_t STREAM_CONF_THRESH = 2;   // +1/+2 forward steps to mark stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2;   // non-stream promote at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3;   // stream/quarantine promote at 3rd demand hit
static constexpr uint8_t HITS_PROMOTE_SHORT = 2;   // short-reuse PCs promote at 2nd hit even if stream
static constexpr uint8_t INSERT_DEPTH_A     = 2;   // Mode A warm insertion (hot PC)
static constexpr uint8_t INSERT_TAIL_NEAR   = 6;   // near tail (not max)
static constexpr int8_t  MODEB_THRESH       = 2;   // enable Mode B when bandit_score >= 2
static constexpr uint8_t SR_WINDOW_INIT     = 6;   // short-reuse countdown
static constexpr uint8_t SHCT_HOT_THRESH    = 8;   // hot if >= 8 (4-bit)
static constexpr uint8_t STREAM_DEMOTE_ON_TOUCH = 1; // demote quarantined on subthreshold hit

// Leader set sampling (64 sets): even->A, odd->B
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// Per-line metadata (packed conceptually)
// rrpv: 3b, stream_lock/quarantine: 1b, hitcnt: 2b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1 if quarantined (stream/prefetch)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 0..3 (sat)

// Per-set bandit selector (follower sets only)
static int8_t bandit_score[LLC_SETS];

// Mode A: tiny SHCT (Hawkeye-lite)
static constexpr uint32_t SHCT_SIZE = 512; // 512 entries x 4b
static uint8_t shct[SHCT_SIZE];            // 0..15
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// Mode B: PC stream + coldness + short-reuse hints
static constexpr uint32_t PC_TBL_SIZE = 512;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 4-bit PC "usefulness" (decayed)
static uint8_t  pc_sr_win[PC_TBL_SIZE];    // 3-bit short-reuse countdown
static uint8_t  pc_short[PC_TBL_SIZE];     // 1-bit short-reuse hint
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// Saturating helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x) { if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }
static inline void sat_toward_zero(int8_t& x) { if (x > 0) x--; else if (x < 0) x++; }

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;  // long re-reference distance initially
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
        }
    }
    std::memset(shct, 0, sizeof(shct));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
    std::memset(pc_cold, 0, sizeof(pc_cold));
    std::memset(pc_sr_win, 0, sizeof(pc_sr_win));
    std::memset(pc_short, 0, sizeof(pc_short));
}

// RRIP victim selection with bounded aging to guarantee a victim
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w; // safety
        if (rrpv[set][w] == maxRRPV) return w;
    }
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    return rrip_victim_and_age(set, current_set);
}

// Update replacement state
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

    // Derive PC indices and current line
    const uint32_t pi = pc_index(PC);
    const uint32_t hi = shct_idx(PC);
    const uint16_t curL = line12(paddr);
    const uint16_t lastL = pc_last_line[pi];

    // Forward-run (+1/+2) detection based on previous touch of this PC
    bool fwd1 = (uint16_t)(lastL + 1u) == curL;
    bool fwd2 = (uint16_t)(lastL + 2u) == curL;
    bool fwd12 = (fwd1 || fwd2);
    bool stream_pc = (pc_conf[pi] >= STREAM_CONF_THRESH) && fwd12;
    bool short_pc = (pc_short[pi] != 0);

    // Per-set mode selection: default Hawkeye-lite (A), enable B only if strong evidence
    bool modeB = LEADER_B(set) || (!LEADER_A(set) && (bandit_score[set] >= MODEB_THRESH));

    // Periodic decay toward neutrality on follower sets (very light)
    if (!SEL_SAMPLED(set)) {
        if ((PC & 0x3Fu) == 0u) sat_toward_zero(bandit_score[set]);
    }

    // On hit handling
    if (hit) {
        // Update SHCT/PC coldness on useful demand hits
        if (is_demand(type)) {
            sat_inc_u4(shct[hi]);
            sat_inc_u4(pc_cold[pi]);

            // multi-hit promotion gate
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            uint8_t thr = stream_lock[set][way] ? (short_pc ? HITS_PROMOTE_SHORT : HITS_PROMOTE_STR) : HITS_PROMOTE_NS;

            if (hitcnt[set][way] >= thr) {
                // Graduate from quarantine (if any) and reward Mode B on followers
                rrpv[set][way] = 0;
                if (stream_lock[set][way] && !SEL_SAMPLED(set)) {
                    sat_inc_i8(bandit_score[set]);
                }
                stream_lock[set][way] = 0;
            } else {
                // Subthreshold: keep stream lines cold; non-stream get gentle promotion
                if (stream_lock[set][way] && STREAM_DEMOTE_ON_TOUCH) {
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                } else {
                    if (rrpv[set][way] > 0) rrpv[set][way]--;
                }
            }

            // Short-reuse hint refresh
            pc_short[pi] = 1;
            pc_sr_win[pi] = SR_WINDOW_INIT;
        } else if (type == ACCESS_PREFETCH) {
            // Prefetch hit: keep at tail; do not count toward promotion threshold
            if (stream_lock[set][way] && STREAM_DEMOTE_ON_TOUCH) {
                if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            }
        }

        // Stream confidence update
        if (fwd12) sat_inc_u2(pc_conf[pi]); else sat_dec_u2(pc_conf[pi]);
        pc_last_line[pi] = curL;

        // Countdown short-reuse window
        if (pc_short[pi]) {
            if (pc_sr_win[pi] > 0) pc_sr_win[pi]--;
            if (pc_sr_win[pi] == 0) pc_short[pi] = 0;
        }

        return;
    }

    // Miss path (new line will be installed into 'way'):
    // Score the evicted old line outcome for bandit (only followers)
    if (!SEL_SAMPLED(set)) {
        if (stream_lock[set][way]) {
            // Quarantined line died without graduating
            sat_dec_i8(bandit_score[set]);
        }
        // Non-quarantined line evicted is neutral here (graduations rewarded on hit path)
    }

    // Decide insertion for the incoming line
    uint8_t ins_rrpv = maxRRPV; // default hard tail
    uint8_t ins_lock = 1;       // default quarantine
    hitcnt[set][way] = 0;

    // Mode A: Hawkeye-lite
    if (!modeB || LEADER_A(set)) {
        // Prefetches always deep tail
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV;
            ins_lock = 1;
        } else if (type == ACCESS_WRITEBACK) {
            // Never bypass writebacks: warm-ish insertion
            ins_rrpv = INSERT_TAIL_NEAR;
            ins_lock = 0;
        } else {
            // Demand: SHCT-guided
            if (shct[hi] >= SHCT_HOT_THRESH) {
                ins_rrpv = INSERT_DEPTH_A; // warm insertion near MRU
                ins_lock = 0;
            } else {
                ins_rrpv = INSERT_TAIL_NEAR; // cold PCs closer to tail
                ins_lock = 1;
            }
        }
    }
    // Mode B: StreamClamp
    else {
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; // deepest tail
            ins_lock = 1;
        } else if (type == ACCESS_WRITEBACK) {
            // writebacks: keep resident but not MRU
            ins_rrpv = INSERT_TAIL_NEAR;
            ins_lock = 0;
        } else {
            // Demand
            if (stream_pc && !short_pc) {
                // Strong stream → hard-tail quarantine (bypass-equivalent)
                ins_rrpv = maxRRPV;
                ins_lock = 1;
                // Penalize PC coldness on streams
                sat_dec_u4(pc_cold[pi]);
            } else if (stream_pc && short_pc) {
                // Short-reuse stream → shallower tail but still quarantined
                ins_rrpv = INSERT_TAIL_NEAR;
                ins_lock = 1;
            } else {
                // Not stream-detected: use PC coldness and short-reuse hint
                if (pc_cold[pi] >= 8 || short_pc) {
                    ins_rrpv = INSERT_DEPTH_A; // warm insertion
                    ins_lock = 0;
                } else {
                    ins_rrpv = INSERT_TAIL_NEAR; // cautious tail insertion
                    ins_lock = 1;
                }
            }
        }
    }

    // Apply insertion state
    rrpv[set][way] = ins_rrpv;
    stream_lock[set][way] = ins_lock;

    // Update SHCT on miss (PC not yet proven hot)
    if (is_demand(type)) {
        sat_dec_u4(shct[hi]);
        // If we just inserted due to stream classification, cool the PC a bit
        if (stream_pc) sat_dec_u4(pc_cold[pi]);
    }

    // Stream confidence tracking update after handling miss
    if (fwd12) sat_inc_u2(pc_conf[pi]); else sat_dec_u2(pc_conf[pi]);
    pc_last_line[pi] = curL;

    // Countdown short-reuse window
    if (pc_short[pi]) {
        if (pc_sr_win[pi] > 0) pc_sr_win[pi]--;
        if (pc_sr_win[pi] == 0) pc_short[pi] = 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}