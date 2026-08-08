#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

// ChampSim CRC2 constants
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (aligned to CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK);
}

// RRIP parameters (3-bit)
static constexpr uint8_t maxRRPV = 7;

// Tunables
static constexpr uint8_t STREAM_CONF_THRESH = 2;   // +1/+2 forward steps to mark stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2;   // non-stream promote after 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3;   // quarantined/stream promote after 3rd demand hit
static constexpr uint8_t INSERT_DEPTH_A     = 2;   // Mode A warm insertion depth (near MRU)
static constexpr int8_t  MODEB_THRESH       = 2;   // enable Mode B when bandit_score >= 2
static constexpr uint8_t SR_WINDOW_INIT     = 6;   // short-reuse window (per-PC countdown)
static constexpr uint8_t SHCT_HOT_THRESH    = 8;   // SHCT hot threshold (4-bit)
static constexpr uint8_t STREAM_DEMOTE_ON_TOUCH = 1; // demote quarantined lines on touch

// Leader set sampling (64 sets): even->A, odd->B
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // ((low 6 bits) == (next 6 bits))
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// Per-line metadata (bit-packed conceptually; stored in bytes here)
// rrpv: 3b, quarantine/stream_lock: 1b, hitcnt: 2b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1 if quarantined (stream/prefetch)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 0..3 (sat)

// Per-set bandit (signed, saturating) to decide A vs B on follower sets
static int8_t bandit_score[LLC_SETS];

// Mode A: tiny SHCT (Hawkeye-lite insertion guidance)
static constexpr uint32_t SHCT_SIZE = 512; // 512 entries x 4b
static uint8_t shct[SHCT_SIZE];            // store 0..15 (we use 0..15, hot if >= SHCT_HOT_THRESH)
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// Mode B: PC stream + coldness + short-reuse hints
static constexpr uint32_t PC_TBL_SIZE = 512;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index of last touch (stored in 16b)
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

// RRIP victim selection with bounded global aging to guarantee a victim
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // Try to find maxRRPV quickly
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w; // safety (should have been caught earlier)
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // Age until a victim emerges (at most 8 passes)
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: choose the most aged
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// Victim selection
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

    uint32_t v = rrip_victim_and_age(set, current_set);

    // Bandit feedback on eviction: only account quarantined (Mode B's hallmark) lines
    if (stream_lock[set][v]) {
        if (hitcnt[set][v] >= HITS_PROMOTE_NS) {
            // Quarantine rescued a reuser => reward B
            sat_inc_i8(bandit_score[set]);
        } else {
            // Dead/one-hit under quarantine => penalize B more aggressively
            sat_dec_i8(bandit_score[set]);
            sat_dec_i8(bandit_score[set]);
        }
    }

    return v;
}

// Update replacement state on each access/fill
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

    if (type == ACCESS_WRITEBACK) return;

    // Per-PC stream/coldness update (based on current access)
    const uint32_t pidx = pc_index(PC);
    const uint16_t cur_line = line12(paddr);
    // Stream detector: +1/+2 forward steps increase confidence, otherwise decay
    int16_t delta = (int16_t)cur_line - (int16_t)pc_last_line[pidx];
    if (delta == 1 || delta == 2) {
        if (pc_conf[pidx] < 3) pc_conf[pidx]++;
    } else {
        if (pc_conf[pidx] > 0) pc_conf[pidx]--;
    }
    pc_last_line[pidx] = cur_line;

    // Decide policy mode for this set (leaders explore; followers depend on bandit)
    bool forceA = LEADER_A(set);
    bool forceB = LEADER_B(set);
    bool useB = forceB || (!forceA && (bandit_score[set] >= MODEB_THRESH));

    // On a hit: gated promotions
    if (hit) {
        // Count only demand hits for promotion gating
        if (is_demand(type)) {
            sat_inc_u2(hitcnt[set][way]); // 0->1->2->3 (sat)
            // Optional demotion for quarantined touches (scan resistance)
            if (STREAM_DEMOTE_ON_TOUCH && stream_lock[set][way] && rrpv[set][way] < maxRRPV) {
                rrpv[set][way]++;
            }

            // Short-reuse window can allow earlier rescue for frequently short-reuse PCs
            bool short_window = (pc_sr_win[pidx] > 0);
            uint8_t need_hits = stream_lock[set][way] ? (short_window ? 2 : HITS_PROMOTE_STR) : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need_hits) {
                rrpv[set][way] = 0;          // promote to MRU
                stream_lock[set][way] = 0;   // clear quarantine once rescued
                // Train PC usefulness positively on proven reuse
                sat_inc_u4(pc_cold[pidx]);
                if (short_window) { // consume short-reuse window gradually
                    sat_dec_u3(pc_sr_win[pidx]);
                }
                // Mode A SHCT: reward hot PCs on 2+ demand hits
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    if (shct[shct_idx(PC)] < 15) shct[shct_idx(PC)]++;
                }
            }
        }
        return;
    }

    // Miss/Fill path (re-initialize metadata for the filled line)
    hitcnt[set][way] = 0;
    stream_lock[set][way] = 0;

    // Prefetches always quarantine at tail; never promote on first touch
    if (type == ACCESS_PREFETCH) {
        rrpv[set][way] = maxRRPV;
        stream_lock[set][way] = 1; // quarantined
        return;
    }

    // Demand fill: choose Mode A vs Mode B insertion
    bool stream_now = (pc_conf[pidx] >= STREAM_CONF_THRESH) && (delta == 1 || delta == 2);

    if ((useB && stream_now)) {
        // Mode B: Forward-run stream quarantine (bypass-equivalent)
        rrpv[set][way] = maxRRPV;
        stream_lock[set][way] = 1;
        // Start a short-reuse window to allow quicker rescue for this PC if reuse appears
        pc_sr_win[pidx] = SR_WINDOW_INIT;
        pc_short[pidx] = 1;
        // Do not train SHCT here; keep Mode A unbiased
        return;
    }

    if (useB) {
        // Mode B non-stream: PC coldness-aware insertion
        // Cold PCs -> hard tail; warm/hot PCs -> mid/near-MRU
        uint8_t cold = pc_cold[pidx];
        if (cold < (SHCT_HOT_THRESH >> 1)) {        // very cold
            rrpv[set][way] = maxRRPV;
            stream_lock[set][way] = 1;              // quarantine cold fills
        } else if (cold < SHCT_HOT_THRESH) {        // warming
            rrpv[set][way] = (maxRRPV > 3) ? 3 : maxRRPV - 1;
        } else {                                    // hot PC
            rrpv[set][way] = 1;
        }
        return;
    }

    // Mode A (Hawkeye-lite): SHCT-guided insertion; promote only after 2nd hit
    {
        uint8_t val = shct[shct_idx(PC)];
        if (val >= SHCT_HOT_THRESH) {
            // Friendly PC: near MRU insertion
            rrpv[set][way] = (INSERT_DEPTH_A < maxRRPV) ? INSERT_DEPTH_A : maxRRPV - 1;
        } else {
            // Averse PC: insert far from MRU
            rrpv[set][way] = maxRRPV;
            stream_lock[set][way] = 1; // quarantine until proven
        }
    }
}

// Print end-of-simulation statistics (required empty)
void PrintStats() {}

// Print periodic statistics (required empty)
void PrintStats_Heartbeat() {}