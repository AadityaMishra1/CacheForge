#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) { return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK); }

static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP

// Tunables
static constexpr uint8_t STREAM_CONF_THRESH = 2;  // +1/+2 forward steps to tag stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2;  // non-stream promotion on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3;  // stream base promotion on 3rd demand hit
static constexpr uint8_t INSERT_DEPTH_A     = 2;  // Mode A hot insert depth (near MRU)
static constexpr int8_t  MODEB_THRESH       = 3;  // enable Mode B when bandit_score >= 3
static constexpr uint8_t SR_WINDOW_INIT     = 7;  // short-reuse window (per-PC countdown)
static constexpr uint8_t SHCT_HOT_THRESH    = 8;  // SHCT hot threshold (4-bit counter)
static constexpr uint8_t STREAM_DEMOTE_ON_TOUCH = 1; // demote streaming lines when not yet promoted

// Per-line metadata: rrpv(3b), stream_lock(1b), hitcnt(2b: 3 used as prefetch sentinel)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];

// Per-set bandit selector (signed, saturating)
static int8_t bandit_score[LLC_SETS];

// Mode A: Hawkeye-like SHCT (1K x 4b logical)
static constexpr uint32_t SHCT_SIZE = 1024;
static uint8_t hawk_shct[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// Mode B: PC stream detector (+1/+2), coldness and short-reuse hints
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index of last access by PC
static uint16_t pc_sr_line[PC_TBL_SIZE];   // 12-bit line index tagged at last stream fill
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 4-bit PC coldness (0=cold..15=hot)
static uint8_t  pc_sr_win[PC_TBL_SIZE];    // 3-bit short-reuse window (counts down)
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
            rrpv[s][w] = maxRRPV;  // tail (long re-reference)
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
        }
    }
    std::memset(hawk_shct, 0, sizeof(hawk_shct));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_sr_line, 0, sizeof(pc_sr_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
    std::memset(pc_cold, 0, sizeof(pc_cold));
    std::memset(pc_sr_win, 0, sizeof(pc_sr_win));
    std::memset(pc_short, 0, sizeof(pc_short));
}

// Ensure a victim exists: RRIP aging with bounded passes
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // If any invalid way exists, the caller handles it before calling this
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all lines by 1 (saturating)
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
    // RRIP victim selection
    uint32_t v = rrip_victim_and_age(set, current_set);

    // Bandit feedback on eviction: reward/penalize Mode B only on quarantined (stream) lines,
    // fast decay on dead stream lines; mild decay on general dead lines.
    if (stream_lock[set][v]) {
        if (hitcnt[set][v] >= 2) {
            sat_inc_i8(bandit_score[set]);   // quarantine paid off (short reuse rescued)
        } else {
            sat_dec_i8(bandit_score[set]);   // dead stream -> decay
            sat_dec_i8(bandit_score[set]);   // fast decay
        }
    } else {
        if (hitcnt[set][v] == 0) {
            sat_dec_i8(bandit_score[set]);   // dead non-stream
        }
    }
    return v;
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

    // Never act on writebacks
    if (type == ACCESS_WRITEBACK) return;

    const uint32_t pidx = pc_index(PC);
    const uint16_t l12  = line12(paddr);

    // Per-PC forward-run detector (+1/+2 steps)
    int16_t delta = (int16_t)l12 - (int16_t)pc_last_line[pidx];
    bool fwd = (delta == 1) || (delta == 2);
    if (fwd) sat_inc_u2(pc_conf[pidx]); else sat_dec_u2(pc_conf[pidx]);
    pc_last_line[pidx] = l12;

    // Short-reuse window ageing
    if (pc_sr_win[pidx] > 0) pc_sr_win[pidx]--;
    if (pc_sr_win[pidx] == 0) pc_short[pidx] = 0; // clear hint when window expires

    // SHCT and PC-coldness updates: demand hits strengthen, demand misses weaken
    if (hit) {
        if (is_demand(type)) {
            sat_inc_u4(hawk_shct[shct_idx(PC)]);
            sat_inc_u4(pc_cold[pidx]);
        }
    } else {
        if (is_demand(type)) {
            sat_dec_u4(hawk_shct[shct_idx(PC)]);
            sat_dec_u4(pc_cold[pidx]);
        }
    }

    // Promotion thresholds
    const uint8_t ns_promote_hits  = HITS_PROMOTE_NS;
    const uint8_t str_base_promote = HITS_PROMOTE_STR;

    if (hit) {
        // On hit, update hit counters and decide promotions/demotions
        if (type == ACCESS_PREFETCH) return; // shouldn't occur as a hit type, but keep safe

        // Prefetch quarantine: first demand touch after prefetch should not promote
        if (hitcnt[set][way] == 3) {
            hitcnt[set][way] = 1; // first demand touch after prefetch
            // Keep position conservative
            if (stream_lock[set][way]) {
                // demote streaming line slightly to keep near tail
                for (uint8_t i = 0; i < STREAM_DEMOTE_ON_TOUCH; i++) sat_inc_u3(rrpv[set][way]);
            } else {
                if (rrpv[set][way] > 0) rrpv[set][way]--;
            }
            return;
        }

        // Increment demand-hit count (max 3)
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;

        if (stream_lock[set][way]) {
            // Elastic rescue for stream-tagged lines
            bool short_reuse = (pc_short[pidx] != 0) || (pc_sr_win[pidx] > 0);
            uint8_t need = short_reuse ? 2 : str_base_promote;
            if (hitcnt[set][way] >= need) {
                // Promote and clear stream lock
                rrpv[set][way] = 0;
                stream_lock[set][way] = 0;
                // Once observed, keep short_reuse sticky until window expires
                pc_short[pidx] = 1;
            } else {
                // Not enough hits yet: demote to keep near tail
                for (uint8_t i = 0; i < STREAM_DEMOTE_ON_TOUCH; i++) sat_inc_u3(rrpv[set][way]);
            }
        } else {
            // Non-stream: promote on 2nd demand hit
            if (hitcnt[set][way] >= ns_promote_hits) {
                rrpv[set][way] = 0; // MRU
            } else {
                // Mild recency improvement
                if (rrpv[set][way] > 0) rrpv[set][way]--;
            }
        }
        return;
    }

    // Miss path: choose insertion behavior (way was chosen already)
    // Mode selection
    bool modeB = (bandit_score[set] >= MODEB_THRESH);

    // Hotness by SHCT and coldness
    uint8_t shct_val = hawk_shct[shct_idx(PC)];
    bool shct_hot = (shct_val >= SHCT_HOT_THRESH);
    bool pc_is_hot = (pc_cold[pidx] >= SHCT_HOT_THRESH);

    // Stream prediction
    bool stream_pred = modeB && (pc_conf[pidx] >= STREAM_CONF_THRESH);
    bool mark_stream = false;

    if (type == ACCESS_PREFETCH) {
        // Prefetch: always tail insert, quarantine
        rrpv[set][way] = maxRRPV;
        hitcnt[set][way] = 3;     // sentinel for prefetch quarantine
        stream_lock[set][way] = stream_pred ? 1 : 0;
        if (stream_pred) {
            pc_sr_win[pidx]  = SR_WINDOW_INIT;
            pc_sr_line[pidx] = l12;
        }
        return;
    }

    // Demand insertion
    if (stream_pred) {
        // Third or more forward step? Apply deterministic 75% quarantine unless SHCT says hot
        if ((pc_conf[pidx] >= 3) && !shct_hot) {
            uint32_t sel = ((uint32_t)(PC >> 2) ^ (uint32_t)l12) & 3u; // 0..3
            mark_stream = (sel != 0); // ~75% quarantine
        } else {
            mark_stream = true; // at least two forward steps: hard quarantine
        }
    }

    if (mark_stream) {
        // Hard-tail quarantine for streams
        rrpv[set][way] = maxRRPV;
        stream_lock[set][way] = 1;
        hitcnt[set][way] = 0;
        pc_sr_win[pidx]  = SR_WINDOW_INIT;
        pc_sr_line[pidx] = l12;
        // Do not promote on first hit
        return;
    }

    // Mode A (or overridden as hot): Hawkeye-like adaptive insertion
    // Hot PCs near-MRU, cold at tail; default conservative for neutrality
    if (shct_hot || pc_is_hot) {
        rrpv[set][way] = INSERT_DEPTH_A; // near-MRU
    } else {
        // Conservative insertion to avoid pollution
        rrpv[set][way] = maxRRPV - 1;
    }
    stream_lock[set][way] = 0;
    hitcnt[set][way] = 0;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}