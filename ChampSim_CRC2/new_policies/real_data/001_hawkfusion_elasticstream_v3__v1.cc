#include <vector>
#include <cstdint>
#include <iostream>
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

static constexpr uint8_t maxRRPV = 7;

// Tunables (retuned for stronger streaming protection and safer elastic rescue)
static constexpr uint8_t STREAM_CONF_THRESH = 2;  // +1/+2 forward steps to tag stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2;  // non-stream promotion on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3;  // stream-tagged base promotion on 3rd demand hit (elastic may lower to 2)
static constexpr uint8_t INSERT_DEPTH_A     = 2;  // Mode A hot insert depth (near MRU)
static constexpr int8_t  MODEB_THRESH       = 2;  // enable Mode B when bandit_score >= 2
static constexpr uint8_t SR_WINDOW_INIT     = 7;  // short-reuse window (per-PC countdown)
static constexpr uint8_t SHCT_HOT_THRESH    = 8;  // hot threshold for SHCT/pc_cold

// Per-line metadata (packed logically): rrpv(3b), stream_lock(1b), hitcnt(2b: 3 used as prefetch sentinel)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];

// Per-set bandit selector (defaults to Mode A, small signed score saturated)
static int8_t bandit_score[LLC_SETS];

// Mode A: Hawkeye-like SHCT (1K x 4b)
static constexpr uint32_t SHCT_SIZE = 1024;
static uint8_t hawk_shct[SHCT_SIZE];
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// Mode B: PC stream detector (+1/+2), PC coldness, and elastic short-reuse hints
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index of last access by PC
static uint16_t pc_sr_line[PC_TBL_SIZE];   // 12-bit line index tagged at last stream fill for short-reuse
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 4-bit PC coldness (0=cold..15=hot)
static uint8_t  pc_sr_win[PC_TBL_SIZE];    // 3-bit short-reuse window (counts down on each access by PC)
static uint8_t  pc_short[PC_TBL_SIZE];     // 1-bit sticky short-reuse hint
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// Saturating helpers
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;  // tail
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
    // If any invalid way exists, it should be handled by caller
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age by 1 (saturating)
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
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection
    uint32_t v = rrip_victim_and_age(set, current_set);
    // Update bandit with the fate of quarantined (stream) lines
    if (stream_lock[set][v]) {
        if (hitcnt[set][v] >= 2) sat_inc_i8(bandit_score[set]);
        else                      sat_dec_i8(bandit_score[set]);
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

    // Update per-PC forward-run detector and short-reuse window
    int16_t delta = (int16_t)l12 - (int16_t)pc_last_line[pidx];
    bool fwd = (delta == 1) || (delta == 2);
    if (fwd) sat_inc_u2(pc_conf[pidx]); else sat_dec_u2(pc_conf[pidx]);
    pc_last_line[pidx] = l12;
    if (pc_sr_win[pidx] > 0) pc_sr_win[pidx]--;

    // Hawkeye-like SHCT and PC coldness updates
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

    // Prefetch quarantine management: sentinel '3' means "prefilled by prefetch"
    if (hit) {
        // First demand hit on a prefetched line: consume quarantine, no promotion
        if (is_demand(type) && hitcnt[set][way] == 3) {
            hitcnt[set][way] = 0;
            return;
        }

        // Multi-hit gated promotion
        if (stream_lock[set][way]) {
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++; // demand-only
                // Elastic promotion: allow on 2nd hit if short-reuse is predicted/observed
                bool short_reuse = (pc_short[pidx] != 0) || ((pc_sr_win[pidx] > 0) && (pc_sr_line[pidx] == l12));
                uint8_t need = short_reuse ? 2 : HITS_PROMOTE_STR;
                if (hitcnt[set][way] >= need) {
                    rrpv[set][way] = 0;      // MRU
                    stream_lock[set][way] = 0;
                    if (short_reuse) pc_short[pidx] = 1; // reinforce
                } else {
                    if (rrpv[set][way] > 0) rrpv[set][way]--; // gentle nudge
                }
            }
        } else {
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv[set][way] = 0; // MRU
                }
            }
        }
        return;
    }

    // Miss path: choose insertion policy based on mode and predictors
    bool is_pf = (type == ACCESS_PREFETCH);
    bool use_modeB = (bandit_score[set] >= MODEB_THRESH);
    bool stream_tag = (pc_conf[pidx] >= STREAM_CONF_THRESH);

    if (is_pf) {
        // Always hard-tail quarantine for prefetch fills
        rrpv[set][way] = maxRRPV;
        hitcnt[set][way] = 3; // prefetch sentinel
        stream_lock[set][way] = stream_tag ? 1 : 0;
        if (stream_tag) {
            pc_sr_line[pidx] = l12;
            pc_sr_win[pidx]  = SR_WINDOW_INIT;
        }
        return;
    }

    // Demand/RFO insertion
    if (use_modeB && stream_tag) {
        // Stream quarantine at tail; elastic short-reuse window armed
        rrpv[set][way] = maxRRPV;
        stream_lock[set][way] = 1;
        hitcnt[set][way] = 0;
        pc_sr_line[pidx] = l12;
        pc_sr_win[pidx]  = SR_WINDOW_INIT;
        // Note: we do not truly bypass insertion in CRC2; hard-tail quarantine approximates bypass safely
    } else {
        // Mode A Hawkeye-like insertion using SHCT + PC coldness
        bool hot = (hawk_shct[shct_idx(PC)] >= SHCT_HOT_THRESH) || (pc_cold[pidx] >= SHCT_HOT_THRESH);
        rrpv[set][way] = hot ? INSERT_DEPTH_A : maxRRPV;
        stream_lock[set][way] = 0;
        hitcnt[set][way] = 0;
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