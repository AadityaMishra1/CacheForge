#include <vector>
#include <cstdint>
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
static inline bool is_demand(uint32_t type) { return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK); }

static constexpr uint8_t maxRRPV = 7;

// Tunables
static constexpr uint8_t STREAM_CONF_THRESH   = 2; // need 2 forward (+1/+2) steps
static constexpr uint8_t INSERT_DEPTH_A       = 2; // Hawkeye-friendly shallow insert
static constexpr uint8_t UNFRIENDLY_DEPTH     = 6; // deep insert for unfriendly/cold
static constexpr uint8_t PRESSURE_DEPTH       = 6; // set pressure threshold depth
static constexpr uint8_t PRESSURE_WAYS_THRESH = 12;// #ways at/above depth to call "under pressure"
static constexpr int8_t  MODEB_ENABLE_THRESH  = 3; // hysteresis enable
static constexpr int8_t  MODEB_DISABLE_THRESH = 1; // hysteresis disable
static constexpr uint32_t BANDIT_DECAY_PERIOD = 8192; // periodic gentle decay
static constexpr uint8_t HAWK_FRIENDLY_THRESH = 8; // SHCT >= 8 => friendly
static constexpr uint8_t PC_COLD_THRESH       = 2; // pc_cold >=2 => cold

// Per-line metadata (packed logically): rrpv(3), stream_lock(1), hitcnt(2)
// hitcnt encoding: 0/1 = number of demand hits seen (capped at 1)
//                  3->2->1 = prefetch quarantine countdown (two demand hits)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];

// Per-set selector (bandit with hysteresis)
static int8_t bandit_score[LLC_SETS];
static uint32_t bandit_decay_ctr = 0;

// Hawkeye-like PC usefulness (Mode A guide): 1K-entry 4-bit SHCT
static constexpr uint32_t SHCT_SIZE = 1024;
static uint8_t hawk_shct[SHCT_SIZE]; // 0..15
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// RunShield per-PC stream detector + coldness
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // low 12b of line addr
static uint8_t  pc_conf[PC_TBL_SIZE];      // 0..3 forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 0..3 coldness (higher=cold)
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// Utilities
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }

static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // invalid should be handled by caller
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age all by 1 (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // fallback: pick most aged
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

static inline bool set_under_pressure(uint32_t set) {
    uint32_t cnt = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= PRESSURE_DEPTH) cnt++;
    }
    return (cnt >= PRESSURE_WAYS_THRESH);
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    for (uint32_t i = 0; i < SHCT_SIZE; i++) hawk_shct[i] = 0;
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line[i] = 0;
        pc_conf[i] = 0;
        pc_cold[i] = 1; // slightly cold start
    }
    bandit_decay_ctr = 0;
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

    // Never change state for writebacks
    if (type == ACCESS_WRITEBACK) return;

    // Periodic gentle decay of selector (per-access, current set only)
    bandit_decay_ctr++;
    if ((bandit_decay_ctr % BANDIT_DECAY_PERIOD) == 0) {
        if (bandit_score[set] > 0) bandit_score[set]--;
        else if (bandit_score[set] < 0) bandit_score[set]++;
    }

    // Selector: Mode B only if clearly better
    bool use_modeB = (bandit_score[set] >= MODEB_ENABLE_THRESH);

    // Per-PC stream detection context (before training)
    uint32_t pidx = pc_index(PC);
    uint16_t prev_line = pc_last_line[pidx];
    uint16_t cur_line  = line12(paddr);
    int diff = (int)cur_line - (int)prev_line;
    bool forward_run = (diff == 1) || (diff == 2);
    uint8_t conf_before = pc_conf[pidx];

    // Hawkeye index
    uint32_t hidx = shct_idx(PC);

    // Hit path
    if (hit) {
        // Train Hawkeye and coldness on demand reuse
        if (is_demand(type)) {
            sat_inc_u4(hawk_shct[hidx]); // reuse observed
            if (pc_cold[pidx] > 0) pc_cold[pidx]--; // warmer
        }

        // Demand hit handling with multi-hit gating and prefetch quarantine
        if (is_demand(type)) {
            // Prefetch quarantine first: values 3->2->1 (no promotion while >=2)
            if (hitcnt[set][way] >= 2) {
                // still quarantined, consume one credit, no promotion
                hitcnt[set][way]--; // 3->2 or 2->1
            } else {
                // Not in quarantine
                if (stream_lock[set][way]) {
                    // Stream-locked line: escape on 2nd demand hit
                    if (hitcnt[set][way] == 0) {
                        hitcnt[set][way] = 1; // first demand hit, stay locked, no promote
                    } else { // hitcnt==1 -> second demand hit
                        stream_lock[set][way] = 0; // escape
                        // reward Mode B
                        sat_inc_i8(bandit_score[set]);
                        // full promotion on escape
                        rrpv[set][way] = 0;
                        // keep hitcnt at 1 as sticky "reused"
                    }
                } else {
                    // Non-stream line: multi-hit promotion
                    if (hitcnt[set][way] == 0) {
                        // first demand hit: minor bump only
                        if (rrpv[set][way] > 1) rrpv[set][way] = 1;
                        hitcnt[set][way] = 1;
                    } else {
                        // second+ demand hit: promote
                        // For cold PCs, approximate 3-hit gate by gradual promotion
                        bool pc_cold_now = (pc_cold[pidx] >= PC_COLD_THRESH);
                        if (pc_cold_now && rrpv[set][way] > 0) {
                            rrpv[set][way] = 1; // one more hit will reach MRU
                        } else {
                            rrpv[set][way] = 0; // MRU
                        }
                    }
                }
            }
        }

        // Train stream detector (after decisions)
        if (forward_run) sat_inc_u2(pc_conf[pidx]);
        else pc_conf[pidx] = 0;
        pc_last_line[pidx] = cur_line;
        return;
    }

    // Miss path (fill about to happen in 'way'): penalize dead quarantines
    if (stream_lock[set][way]) {
        // Evicting a still-locked stream line -> penalize Mode B
        sat_dec_i8(bandit_score[set]);
    }

    // Clear old metadata in victim way
    stream_lock[set][way] = 0;
    hitcnt[set][way] = 0;

    // Compute predictors for insertion
    bool stream_pred = use_modeB && (conf_before >= STREAM_CONF_THRESH) && forward_run;
    bool pressure = set_under_pressure(set);
    bool pc_is_cold = (pc_cold[pidx] >= PC_COLD_THRESH) || (hawk_shct[hidx] <= (HAWK_FRIENDLY_THRESH >> 1));

    // Decide insertion policy
    if (type == ACCESS_PREFETCH) {
        // Always tail-insert and quarantine for two demand hits
        rrpv[set][way] = maxRRPV;
        hitcnt[set][way] = 3; // two demand hits to escape quarantine: 3->2->1
        stream_lock[set][way] = 0;
    } else {
        if (stream_pred) {
            // Stream detected: hard-tail insert, lock; under pressure acts like bypass
            rrpv[set][way] = maxRRPV;
            stream_lock[set][way] = 1;
            hitcnt[set][way] = 0;
            (void)pressure; // tail insert handles pressure naturally
        } else {
            // Non-stream: Hawkeye-guided adaptive insertion
            uint8_t depth = (hawk_shct[hidx] >= HAWK_FRIENDLY_THRESH) ? INSERT_DEPTH_A : UNFRIENDLY_DEPTH;
            if (use_modeB && pc_is_cold) {
                depth = UNFRIENDLY_DEPTH; // keep cold PCs deep
            }
            if (depth > maxRRPV) depth = maxRRPV;
            rrpv[set][way] = depth;
            stream_lock[set][way] = 0;
            hitcnt[set][way] = 0;
        }

        // Hawkeye dead-on-arrival training and PC coldness
        sat_dec_u4(hawk_shct[hidx]);           // inserted => assume dead until proven
        sat_inc_u2(pc_cold[pidx]);             // colder until reuse
    }

    // Train stream detector (after decisions)
    if (forward_run) sat_inc_u2(pc_conf[pidx]);
    else pc_conf[pidx] = 0;
    pc_last_line[pidx] = cur_line;

    // Hysteresis safety: if score dipped low, ensure Mode B disables
    if (bandit_score[set] <= MODEB_DISABLE_THRESH && bandit_score[set] > 0) {
        // small nudge toward disable boundary to avoid oscillations
        bandit_score[set] = MODEB_DISABLE_THRESH;
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