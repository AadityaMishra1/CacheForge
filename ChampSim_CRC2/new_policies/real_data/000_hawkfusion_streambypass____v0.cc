#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

// HawkFusion-StreamBypass++
// - Mode A: Hawkeye-like PC usefulness (SHCT-style) driving adaptive insertion.
// - Mode B: forward-run PC stream detector; quarantine streams (RRPV=max), multi-hit gated promotion.
// - Per-set bandit: defaults to Mode A; only enables Mode B if previous quarantined lines (in this set)
//   achieved 2+ demand hits more often than not. No true bypass (we always pick a victim 0..15).

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
static constexpr uint8_t maxRRPV = 7;

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) { return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK); }

// ---------------- Tunables ----------------
static constexpr uint8_t STREAM_CONF_THRESH = 2; // need 2 forward steps (+1/+2) to tag as stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2; // non-stream: promote on 2nd+ demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3; // stream-tagged: promote on 3rd demand hit
static constexpr uint8_t INSERT_DEPTH_A     = 2; // Mode A: predicted-hot insert RRPV=2 (near MRU)
static constexpr int8_t  MODEB_MARGIN       = 2; // enable Mode B if bandit_score >= 2

// ---------------- Per-line metadata (packed logically) ----------------
// rrpv: 3b, stream_lock:1b (1 when quarantined as stream), hitcnt:2b (0..3; 3 is prefetch sentinel)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector ----------------
static int8_t bandit_score[LLC_SETS]; // signed score in [-31..31]

// ---------------- Mode A: Hawkeye-like PC usefulness (SHCT-style) ----------------
static constexpr uint32_t SHCT_SIZE = 1024; // 1K-entry usefulness table
static uint8_t hawk_shct[SHCT_SIZE];        // 4-bit counters [0..15]
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// ---------------- Mode B: PC stream detector + PC coldness ----------------
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line ID (lower bits of line address)
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 4-bit expected-use (0=cold .. 15=hot)
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// ---------------- Utilities ----------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }

// Global light epoch for gentle decay without extra storage
static uint64_t global_epoch = 0;

// ---------------- Init ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
        }
    }
    std::memset(hawk_shct, 0, sizeof(hawk_shct));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
    std::memset(pc_cold, 1, sizeof(pc_cold)); // slightly cold bias
    global_epoch = 0;
}

// Ensure a maxRRPV victim exists by aging; bounded passes guarantee termination
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // If any invalid way exists, it should be returned by GetVictimInSet before calling this helper.
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all ways by 1 (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: pick the most aged
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    (void)cpu; (void)PC; (void)paddr; (void)type;
    // RRIP victim selection (common to both modes)
    return rrip_victim_and_age(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    // Ignore writebacks for replacement state
    if (type == ACCESS_WRITEBACK) return;

    // Track prefetch sentinel: first demand hit on a prefetched line does not promote
    if (type == ACCESS_PREFETCH) {
        if (!hit) hitcnt[set][way] = 3; // sentinel state until first demand hit
    }

    // Mode B stream detection: update per-PC run detector on every access (demand/pf)
    const uint32_t pidx = pc_index(PC);
    const uint16_t curr_line12 = line12(paddr);
    int16_t delta = (int16_t)curr_line12 - (int16_t)pc_last_line[pidx];
    bool fwd = (delta == 1) || (delta == 2);
    if (fwd) sat_inc_u2(pc_conf[pidx]); else sat_dec_u2(pc_conf[pidx]);
    pc_last_line[pidx] = curr_line12;
    bool will_stream = (pc_conf[pidx] >= STREAM_CONF_THRESH);

    // Selector: enable Mode B on this set if confident; slow global decay to avoid stickiness
    bool use_modeB = (bandit_score[set] >= MODEB_MARGIN);
    if ((global_epoch & 0x3FFFu) == 0) { // occasional decay toward 0
        if (bandit_score[set] > 0) sat_dec_i8(bandit_score[set]);
        else if (bandit_score[set] < 0) sat_inc_i8(bandit_score[set]);
    }

    // On hit: multi-hit gated promotion
    if (hit) {
        if (is_demand(type)) {
            // Clear prefetch sentinel on first demand hit
            if (hitcnt[set][way] == 3) hitcnt[set][way] = 0;
            // Count this demand hit (capped)
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (hitcnt[set][way] >= need) {
                rrpv[set][way] = 0; // promote to MRU
                // Once a quarantined stream proves reuse, clear the lock
                if (stream_lock[set][way] && hitcnt[set][way] >= HITS_PROMOTE_STR)
                    stream_lock[set][way] = 0;
            } else {
                // Gentle partial promotion for non-streams to help short reuse
                if (!stream_lock[set][way] && rrpv[set][way] > 0) rrpv[set][way]--;
            }
            // Train Mode A usefulness on real (confirmed) reuse
            if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                uint32_t hx = shct_idx(PC);
                sat_inc_u4(hawk_shct[hx]);
                sat_inc_u4(pc_cold[pidx]);
            }
        }
        return;
    }

    // We are inserting a new line (miss path): attribute credit/penalty to the evicted line (residing at [set][way])
    // Only consider stream-quarantined lines for Mode B bandit accounting
    if (stream_lock[set][way]) {
        if (hitcnt[set][way] >= HITS_PROMOTE_NS) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
    }
    // Train coldness/usefulness for Mode A: dead (0/1-hit) lines are cold, multi-hit lines are hot
    {
        // We do not know the evicted PC here; approximate via the current PC index to keep storage bounded
        if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
            if (is_demand(type)) sat_inc_u4(hawk_shct[shct_idx(PC)]);
            sat_inc_u4(pc_cold[pidx]);
        } else {
            if (is_demand(type)) sat_dec_u4(hawk_shct[shct_idx(PC)]);
            sat_dec_u4(pc_cold[pidx]);
        }
    }

    // Decide insertion policy
    uint8_t new_rrpv = maxRRPV;
    uint8_t new_stream_lock = 0;
    uint8_t new_hitcnt = (type == ACCESS_PREFETCH) ? 3 : 0;

    if (use_modeB && will_stream) {
        // Stream quarantine: hard-tail insert, no promotion until 3rd demand hit
        new_rrpv = maxRRPV;
        new_stream_lock = 1;
        // prefetches already tagged with sentinel
    } else {
        // Mode A (fallback) Hawkeye-like usefulness
        uint8_t score = hawk_shct[shct_idx(PC)];
        // Bias by per-PC coldness (Mode B's PC cold table) for irregular workloads
        uint8_t cold = pc_cold[pidx];
        bool predicted_hot = (score >= 8) || (cold >= 10);
        if (type == ACCESS_PREFETCH) {
            // Prefetches always at tail
            new_rrpv = maxRRPV;
        } else if (predicted_hot) {
            new_rrpv = INSERT_DEPTH_A; // near-MRU insertion for hot PCs
        } else {
            new_rrpv = maxRRPV; // cold PCs at tail
        }
    }

    // Install the new line metadata
    rrpv[set][way] = new_rrpv;
    stream_lock[set][way] = new_stream_lock;
    hitcnt[set][way] = new_hitcnt;

    global_epoch++;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}