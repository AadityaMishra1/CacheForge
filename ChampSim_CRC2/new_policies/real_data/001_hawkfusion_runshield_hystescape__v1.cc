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
static inline bool is_demand(uint32_t type) { return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK); }

static constexpr uint8_t maxRRPV = 7;

// ---------------- Tunables (retuned) ----------------
static constexpr uint8_t STREAM_CONF_THRESH   = 2; // need 2 forward (+1/+2) steps
static constexpr uint8_t INSERT_DEPTH_A       = 2; // Hawkeye-friendly insertion depth
static constexpr uint8_t PRESSURE_DEPTH       = 6; // set pressure: many ways at/above this
static constexpr uint8_t PRESSURE_WAYS_THRESH = 12;// #ways at/above depth to call "under pressure"
static constexpr int8_t  MODEB_ENABLE_THRESH  = 3; // hysteresis enable
static constexpr int8_t  MODEB_DISABLE_THRESH = 1; // hysteresis disable (we use >=3 to enable in practice)
static constexpr uint32_t BANDIT_DECAY_PERIOD = 8192; // periodic gentle decay to avoid lock-in

// ---------------- Per-line metadata (packed logically) ----------------
// rrpv: 3 bits, stream_lock:1 bit, hitcnt:2 bits (0/1=hit count; 2/3=prefetch quarantine countdown: 3->2->1)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];

// ---------------- Per-set selector (bandit with hysteresis) ----------------
static int8_t bandit_score[LLC_SETS];
static uint32_t bandit_decay_ctr = 0;

// ---------------- Hawkeye-like PC usefulness (Mode A guide) ----------------
static constexpr uint32_t SHCT_SIZE = 1024;
static uint8_t hawk_shct[SHCT_SIZE]; // 4-bit saturating
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// ---------------- RunShield per-PC stream detector + coldness ----------------
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // store low 12 bits of line address
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 2-bit coldness (higher=cold)
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// ---------------- Utilities ----------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }

static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // invalid return should be handled by caller before using this
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age all ways by 1 (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // fallback: choose most aged
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

// ---------------- API ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
        }
        bandit_score[s] = 0;
    }
    std::memset(hawk_shct, 0, sizeof(hawk_shct));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
    std::memset(pc_cold, 1, sizeof(pc_cold)); // slightly cold start
    bandit_decay_ctr = 0;
}

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

    // Hysteresis-enabled selector (Mode B only if clearly better)
    bool use_modeB = (bandit_score[set] >= MODEB_ENABLE_THRESH);

    // Per-PC stream detection (use confidence before update for decisions)
    uint32_t pidx = pc_index(PC);
    uint16_t prev_line = pc_last_line[pidx];
    uint16_t cur_line  = line12(paddr);
    int diff = (int)cur_line - (int)prev_line;
    bool forward_run = (diff == 1) || (diff == 2);
    uint8_t conf_before = pc_conf[pidx];

    // Train Hawkeye/Coldness on this access
    uint32_t hidx = shct_idx(PC);
    if (hit && is_demand(type)) {
        sat_inc_u4(hawk_shct[hidx]);  // reuse observed
        sat_dec_u2(pc_cold[pidx]);    // warmer
    }

    if (hit) {
        // Demand hit handling with multi-hit gate and prefetch quarantine
        if (is_demand(type)) {
            // Prefetch quarantine first: values 3->2->1, no promotion
            if (hitcnt[set][way] >= 2) {
                hitcnt[set][way]--; // consume quarantine token
                if (rrpv[set][way] > 0) rrpv[set][way]--; // slight aging toward MRU
            } else {
                // Counted hits: promote only on 2nd counted hit
                if (hitcnt[set][way] == 0) {
                    hitcnt[set][way] = 1; // first counted hit, no promotion
                    if (rrpv[set][way] > 0) rrpv[set][way]--;
                } else { // hitcnt == 1 -> 2nd counted hit
                    // If stream-locked, escape quarantine now (reward Mode B) and promote
                    if (stream_lock[set][way]) {
                        stream_lock[set][way] = 0; // 2-hit escape
                        sat_inc_i8(bandit_score[set]); // reward B for correctly quarantining a reusable line
                    }
                    rrpv[set][way] = 0;   // promote to MRU
                    hitcnt[set][way] = 1; // keep as "seen reuse"
                }
            }
        } else {
            // Prefetch/RFO hits: do not promote to MRU; slight aging toward MRU
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
    } else {
        // Miss path: may bypass (never for WB), else insert
        // Penalize Mode B if we evict a still-stream-locked line
        if (stream_lock[set][way]) {
            sat_dec_i8(bandit_score[set]);
        }

        bool do_bypass = false;
        bool demand = is_demand(type);

        // Mode B stream handling
        if (use_modeB && (conf_before >= STREAM_CONF_THRESH)) {
            // Cold-PC-aware bypass under set pressure (LOAD/RFO/PREFETCH only)
            if ((pc_cold[pidx] >= 2) && set_under_pressure(set)) {
                do_bypass = true;
            }
        }

        if (do_bypass) {
            // SHiP-style: treat as cold (on insertion opportunity) to push PC colder
            if (demand || type == ACCESS_PREFETCH) {
                sat_dec_u4(hawk_shct[hidx]); // speculative dead on "insertion" opportunity
                sat_inc_u2(pc_cold[pidx]);
            }
            // No state changes to the chosen way => true bypass
        } else {
            // Decide insertion policy
            uint8_t ins_rrpv = maxRRPV; // default tail
            uint8_t ins_hitc = 0;
            uint8_t ins_stream = 0;

            if (type == ACCESS_PREFETCH) {
                // Prefetch always tail; quarantine for two hits
                ins_rrpv = maxRRPV;
                ins_hitc = 3; // two-hit quarantine countdown: 3->2->1
                // Stream-tag prefetch too if confident
                if (conf_before >= STREAM_CONF_THRESH) ins_stream = 1;
            } else {
                if (use_modeB && (conf_before >= STREAM_CONF_THRESH)) {
                    // Quarantine streams hard-tail; allow 2-hit escape later
                    ins_rrpv = maxRRPV;
                    ins_stream = 1;
                } else {
                    // Mode A (Hawkeye-like) adaptive insertion
                    bool friendly = (hawk_shct[hidx] >= 8);
                    ins_rrpv = friendly ? INSERT_DEPTH_A : (uint8_t)6;
                    ins_stream = 0;
                }
                ins_hitc = 0; // demand insert starts with 0 counted hits
            }

            // Train SHCT/coldness on insertion (SHiP: dec on insert)
            if (is_demand(type) || type == ACCESS_PREFETCH) {
                sat_dec_u4(hawk_shct[hidx]); // assume dead until reuse
                sat_inc_u2(pc_cold[pidx]);   // a bit colder until proven otherwise
            }

            // Commit insertion metadata
            rrpv[set][way] = ins_rrpv;
            hitcnt[set][way] = ins_hitc;
            stream_lock[set][way] = ins_stream;
        }
    }

    // Update per-PC stream detector state after using the old confidence
    if (forward_run) sat_inc_u2(pc_conf[pidx]);
    else             sat_dec_u2(pc_conf[pidx]);
    pc_last_line[pidx] = cur_line;

    // Periodic bandit decay toward 0 (stability/hysteresis)
    bandit_decay_ctr++;
    if ((bandit_decay_ctr & (BANDIT_DECAY_PERIOD - 1)) == 0) {
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (bandit_score[s] > 0) bandit_score[s]--;
            else if (bandit_score[s] < 0) bandit_score[s]++;
        }
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}