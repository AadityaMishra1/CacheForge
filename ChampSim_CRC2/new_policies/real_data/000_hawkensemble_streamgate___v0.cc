#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

// HawkEnsemble-StreamGate+
// - Mode A: Hawkeye-like (friendly vs cold PC-based insertion over RRIP)
// - Mode B: StreamGate+ (PC stride/run detector, hard-tail stream insertion, multi-hit promotion, prefetch quarantine)
// - Per-set conservative bandit selector defaults to Mode A; only enables Mode B when it wins by margin
//
// Key tunables:
//   STREAM_CONF_THRESH = 2        // 2-step forward run
//   HITS_TO_PROMOTE    = 2        // multi-hit promotion
//   MODEB_MARGIN       = 2        // bandit margin to switch to Mode B
//   INSERT_DEPTH_A     = 2        // Mode A friendly insertion RRPV
//   INSERT_DEPTH_B     = 1        // Mode B friendly insertion RRPV
//   AGING_PERIOD       = 32       // per-set frequency aging period
//
// Notes:
// - No true bypass (CRC2 API constraint). Streaming "bypass" realized via RRPV=max insertion + no-promotion.
// - Never bypass WRITEBACK.
// - Prefetch quarantine: insert at tail (RRPV=7) and do not promote on first demand hit.
// - Multi-hit promotion: promote only on 2nd+ demand hit (demand only).
// - In-set 2-bit freq tiebreaker when multiple max-RRPV victims.
// - Sampled PC-signature stored only for sampled sets (for PC coldness training).

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
static constexpr uint8_t maxRRPV = 7;

// Sampling macro (64 sets), same structure as Hawkeye
static constexpr uint32_t LG_SETS = 11; // log2(LLC_SETS) for 2048
#define SAMPLED_SET(set) ( ((set) & 0x3F) == (((set) >> (LG_SETS - 6)) & 0x3F) )

// Tunables
static constexpr uint8_t STREAM_CONF_THRESH = 2;
static constexpr uint8_t HITS_TO_PROMOTE    = 2;
static constexpr int8_t  MODEB_MARGIN       = 2;
static constexpr uint8_t INSERT_DEPTH_A     = 2;
static constexpr uint8_t INSERT_DEPTH_B     = 1;
static constexpr uint8_t AGING_PERIOD       = 32;

// Per-line metadata (10 bits info-theoretically)
// rrpv:3, owner_mode:1, pref:1, stream_lock:1, hitcnt:2, freq2:2
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t owner_mode[LLC_SETS][LLC_WAYS];   // 0: Mode A, 1: Mode B (for bandit credit)
static uint8_t is_prefetch_line[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];  // do not promote until multi-hit proven
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];       // 2-bit: 0..3
static uint8_t freq2[LLC_SETS][LLC_WAYS];        // 2-bit: 0..3

// Sampled per-line PC signature (only used for sampled sets) -> 11 bits index space, stored in 16-bit container
static uint16_t sampled_pc_sig[LLC_SETS][LLC_WAYS];

// Per-set bandit selector and bookkeeping (conservative fallback to Mode A)
static int8_t scoreA[LLC_SETS];  // saturating [-31,31]
static int8_t scoreB[LLC_SETS];  // saturating [-31,31]
static uint8_t set_epoch[LLC_SETS];  // for freq aging

// Track last chosen victim per set to attribute eviction rewards in Update (safe and bounded)
static uint8_t last_victim_way[LLC_SETS];
static uint8_t last_victim_valid[LLC_SETS];

// PC tables (global)
static constexpr uint32_t PC_TBL_SIZE = 2048;
static uint32_t pc_last_line[PC_TBL_SIZE]; // lower bits of last line address
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_runlen[PC_TBL_SIZE];    // 2-bit recent forward-run length
static uint8_t  pc_cold[PC_TBL_SIZE];      // 2-bit expected-use (0..3)

// Hash/index helper
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)(pc) & (PC_TBL_SIZE - 1); }
static inline uint32_t line_addr(uint64_t paddr) { return (uint32_t)((paddr >> 6) & 0xFFFF); } // keep lower 16 bits

static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -31) x--; }

static inline bool is_demand(uint32_t type) {
    return (type != PREFETCH) && (type != WRITEBACK);
}

void InitReplacementState() {
    // Init per-line
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV; // start cold
            owner_mode[s][w] = 0;
            is_prefetch_line[s][w] = 0;
            stream_lock[s][w] = 0;
            hitcnt[s][w] = 0;
            freq2[s][w] = 0;
            sampled_pc_sig[s][w] = 0;
        }
        scoreA[s] = 0;
        scoreB[s] = -2; // bias toward Mode A initially
        set_epoch[s] = 0;
        last_victim_way[s] = 0;
        last_victim_valid[s] = 0;
    }
    // Init PC tables
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
    std::memset(pc_runlen, 0, sizeof(pc_runlen));
    std::memset(pc_cold, 1, sizeof(pc_cold)); // start slightly cold (1)
}

static inline uint32_t choose_max_rrpv_victim(uint32_t set, const BLOCK* current_set) {
    // Among maxRRPV lines, pick the one with smallest freq2; tie-breaker: prefer evicting prefetched lines
    uint32_t victim = 0;
    uint8_t best_freq = 255;
    uint8_t best_pref = 255;
    bool found = false;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) {
            uint8_t f = freq2[set][w];
            uint8_t p = is_prefetch_line[set][w];
            if (!found || (f < best_freq) || (f == best_freq && p > best_pref)) {
                found = true;
                best_freq = f;
                best_pref = p;
                victim = w;
            }
        }
    }
    if (found) return victim;
    // Should not reach; caller ensures at least one maxRRPV exists
    return 0;
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
        if (!current_set[w].valid) {
            last_victim_way[set] = w;
            last_victim_valid[set] = 0; // no eviction occurred
            return w;
        }
    }

    // Find victim via RRIP; age until a maxRRPV appears (bounded loop)
    for (uint32_t i = 0; i <= maxRRPV; i++) {
        // Check for any maxRRPV
        bool any_max = false;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) { any_max = true; break; }
        }
        if (any_max) {
            uint32_t v = choose_max_rrpv_victim(set, current_set);
            last_victim_way[set] = v;
            last_victim_valid[set] = current_set[v].valid ? 1 : 0;
            return v;
        }
        // Age all lines (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }

    // Fallback (should not happen)
    last_victim_way[set] = 0;
    last_victim_valid[set] = 1;
    return 0;
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
    // Ignore writebacks for policy updates
    if (type == WRITEBACK)
        return;

    // Per-set frequency aging
    set_epoch[set]++;
    if ((set_epoch[set] % AGING_PERIOD) == 0) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (freq2[set][w] > 0) freq2[set][w]--;
        }
    }

    // Update PC stride/run detector
    uint32_t laddr = line_addr(paddr);
    uint32_t pi = pc_index(PC);
    int32_t delta = (int32_t)laddr - (int32_t)pc_last_line[pi];
    bool forward_stride12 = (delta == 1) || (delta == 2);
    if (forward_stride12) {
        if (pc_runlen[pi] < 3) pc_runlen[pi]++;
        if (pc_conf[pi] < 3) pc_conf[pi]++; // accumulate confidence on forward run
    } else {
        if (pc_runlen[pi] > 0) pc_runlen[pi]--;
        if (pc_conf[pi] > 0) pc_conf[pi]--;
    }
    pc_last_line[pi] = laddr;

    bool stream_candidate = (pc_conf[pi] >= STREAM_CONF_THRESH);

    // Conservative per-set bandit: default Mode A; enable Mode B only when winning and only for streams
    bool prefer_modeB = false;
    int8_t advantage = (int8_t)(scoreB[set] - scoreA[set]);
    if (advantage >= MODEB_MARGIN && stream_candidate)
        prefer_modeB = true;
    uint8_t active_mode = prefer_modeB ? 1 : 0; // 0=A, 1=B

    // On hit: multi-hit promotion + quarantine rules
    if (hit) {
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            // Prefetch quarantine: do not promote on first demand hit
            if (is_prefetch_line[set][way] && hitcnt[set][way] < HITS_TO_PROMOTE) {
                if (rrpv[set][way] > 0) rrpv[set][way]--; // slight nudge, no full promote
            }
            // Stream lock: do not promote until multi-hit
            else if (stream_lock[set][way] && hitcnt[set][way] < HITS_TO_PROMOTE) {
                if (rrpv[set][way] > 0) rrpv[set][way]--; // slight nudge
            }
            else {
                // Confirmed reuse (>=2 hits): promote to MRU and clear quarantines
                if (hitcnt[set][way] >= HITS_TO_PROMOTE) {
                    rrpv[set][way] = 0;
                    is_prefetch_line[set][way] = 0;
                    stream_lock[set][way] = 0;
                } else {
                    if (rrpv[set][way] > 0) rrpv[set][way]--;
                }
            }
            // Frequency bump
            if (freq2[set][way] < 3) freq2[set][way]++;
            // Reward PCs that reach multi-hit
            if (hitcnt[set][way] == HITS_TO_PROMOTE && SAMPLED_SET(set)) {
                uint16_t sig = sampled_pc_sig[set][way];
                if (sig < PC_TBL_SIZE) sat_inc_u2(pc_cold[sig]);
            }
        }
        return;
    }

    // MISS path: train bandit on the evicted line (if any)
    if (last_victim_valid[set]) {
        uint8_t evw = last_victim_way[set];
        uint8_t ev_hits = hitcnt[set][evw];
        uint8_t ev_mode = owner_mode[set][evw]; // 0=A,1=B
        if (ev_hits >= HITS_TO_PROMOTE) {
            if (ev_mode) sat_inc_i8(scoreB[set]); else sat_inc_i8(scoreA[set]);
        } else {
            if (ev_mode) sat_dec_i8(scoreB[set]); else sat_dec_i8(scoreA[set]);
        }
        // Train PC coldness on sampled sets using victim's insertion PC
        if (SAMPLED_SET(set)) {
            uint16_t sig = sampled_pc_sig[set][evw];
            if (sig < PC_TBL_SIZE) {
                if (ev_hits >= HITS_TO_PROMOTE) sat_inc_u2(pc_cold[sig]);
                else sat_dec_u2(pc_cold[sig]);
            }
        }
    }
    last_victim_valid[set] = 0; // consumed

    // Decide insertion RRPV
    uint8_t new_rrpv = maxRRPV;
    uint8_t new_stream_lock = 0;
    uint8_t new_prefetch = (type == PREFETCH) ? 1 : 0;

    // PC coldness prediction (SHiP-like but very small); used by both modes
    uint8_t pc_expect = pc_cold[pi]; // 0..3
    bool pc_friendly = (pc_expect >= 2);

    if (type == PREFETCH) {
        // Prefetch quarantine at tail
        new_rrpv = maxRRPV;
        new_stream_lock = 1;
    } else {
        if (active_mode == 1) {
            // Mode B: StreamGate+
            if (stream_candidate) {
                // Aggressive stream quarantine: hard-tail insert, no promote until multi-hit
                new_rrpv = maxRRPV;
                new_stream_lock = 1;
            } else {
                // Non-stream under Mode B: be a bit more optimistic than Mode A if PC friendly
                new_rrpv = pc_friendly ? INSERT_DEPTH_B : maxRRPV;
                new_stream_lock = pc_friendly ? 0 : 1; // cold PCs gated
            }
        } else {
            // Mode A: Hawkeye-like friendly vs cold PC insertion
            new_rrpv = pc_friendly ? INSERT_DEPTH_A : maxRRPV;
            new_stream_lock = pc_friendly ? 0 : 1; // gate cold PCs
        }
    }

    // Install metadata for the filled line
    rrpv[set][way] = new_rrpv;
    owner_mode[set][way] = active_mode;
    is_prefetch_line[set][way] = new_prefetch;
    stream_lock[set][way] = new_stream_lock;
    hitcnt[set][way] = 0;
    // Slight initial freq credit for non-stream friendly lines
    freq2[set][way] = (new_rrpv < maxRRPV) ? 1 : 0;

    // Record PC signature for sampled sets to train coldness later
    if (SAMPLED_SET(set)) {
        sampled_pc_sig[set][way] = (uint16_t)pc_index(PC);
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}