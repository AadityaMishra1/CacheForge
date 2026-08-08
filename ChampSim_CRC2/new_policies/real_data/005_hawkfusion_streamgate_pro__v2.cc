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

// Tunables
static constexpr uint8_t STREAM_CONF_THRESH = 2;   // +1/+2 forward steps to mark stream
static constexpr uint8_t HITS_PROMOTE_NS    = 2;   // non-stream promote at 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 3;   // stream/quarantine promote at 3rd demand hit
static constexpr uint8_t HITS_PROMOTE_SHORT = 2;   // short-reuse PCs promote at 2nd demand hit
static constexpr uint8_t INSERT_DEPTH_A     = 2;   // Mode A warm insertion (hot PC)
static constexpr uint8_t INSERT_TAIL_NEAR   = 6;   // near tail
static constexpr int8_t  MODEB_THRESH       = 2;   // enable Mode B when selector_score >= 2
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

// Per-line metadata (packed conceptually): rrpv:3b, stream_lock:1b, hitcnt:2b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1 if quarantined (stream/prefetch)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 0..3 (saturating)

// For training on eviction: record chosen victim's prior state in GetVictimInSet
static uint8_t last_victim_quarantine[LLC_SETS]; // 0/1
static uint8_t last_victim_hitcnt[LLC_SETS];     // 0..3

// Selector (defaults to Mode A)
static int8_t selector_score = 0; // positive favors Mode B
static uint32_t fills_since_decay = 0;

// Mode A: tiny SHCT (Hawkeye-lite)
static constexpr uint32_t SHCT_SIZE = 512; // 512 entries x 4b
static uint8_t shct[SHCT_SIZE];            // 0..15
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// Mode B: PC stream + coldness + short-reuse hints
static constexpr uint32_t PC_TBL_SIZE = 512;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index
static uint8_t  pc_conf[PC_TBL_SIZE];      // 0..3 forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 0..15 "coldness"
static uint8_t  pc_sr_win[PC_TBL_SIZE];    // 0..7 short-reuse countdown
static uint8_t  pc_short[PC_TBL_SIZE];     // 0/1 short-reuse hint
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
        last_victim_quarantine[s] = 0;
        last_victim_hitcnt[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
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
    selector_score = 0;
    fills_since_decay = 0;
}

// RRIP victim selection with bounded aging to guarantee a victim
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Try direct hit at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // Age at most 8 steps
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback: pick the largest RRPV
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

    // If any invalid way exists, return it immediately, and clear last-victim training markers
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) {
            last_victim_quarantine[set] = 0;
            last_victim_hitcnt[set] = 0;
            return w;
        }
    }

    uint32_t victim = rrip_victim_and_age(set, current_set);
    // Record prior state for training at fill time
    last_victim_quarantine[set] = stream_lock[set][victim] ? 1 : 0;
    last_victim_hitcnt[set] = (hitcnt[set][victim] & 0x3u);
    return victim;
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

    const uint32_t pi = pc_index(PC);
    const uint32_t hi = shct_idx(PC);
    const uint16_t curL = line12(paddr);

    // Update per-PC stream run detector on demand accesses
    if (is_demand(type)) {
        uint16_t lastL = pc_last_line[pi];
        bool fwd = (curL == (uint16_t)(lastL + 1)) || (curL == (uint16_t)(lastL + 2));
        if (fwd) sat_inc_u2(pc_conf[pi]);
        else     sat_dec_u2(pc_conf[pi]);
        pc_last_line[pi] = curL;
    }

    // Decide mode via leaders + selector (defaults to Mode A)
    bool use_modeB = false;
    if (LEADER_B(set)) use_modeB = true;
    else if (LEADER_A(set)) use_modeB = false;
    else use_modeB = (selector_score >= MODEB_THRESH);

    // Promotion on hits (multi-hit gating)
    if (hit) {
        // Demand hit handling
        if (is_demand(type)) {
            // SHiP-like: reward useful PC, cool down coldness
            sat_inc_u4(shct[hi]);
            sat_dec_u4(pc_cold[pi]);

            // Short-reuse window update: a demand hit consumes one step
            if (pc_sr_win[pi] > 0) {
                pc_short[pi] = 1;
                pc_sr_win[pi]--;
                if (pc_sr_win[pi] == 0) pc_short[pi] = 0;
            }

            // Line-level multi-hit gate
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;

            if (stream_lock[set][way]) {
                uint8_t need = pc_short[pi] ? HITS_PROMOTE_SHORT : HITS_PROMOTE_STR;
                if (hitcnt[set][way] >= need) {
                    // Graduate from quarantine
                    rrpv[set][way] = 0;
                    stream_lock[set][way] = 0;
                    // Reward Mode B only on leader-B sets
                    if (LEADER_B(set)) sat_inc_i8(selector_score);
                } else if (STREAM_DEMOTE_ON_TOUCH) {
                    // Keep quarantined lines cold on subthreshold touches
                    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv[set][way] = 0; // promote to MRU
                }
            }
        } else {
            // Prefetch hit: do not promote aggressively; mild aging protection
            if (stream_lock[set][way]) {
                if (rrpv[set][way] > 0) rrpv[set][way]--; // gentle move toward reuse
            }
        }
        return;
    }

    // Miss/fill path below
    // Periodic decay toward Mode A
    fills_since_decay++;
    if ((fills_since_decay & 0xFFFu) == 0) { // every 4096 fills
        sat_toward_zero(selector_score);
    }

    // Train on eviction outcome observed at GetVictimInSet (leaders only)
    if (LEADER_B(set)) {
        if (last_victim_quarantine[set] && (last_victim_hitcnt[set] < HITS_PROMOTE_STR)) {
            // quarantined line died without graduating -> penalize Mode B
            sat_dec_i8(selector_score);
        }
    }

    // SHiP-like: penalize noisy PCs on demand misses
    if (is_demand(type)) {
        sat_dec_u4(shct[hi]); // demand miss is a negative signal
        sat_inc_u4(pc_cold[pi]);
    }

    // Decide insertion policy
    uint8_t ins_rrpv = INSERT_TAIL_NEAR;
    uint8_t ins_lock = 0;
    uint8_t ins_hitc = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; keep low priority
        ins_rrpv = INSERT_TAIL_NEAR;
        ins_lock = 0;
    } else if (type == ACCESS_PREFETCH) {
        ins_rrpv = maxRRPV;
        ins_lock = 1; // quarantined
    } else {
        // Demand fill
        bool streaming = (pc_conf[pi] >= STREAM_CONF_THRESH);
        if (use_modeB) {
            if (streaming) {
                // Hard-tail quarantine (bypass-equivalent)
                ins_rrpv = maxRRPV;
                ins_lock = 1;
            } else if (pc_cold[pi] >= SHCT_HOT_THRESH) {
                ins_rrpv = INSERT_TAIL_NEAR;
                ins_lock = 0;
            } else {
                ins_rrpv = INSERT_DEPTH_A;
                ins_lock = 0;
            }
        } else {
            // Mode A (Hawkeye-lite)
            if (shct[hi] >= SHCT_HOT_THRESH) ins_rrpv = INSERT_DEPTH_A;
            else                             ins_rrpv = INSERT_TAIL_NEAR;
            ins_lock = 0;
        }
    }

    // Apply insertion
    rrpv[set][way] = ins_rrpv;
    stream_lock[set][way] = ins_lock;
    hitcnt[set][way] = ins_hitc;

    // Start/refresh short-reuse window only for demand fills
    if (is_demand(type)) {
        pc_sr_win[pi] = SR_WINDOW_INIT;
        pc_short[pi] = 0; // will be set on a quick subsequent hit
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}