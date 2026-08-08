#include <cstdint>
#include <cstring>
#include <map>
#include "../inc/champsim_crc2.h"

// HawkFusion-RunShield: Hawkeye-like Mode A + RunShield Mode B with per-set fallback to A
// Key tunables (safe to tweak):
//   STREAM_CONF_THRESH = 2   // PC stride (+1/+2) confidence to quarantine streams
//   INSERT_DEPTH_A     = 2   // Mode A friendly insertion depth (lower is closer to MRU)
//   HITS_PROMOTE_NS    = 2   // Promote non-stream lines on 2nd+ demand hit
//   HITS_PROMOTE_STR   = 3   // Promote stream-tagged lines on 3rd demand hit
//   MODEB_MARGIN       = 2   // Enable Mode B if bandit score >= 2
//
// Notes:
// - No true bypass: streaming "bypass" is realized by inserting at RRPV=max and not promoting until reuse is proven.
// - WRITEBACK never changes replacement state.
// - Prefetches always insert at tail (RRPV=7) and first demand hit does not promote (encoded via hitcnt=3 sentinel).
// - Bandit credit/penalty attributed only for stream-quarantined (Mode B) lines on eviction (fill over a victim).

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
static constexpr uint8_t maxRRPV = 7;

// Access types (match ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) { return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK); }

// ---------------- Tunables ----------------
static constexpr uint8_t STREAM_CONF_THRESH = 2;
static constexpr uint8_t INSERT_DEPTH_A     = 2;
static constexpr uint8_t HITS_PROMOTE_NS    = 2;
static constexpr uint8_t HITS_PROMOTE_STR   = 3;
static constexpr int8_t  MODEB_MARGIN       = 2;

// ---------------- Per-line metadata (packed logically; stored as bytes) ----------------
// rrpv: 3 bits (stored in uint8_t), stream_lock:1 bit, hitcnt:2 bits (0..3; 3=prefetch sentinel)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1 if quarantined as stream by Mode B
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 0..3 (3 marks prefetched line until its first demand hit)

// ---------------- Per-set selector (bandit; conservative fallback to Mode A) ----------------
// score per set in [-31..31]; enable Mode B only if score >= MODEB_MARGIN
static int8_t bandit_score[LLC_SETS];

// ---------------- Lightweight Hawkeye-like PC usefulness (Mode A guide) ----------------
// 1K-entry 4-bit SHCT; increment on demand reuse, decrement when dead-ish insertions occur
static constexpr uint32_t SHCT_SIZE = 1024;
static uint8_t hawk_shct[SHCT_SIZE]; // 0..15
static inline uint32_t shct_idx(uint64_t pc) { return (uint32_t)pc & (SHCT_SIZE - 1); }

// ---------------- RunShield PC stream detector and coldness ----------------
static constexpr uint32_t PC_TBL_SIZE = 1024;
static uint16_t pc_last_line[PC_TBL_SIZE]; // store lower 12b of line addr; reported as 12b info-theoretically
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence
static uint8_t  pc_cold[PC_TBL_SIZE];      // 2-bit expected-use (Mode B non-stream insertion guide)
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

// ---------------- Utilities ----------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t& x)  { if (x < 31) x++; }
static inline void sat_dec_i8(int8_t& x)  { if (x > -31) x--; }

// ---------------- Init ----------------
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
}

// Ensure a maxRRPV victim exists by aging; tie-breaker is first found
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // If invalid exists, it should be returned by GetVictimInSet prior to calling this helper.
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // age all ways by 1 (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // fallback: pick the most aged
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
    // Selector: enable Mode B only if it has clear advantage on this set
    bool use_modeB = (bandit_score[set] >= MODEB_MARGIN);

    // Victim policy: both modes use RRIP with aging; B's extra tie-breaking is implicit in insertion gates
    (void)cpu; (void)PC; (void)paddr; (void)type;
    return rrip_victim_and_age(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Ignore writebacks
    if (type == ACCESS_WRITEBACK) return;

    // Per-PC stream detection (forward run of +1 or +2 cache lines)
    const uint32_t pidx = pc_index(PC);
    const uint16_t cline = line12(paddr);
    uint16_t last = pc_last_line[pidx];
    uint16_t delta = (uint16_t)((cline - last) & 0x0FFFu);
    bool forward12 = (delta == 1) || (delta == 2);
    if (forward12) sat_inc_u2(pc_conf[pidx]); else sat_dec_u2(pc_conf[pidx]);
    pc_last_line[pidx] = cline;

    // Selector state
    bool use_modeB = (bandit_score[set] >= MODEB_MARGIN);

    // On hit: gated promotion
    if (hit) {
        if (type != ACCESS_PREFETCH) {
            // track reuse for PC usefulness
            sat_inc_u4(hawk_shct[shct_idx(PC)]);
            // prefetch sentinel (3) converts to 1 on first demand hit, without promotion
            if (hitcnt[set][way] == 3) {
                hitcnt[set][way] = 1; // consumed prefetch credit, no promotion yet
            } else if (hitcnt[set][way] < 3) {
                hitcnt[set][way]++;
            }
            // Promotion gating
            if (stream_lock[set][way]) {
                if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
                    rrpv[set][way] = 0;
                    stream_lock[set][way] = 0; // unlock after sufficient reuse
                }
            } else {
                if (hitcnt[set][way] >= HITS_PROMOTE_NS) {
                    rrpv[set][way] = 0;
                }
            }
            // PC coldness warms on demand hit
            sat_inc_u2(pc_cold[pidx]);
        }
        return;
    }

    // Miss: about to fill at (set, way). Attribute bandit credit to the evicted line's prior state.
    // If the evicted line was a Mode B stream quarantine (stream_lock==1) and delivered reuse (>=2 hits), reward B; else penalize.
    uint8_t prev_stream = stream_lock[set][way];
    uint8_t prev_hits   = hitcnt[set][way] & 0x3;
    if (prev_stream) {
        if (prev_hits >= HITS_PROMOTE_NS) sat_inc_i8(bandit_score[set]);
        else                               sat_dec_i8(bandit_score[set]);
    }

    // Reset per-line metadata for the new line
    stream_lock[set][way] = 0;
    hitcnt[set][way] = 0;

    // Insertion policy
    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine: tail insert, mark sentinel so first demand hit will not promote
        rrpv[set][way] = maxRRPV;
        hitcnt[set][way] = 3;
        // If PC looks streaming, lock as stream (never promote until 3 demand hits)
        if (pc_conf[pidx] >= STREAM_CONF_THRESH) stream_lock[set][way] = 1;
        // Prefetch generally cools PC coldness slightly unless already warm
        if (pc_cold[pidx] > 0) pc_cold[pidx]--;
        return;
    }

    // Demand fill
    bool is_stream_pc = (pc_conf[pidx] >= STREAM_CONF_THRESH);
    if (use_modeB && is_stream_pc) {
        // RunShield: aggressive stream quarantine
        rrpv[set][way] = maxRRPV;
        stream_lock[set][way] = 1;
        hitcnt[set][way] = 0;
        if (pc_cold[pidx] > 0) pc_cold[pidx]--; // penalize coldness on detected stream
    } else if (!use_modeB) {
        // Mode A (Hawkeye-guided via lightweight SHCT)
        bool friendly = (hawk_shct[shct_idx(PC)] >= 2);
        rrpv[set][way] = friendly ? INSERT_DEPTH_A : (maxRRPV - 1);
        // Modest exploration: slightly cool on averse insert
        if (!friendly && pc_cold[pidx] > 0) pc_cold[pidx]--;
    } else {
        // Mode B for non-streams: PC coldness guides insertion depth
        if (pc_cold[pidx] <= 1) {
            rrpv[set][way] = (maxRRPV - 1); // near-tail for cold PCs
        } else {
            rrpv[set][way] = 1; // near-MRU for warm PCs
        }
        // On demand insert, do not set prefetch sentinel
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}