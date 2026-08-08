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
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD) || (t==ACCESS_RFO); }

// ---------------- RRIP core (shared by both modes) ----------------
static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// Per-line compact metadata (conceptually bit-packed: rrpv:3b, hitcnt:2b, stream_lock:1b)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3 demand-hit counter
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];// 0/1: quarantined stream/pref line

// ---------------- Leader-set sampling & selector ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set){
    // 64 sampled sets: low 6 bits match the next 6 bits
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); }

// Global conservative selector (defaults to Mode A)
static constexpr uint8_t GSEL_MAX   = 31;  // 5-bit
static constexpr uint8_t GSEL_THRES = 24;  // followers enable Mode B only if >= THRES
static uint8_t GSEL = 0;
static inline void gsel_inc(uint8_t step=1){ if (GSEL + step > GSEL_MAX) GSEL = GSEL_MAX; else GSEL += step; }
static inline void gsel_dec(uint8_t step=1){ if (GSEL > step) GSEL -= step; else GSEL = 0; }
static inline bool use_modeB(uint32_t set){
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}

// ---------------- Mode A: Hawkeye-like friendly/averse SHCT -------
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS) // 2048 entries
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit counters
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit counters

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return (uint16_t)(x & 0x0FFFu); // 12b
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x){ if (x > 0) x--; }

// Leader-only per-line tags for Mode A training
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // 0/1: re-referenced while resident
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // 0/1: inserted by prefetch

// ---------------- Mode B: StreamQuant + TinyLFU -------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

// Per-PC state (conceptually packed: last_line10:10b, stream_conf:2b, use4:4b)
static uint16_t pc_last_line10[PC_TBL_SIZE];
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15

// Tunables (retuned)
// Streaming: 2 consecutive +1/+2 forward steps -> strict stream quarantine
static constexpr uint8_t STREAM_CONF_THRESH = 2;
static constexpr uint8_t STREAM_DEMOTE_STEP = 2; // demote by +2 on stream/quarantine hits
// Multi-hit promotion gates
static constexpr uint8_t HITS_PROMOTE_NS    = 2; // non-stream promote to MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 2; // stream escape on 2nd demand hit
// Insert depths (RRIP)
static constexpr uint8_t INSERT_NEAR_MRU    = 2; // near-MRU
static constexpr uint8_t INSERT_COLD_TAIL   = 6; // near-tail (avoid pollution)
// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH = 7;     // >=7 is "hot"
static constexpr uint32_t LFU_DECAY_PERIOD  = 2048;  // faster decay

// Helpers
static inline void sat_inc_u2(uint8_t& x){ if (x < 3) x++; }
static inline void sat_inc_u4(uint8_t& x){ if (x < 15) x++; }
static inline uint8_t clamp_rrpv(uint8_t v){ return (v > maxRRPV) ? maxRRPV : v; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr){
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static uint32_t op_count = 0;
static inline void maybe_decay_lfu(){
    op_count++;
    if ((op_count & (LFU_DECAY_PERIOD - 1)) == 0){
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
}

// ---------------- Initialization ----------------------------------
void InitReplacementState(){
    for (uint32_t s = 0; s < LLC_SETS; s++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    GSEL = 0;
    op_count = 0;
}

// ---------------- Victim selection (RRIP) -------------------------
static inline uint32_t rrip_pick_victim(uint32_t set){
    // Return first way at maxRRPV; caller ensures no invalid present
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (rrpv[set][w] == maxRRPV) return w;
    }
    return LLC_WAYS; // not found
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
){
    // 1) If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (!current_set[w].valid) return w;
    }

    // 2) RRIP victim with aging; guaranteed to terminate
    while (true){
        uint32_t v = rrip_pick_victim(set);
        if (v != LLC_WAYS){
            // Train leaders (eviction outcome)
            if (SEL_SAMPLED(set)){
                uint32_t slot = LEADER_SLOT(set);
                // Only train if this line was previously tagged (valid eviction)
                uint8_t used = hawk_used[slot][v];
                uint8_t was_pref = hawk_is_pref[slot][v];
                uint16_t sig = (hawk_sig[slot][v] & 0x0FFFu);
                if (current_set[v].valid){
                    // Update SHCT for Mode A training
                    if (was_pref) {
                        if (used) shct_inc(shct_prefetch[shct_idx(sig)]);
                        else      shct_dec(shct_prefetch[shct_idx(sig)]);
                    } else {
                        if (used) shct_inc(shct_demand[shct_idx(sig)]);
                        else      shct_dec(shct_demand[shct_idx(sig)]);
                    }
                    // Update global selector: Mode B wins push up; Mode A wins pull down
                    if (LEADER_B(set)){
                        if (used) gsel_inc(1);
                        else      gsel_dec(3);
                    } else { // LEADER_A
                        if (used) gsel_dec(2);
                        else      gsel_inc(1);
                    }
                }
                // Clear tags for the evicted way
                hawk_used[slot][v] = 0;
                hawk_is_pref[slot][v] = 0;
                hawk_sig[slot][v] = 0;
            }
            return v;
        }
        // Age all ways by +1 (saturating)
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

// ---------------- Update state on hit/fill ------------------------
void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
){
    (void)cpu; (void)victim_addr;
    maybe_decay_lfu();

    // Common derived indices
    uint32_t pc_idx = pc_index(PC);

    if (hit){
        // Mark leader "used" on any hit
        if (SEL_SAMPLED(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        // Update TinyLFU on demand hit
        if (is_demand(type)) sat_inc_u4(pc_use4[pc_idx]);

        // Stream quarantine behavior
        if (stream_lock[set][way]){
            if (is_demand(type)){
                // First demand hit: still locked, demote; second: escape and promote
                if (hitcnt[set][way] + 1 >= HITS_PROMOTE_STR){
                    stream_lock[set][way] = 0; // escape quarantine
                    hitcnt[set][way] = HITS_PROMOTE_STR; // cap for consistency
                    rrpv[set][way] = 0; // promote to MRU on escape
                } else {
                    hitcnt[set][way]++;
                    // demote aggressively
                    uint8_t newv = rrpv[set][way] + STREAM_DEMOTE_STEP;
                    rrpv[set][way] = clamp_rrpv(newv);
                }
            } else {
                // Prefetch hit while locked: demote a bit
                if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
            }
            return;
        }

        // Non-stream lines: multi-hit gating for promotion
        if (is_demand(type)){
            hitcnt[set][way] = (hitcnt[set][way] < 3) ? (uint8_t)(hitcnt[set][way] + 1) : 3;
            if (hitcnt[set][way] >= HITS_PROMOTE_NS){
                rrpv[set][way] = 0; // promote to MRU
            } // else: do not promote on first demand hit
        } else {
            // Prefetch hit: never promote; slight aging to keep scan resistance
            if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }
        return;
    }

    // Miss/Fill path: choose Mode and insertion depth
    bool modeB = use_modeB(set);

    // Default insertion
    uint8_t ins_rrpv = INSERT_COLD_TAIL;
    uint8_t ins_stream_lock = 0;

    if (type == ACCESS_WRITEBACK){
        // Never bypass writebacks: modest priority
        ins_rrpv = INSERT_NEAR_MRU;
        ins_stream_lock = 0;
    } else if (!modeB){
        // Mode A (Hawkeye-like)
        uint16_t sig = pc_sig12(PC);
        uint8_t pred = is_demand(type) ? shct_demand[shct_idx(sig)] : shct_prefetch[shct_idx(sig)];
        bool friendly = (pred >= (SHCT_MAX >> 1)); // mid-threshold
        if (type == ACCESS_PREFETCH){
            ins_rrpv = maxRRPV;     // quarantine prefetches
            ins_stream_lock = 1;
        } else {
            ins_rrpv = friendly ? INSERT_NEAR_MRU : INSERT_COLD_TAIL;
            ins_stream_lock = 0;
        }
    } else {
        // Mode B (StreamQuant-LFU RT-StrictStream)
        bool is_stream = detect_and_update_stream(PC, paddr);
        if (type == ACCESS_PREFETCH){
            ins_rrpv = maxRRPV;     // strict quarantine
            ins_stream_lock = 1;
        } else if (is_stream && is_demand(type)){
            // Strict: stream demand "bypass" via hard-tail insert + quarantine
            ins_rrpv = maxRRPV;
            ins_stream_lock = 1;
        } else {
            // TinyLFU-guided insertion for non-stream
            uint8_t use = pc_use4[pc_idx];
            ins_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_NEAR_MRU : INSERT_COLD_TAIL;
            ins_stream_lock = 0;
        }
    }

    // Apply insertion
    rrpv[set][way]       = clamp_rrpv(ins_rrpv);
    stream_lock[set][way]= ins_stream_lock;
    hitcnt[set][way]     = 0;

    // Leader tagging for training (both modes record, but SHCT only used by Mode A)
    if (SEL_SAMPLED(set)){
        uint32_t slot = LEADER_SLOT(set);
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        hawk_used[slot][way]    = 0;
    }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}