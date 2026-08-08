#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

// CRC2 constants
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD) || (t==ACCESS_RFO); }

// ---------------- RRIP core (shared by both modes) ----------------
static constexpr uint8_t maxRRPV = 7; // 3-bit RRIP
static uint8_t rrpv[LLC_SETS][LLC_WAYS];

// Per-line compact metadata (conceptually bit-packed: rrpv:3b, hitcnt:2b, stream_lock:1b)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // 0..3
static uint8_t stream_lock[LLC_SETS][LLC_WAYS];// 0/1

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
static constexpr uint8_t GSEL_THRES = 16;  // followers enable Mode B only if >= THRES
static uint8_t GSEL = 0;
static inline void gsel_inc(){ if (GSEL < GSEL_MAX) GSEL++; }
static inline void gsel_dec(){ if (GSEL > 0) GSEL--; }
static inline bool use_modeB(uint32_t set){
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (GSEL >= GSEL_THRES);
}

// ---------------- Mode A: Hawkeye-like friendly/averse SHCT -------
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1 << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];   // 5-bit counters
static uint8_t shct_prefetch[SHCT_SIZE]; // 5-bit counters

static inline uint16_t pc_sig12(uint64_t pc){
    // light hash to 12b
    uint64_t x = pc ^ (pc >> 2) ^ (pc >> 5) ^ (pc >> 13);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_SIZE - 1); }
static inline void shct_inc(uint8_t& x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x){ if (x > 0) x--; }

// Leader-only per-line tags for Mode A training
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
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

// Tunables (exposed for exploration)
// Streaming: 2 consecutive +1/+2 forward steps -> stream
static constexpr uint8_t STREAM_CONF_THRESH = 2;
static constexpr uint8_t STREAM_DEMOTE_TOUCH= 1; // demote by +1 on stream/quarantine hits
// Multi-hit promotion gates
static constexpr uint8_t HITS_PROMOTE_NS    = 2; // non-stream promote to MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR   = 2; // stream escape on 2nd demand hit
// Insert depths
static constexpr uint8_t INSERT_WARM_DEPTH  = 2; // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH  = 6; // near-tail (avoid pollution)
// TinyLFU
static constexpr uint8_t PC_USE_HOT_THRESH  = 6; // >=6 is "hot"
static constexpr uint32_t LFU_DECAY_PERIOD  = 4096;

// Helpers
static inline void sat_inc_u2(uint8_t& x){ if (x < 3) x++; }
static inline void sat_inc_u4(uint8_t& x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x){ if (x > 0) x--; }
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
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set){
    // 1) Prefer invalid
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (!current_set[w].valid) return w;
    }
    // 2) Find any at maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++){
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // 3) Bounded aging passes to guarantee termination
    for (int pass = 0; pass < 8; pass++){
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++){
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Fallback (should not occur)
    return 0;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
){
    return rrip_victim_and_age(set, current_set);
}

// ---------------- Update state ------------------------------------
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
){
    maybe_decay_lfu();

    // Mode select (followers default to A unless selector enables B)
    bool modeB = use_modeB(set);

    // WRITEBACKs: never bypass; insert warm on miss, protect slightly on hit
    if (type == ACCESS_WRITEBACK){
        if (hit){
            // keep dirty line alive modestly
            rrpv[set][way] = (rrpv[set][way] > 1) ? (rrpv[set][way] - 1) : 0;
        } else {
            // insert warm
            rrpv[set][way] = INSERT_WARM_DEPTH;
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 0;
        }
        return;
    }

    // Leader-set selector training using misses only (conservative)
    if (!hit && SEL_SAMPLED(set)){
        if (LEADER_A(set)) gsel_inc(); // A missed -> bias toward B
        else if (LEADER_B(set)) gsel_dec(); // B missed -> bias toward A
    }

    // Stream detection (used only by Mode B behavior)
    bool is_stream = false;
    if (modeB && is_demand(type)){
        is_stream = detect_and_update_stream(PC, paddr);
    } else if (modeB && type == ACCESS_PREFETCH){
        // advance detector even on pref touches to follow the run
        (void)detect_and_update_stream(PC, paddr);
    }

    // TinyLFU update on demand events
    if (is_demand(type)) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
    }

    // Mode A leader training (SHiP-style) at eviction time and on hits
    if (SEL_SAMPLED(set)){
        uint32_t slot = LEADER_SLOT(set);
        if (hit){
            // record re-reference
            hawk_used[slot][way] = 1;
        } else {
            // fill: train previous occupant, then tag new line
            uint16_t psig = hawk_sig[slot][way];
            uint8_t  used = hawk_used[slot][way];
            uint8_t  was_pref = hawk_is_pref[slot][way];
            if (psig){
                uint32_t idx = shct_idx(psig);
                if (was_pref) {
                    if (used) shct_inc(shct_prefetch[idx]);
                    else      shct_dec(shct_prefetch[idx]);
                } else {
                    if (used) shct_inc(shct_demand[idx]);
                    else      shct_dec(shct_demand[idx]);
                }
            }
            // install new metadata
            hawk_sig[slot][way] = pc_sig12(PC);
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = (type == ACCESS_PREFETCH) ? 1 : 0;
        }
    }

    if (hit){
        // On hit, apply multi-hit gating and quarantine behavior
        if (stream_lock[set][way]){
            // quarantined (stream or prefetch)
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_STR){
                    // escape quarantine
                    stream_lock[set][way] = 0;
                    rrpv[set][way] = 0; // MRU
                } else {
                    // demote on touch to keep scan resistance
                    uint8_t v = rrpv[set][way];
                    rrpv[set][way] = clamp_rrpv((uint8_t)(v + STREAM_DEMOTE_TOUCH));
                }
            } else {
                // prefetch hit does not count toward escape; demote slightly
                uint8_t v = rrpv[set][way];
                rrpv[set][way] = clamp_rrpv((uint8_t)(v + STREAM_DEMOTE_TOUCH));
            }
        } else {
            // non-stream path: promote to MRU only on 2nd+ demand hit
            if (is_demand(type)) {
                if (hitcnt[set][way] < 3) hitcnt[set][way]++;
                if (hitcnt[set][way] >= HITS_PROMOTE_NS){
                    rrpv[set][way] = 0; // MRU
                } else {
                    // gentle nudge toward MRU on first demand hit
                    if (rrpv[set][way] > 1) rrpv[set][way] -= 1;
                }
            } else {
                // prefetch hits do not promote
                uint8_t v = rrpv[set][way];
                rrpv[set][way] = (v > 0) ? (v - 1) : 0;
            }
        }
        return;
    }

    // Miss path: choose insertion policy per mode
    // Prefetches are always quarantined at tail
    if (type == ACCESS_PREFETCH){
        rrpv[set][way] = maxRRPV;      // tail
        hitcnt[set][way] = 0;
        stream_lock[set][way] = 1;     // quarantine
        return;
    }

    if (modeB){
        // Mode B insertion
        if (is_stream){
            // stream quarantine: hard tail, never promote until 2nd demand hit
            rrpv[set][way] = maxRRPV;
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 1;
        } else {
            // use TinyLFU to steer insertion depth
            bool hot = (pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH);
            rrpv[set][way] = hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
            hitcnt[set][way] = 0;
            stream_lock[set][way] = 0;
        }
    } else {
        // Mode A insertion (Hawkeye-like friendly/averse via SHCT)
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t conf = is_demand(type) ? shct_demand[idx] : shct_prefetch[idx];
        bool friendly = (conf > (SHCT_MAX >> 1)); // >15
        rrpv[set][way] = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        hitcnt[set][way] = 0;
        stream_lock[set][way] = 0; // Mode A does not use quarantine except for prefetch (handled above)
        // For leader sets, tag was already updated earlier
        if (SEL_SAMPLED(set)){
            uint32_t slot = LEADER_SLOT(set);
            hawk_sig[slot][way] = sig;
            hawk_used[slot][way] = 0;
            hawk_is_pref[slot][way] = 0;
        }
    }
}

// ---------------- Stats (kept minimal) ----------------------------
void PrintStats(){}
void PrintStats_Heartbeat(){}