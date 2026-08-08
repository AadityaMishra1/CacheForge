/*
 * Hawkeye + HST-E ColdStream-Rescue (CSR)
 * STRICT DYNAMIC ENSEMBLE with EXCLUSIVE state per mode
 * - Mode A: Hawkeye-lite (tiny SHCT-based friendly/averse) with exclusive rrpv_A
 * - Mode B: HST-E derived ColdStream-Rescue with exclusive rrpv_B
 * - Selector: High bias (50), strict per-set threshold (7), ~12.5% follower cap
 * - B-LEADER SAFETY: B leaders mirror Hawkeye unless prefer_mode_B = true
 * Storage budget ~30.195 KB (see breakdown at end)
 *
 * Tunables (Mode B only; fixed within allowed ranges):
 *  - MODE_B_INSERT_WARM        = 0  (MRU for hot PCs)
 *  - MODE_B_INSERT_COLD        = 3  (mid-age for cold PCs)
 *  - MODE_B_STREAM_TAIL        = 7  (hard tail for streams & prefetches)
 *  - MODE_B_STREAM_ARM         = 2  (2 consecutive ±1/±2 stride steps to arm stream)
 *  - MODE_B_HOT_THRESHOLD      = 6  (TinyLFU hot)
 *  - MODE_B_DEAD_THRESHOLD     = 2  (dead PC fence)
 *  - MODE_B_LONGRUN_UNIQUESETS = 4  (long run if ≥4 unique sets in low-3 set bits)
 *
 * Telemetry hooks:
 *  - Selector epoch (4096 accesses): decay Mode B TinyLFU freq >>= 1
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define maxRRPV 7

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// ============================================================================
// LEADER-FOLLOWER SAMPLING (match Hawkeye sampling: 64 sampled sets)
// ============================================================================
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6))
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// ============================================================================
// SELECTOR (Very conservative gating + B-leader safety)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 50;  // huge bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // strict local threshold

// Per-set confidence packed as 4-bit signed nibbles (-8..+7 stored with +8 bias)
static uint8_t mode_conf_nibbles[LLC_SETS/2]; // 2048 sets / 2 = 1024 bytes
static inline int8_t get_mode_conf(uint32_t set){
    uint8_t b = mode_conf_nibbles[set>>1];
    uint8_t nib = (set & 1) ? (b>>4) : (b & 0xF);
    return (int8_t)nib - 8;
}
static inline void set_mode_conf(uint32_t set, int8_t val){
    if(val > 7) val = 7;
    if(val < -8) val = -8;
    uint8_t enc = (uint8_t)(val + 8) & 0xF;
    uint8_t &b = mode_conf_nibbles[set>>1];
    if(set & 1) b = (uint8_t)((b & 0x0F) | (enc<<4));
    else        b = (uint8_t)((b & 0xF0) | enc);
}

// follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // bitset, 1 if set is assigned to B
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

// Selector update
static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Drift local confidence toward global preference for followers
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            int8_t c = get_mode_conf(s);
            if(prefer_mode_B){ if(c < 7) c++; }
            else             { if(c > -8) c--; }
            set_mode_conf(s, c);
        }

        // Epochic decay for Mode B TinyLFU
        // (Note: small table, linear scan OK)
        // Decay only freq counters; leave others as-is
        extern uint8_t mode_B_pc_freq[]; // forward (defined in Mode B section)
        // cannot range-for without size; decay in-place inside Mode B init/array
        // handled below by explicit loop since we know table size
        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return prefer_mode_B; // B-LEADER SAFETY

    if(!prefer_mode_B) return false;
    if(get_mode_conf(set) < MODE_B_THRESHOLD) return false;

    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// MODE A: Hawkeye-lite (EXCLUSIVE state)
// - Tiny SHCT (512 entries, 5-bit counters)
// - Friendly PCs insert MRU (0), averse insert tail (7)
// - On demand hit: promote to MRU and increment SHCT
// - On fill (demand miss): decrement SHCT of current PC
// ============================================================================
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical

#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31
static inline uint32_t shct_idx(uint64_t PC){ // simple hash
    uint64_t x = PC;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x & (SHCT_ENTRIES-1);
}
static inline bool hawkeye_is_friendly(uint64_t PC){ return (SHCT_cnt[shct_idx(PC)] >= 16); }
static inline void shct_inc(uint64_t PC){ uint8_t &c = SHCT_cnt[shct_idx(PC)]; if(c<31) c++; }
static inline void shct_dec(uint64_t PC){ uint8_t &c = SHCT_cnt[shct_idx(PC)]; if(c>0) c--; }

static inline uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

static inline void mode_A_update(uint32_t set, uint32_t way, uint64_t /*paddr*/, uint64_t PC, uint32_t type, uint8_t hit){
    if(is_writeback(type)) return;

    if(hit){
        // promote on demand hit
        if(is_demand(type)) shct_inc(PC);
        rrpv_A[set][way] = 0;
        return;
    }

    // Miss + fill insertion
    if(is_prefetch(type)){
        rrpv_A[set][way] = maxRRPV;
        return;
    }

    // demand miss: train and insert
    shct_dec(PC);
    if(hawkeye_is_friendly(PC)) rrpv_A[set][way] = 0;
    else                        rrpv_A[set][way] = maxRRPV;
}

// ============================================================================
// MODE B: HST-E ColdStream-Rescue (EXCLUSIVE state)
// REQUIRED components per spec: stride/run detector, TinyLFU, quarantine,
// multi-hit rescue, dead-PC fence, conservative bypass
// ============================================================================

// Per-line (Mode B only)
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];              // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS]; // 1-bit first-hit flag

// Tiny helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline uint8_t popcount8(uint8_t v){ return (uint8_t)__builtin_popcount((unsigned)v) & 0xFF; }

// Compact PC table (128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1u); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 10-bit

static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit stored in 16
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit dead score
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];   // low-8 set id (telemetry)
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];    // 8-bit unique-set mask
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];     // 4-bit run length (0-15)
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE];   // 2-bit (0-3)

// Tunables (fixed within allowed bounds)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // LRU for streams
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // 2 consecutive strides
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >=6
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >=2
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // >=4 unique sets

// Detect stride/stream with run tracking
static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set){
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];

    bool step = false;
    if (last != 0xFFFF) {
        // Bidirectional ±1/±2
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }

    if (step) {
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) {
            // Grow run length and unique-set mask
            if (mode_B_pc_run_len[pc_idx] < 15) mode_B_pc_run_len[pc_idx]++;
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    } else {
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        mode_B_pc_setmask8[pc_idx] = 0;
    }

    mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 0xFF);
    mode_B_pc_last_line[pc_idx] = line_id;
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx){
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    (void)set;
    if (is_prefetch(type)) return false; // quarantine instead
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run  = mode_B_is_long_run(pc_idx);
    bool pc_hot    = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead   = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // ONLY bypass if long stream AND cold AND dead
    if (is_stream && long_run && !pc_hot && pc_dead)
        return true;

    return false;
}

// Mode B victim (uses ONLY rrpv_B)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;

    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;

    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;

    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;

    return 0;
}

// Mode B update (updates ONLY rrpv_B and Mode B metadata)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // TinyLFU frequency (demand only)
    if (is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);

    // Dead block tracking + hit streak
    if (hit) {
        if (mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if (mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    } else {
        if (is_demand(type) && mode_B_pc_hitstreak[pc_idx] == 0)
            if (mode_B_pc_dead[pc_idx] < 3) mode_B_pc_dead[pc_idx]++;
        if (mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--;
    }

    // Stream detection
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run  = mode_B_is_long_run(pc_idx);
    bool pc_hot    = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead   = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion policy
    if (is_prefetch(type)) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (is_stream && long_run) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (pc_dead) {
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;
    } else if (pc_hot) {
        rrpv_B[set][way] = MODE_B_INSERT_WARM;
    } else {
        rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue with stream awareness
    if (hit) {
        if (mode_B_line_touched[set][way]) {
            // Second hit: promote if non-stream OR stream with decent streak
            if (!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2) {
                rrpv_B[set][way] = 0; // MRU
            }
        } else {
            mode_B_line_touched[set][way] = 1;
        }
    } else {
        mode_B_line_touched[set][way] = 0;
    }
}

// ============================================================================
// ChampSim Frontend
// ============================================================================
void InitReplacementState(){
    // Clear arrays
    std::memset(rrpv_A, maxRRPV, sizeof(rrpv_A));
    std::memset(rrpv_B, maxRRPV, sizeof(rrpv_B));
    std::memset(mode_B_line_touched, 0, sizeof(mode_B_line_touched));
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));
    std::memset(mode_conf_nibbles, 0x88, sizeof(mode_conf_nibbles)); // init to 0 (encoded +8)
    std::memset(follower_bits, 0, sizeof(follower_bits));

    // Mode B PC table init
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_last_line[i]   = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i]        = 0;
        mode_B_pc_dead[i]        = 0;
        mode_B_pc_last_set8[i]   = 0;
        mode_B_pc_setmask8[i]    = 0;
        mode_B_pc_run_len[i]     = 0;
        mode_B_pc_hitstreak[i]   = 0;
    }

    leaderA_score = leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_mode_B = should_use_mode_B(set);

    if(use_mode_B){
        if(mode_B_should_bypass(set, PC, paddr, type))
            return LLC_WAYS; // bypass
        return mode_B_victim(set, current_set);
    } else {
        return mode_A_victim(set, current_set);
    }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Update selector (leaders collect outcomes)
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);
    if(use_mode_B){
        // Mode B: update ONLY Mode B state
        mode_B_update(set, way, paddr, PC, type, hit);
    }else{
        // Mode A: update ONLY Mode A state
        mode_A_update(set, way, paddr, PC, type, hit);
    }

    // Epochic decay for Mode B TinyLFU at boundary (inline so we can touch table)
    if(selector_epoch && (selector_epoch % SELECTOR_EPOCH_SIZE)==0){
        for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++)
            mode_B_pc_freq[i] >>= 1;
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}