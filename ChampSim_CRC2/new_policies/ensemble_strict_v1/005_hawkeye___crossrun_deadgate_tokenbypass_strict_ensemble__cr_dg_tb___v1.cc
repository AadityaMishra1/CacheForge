/*
 * Hawkeye + CrossRun-DeadGate TokenBypass (CR-DG-TB)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Tunables (kept conservative):
 * - Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, SELECTOR_EPOCH_SIZE=4096, MAX_MODE_B_FOLLOWERS=256
 * - Mode B thresholds:
 *     HOT_THRESHOLD=8 (TinyLFU), DEAD_THRESHOLD=2 (2-bit), STREAM_ARM=2 (forward +1/+2),
 *     INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7,
 *     PROMOTE_NONSTREAM=2 hits, PROMOTE_STREAM=3 hits,
 *     TOKENS_PER_EPOCH=2 (per-set 2-bit), BYPASS_ON_EXH_P=3/4 (probabilistic bypass when tokens exhausted).
 * - Epoch decay each selector epoch: pc_freq >>=1, pc_dead >>=1, tokens refill to TOKENS_PER_EPOCH.
 *
 * Storage budget: Mode B + selector ~= 27.201 KB (see breakdown section).
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11
#define maxRRPV 7

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// 64 sampled sets: low 6 bits == high 6 bits
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); }

// lightweight CRC32 for hashing / pseudo-randomness
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

//------------------------------------------------------------------------------
// Selector (conservative gating)
//------------------------------------------------------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t  MODE_A_BIAS         = 33;  // high bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 5;   // local per-set threshold

// mode_confidence packed in 4-bit signed nibbles (-8..+7) -> stored 0..15 with +8 bias
static uint8_t mode_conf_nibbles[LLC_SETS/2]; // 2048 sets -> 1024 bytes
static inline int8_t get_mode_conf(uint32_t set){
    uint8_t b = mode_conf_nibbles[set>>1];
    uint8_t nib = (set & 1) ? (b>>4) : (b & 0xF);
    int8_t val = (int8_t)nib - 8;
    return val;
}
static inline void set_mode_conf(uint32_t set, int8_t val){
    if(val > 7) val = 7;
    if(val < -8) val = -8;
    uint8_t enc = (uint8_t)(val + 8) & 0xF;
    uint8_t &b = mode_conf_nibbles[set>>1];
    if(set & 1) { b = (uint8_t)((b & 0x0F) | (enc<<4)); }
    else        { b = (uint8_t)((b & 0xF0) | enc); }
}

// follower cap (~12.5%)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_bits[LLC_SETS/8]; // bitset, 1 if set is assigned to B
static inline bool follower_is_B(uint32_t set){ return (follower_bits[set>>3] >> (set & 7)) & 1u; }
static inline void follower_set_B(uint32_t set){ follower_bits[set>>3] |= (uint8_t)(1u << (set & 7)); }

static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));
        // drift local confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            int8_t c = get_mode_conf(s);
            if(prefer_mode_B){ if(c < 7) c++; }
            else             { if(c > -8) c--; }
            set_mode_conf(s, c);
        }
        leaderA_score = 0; leaderB_score = 0;

        // Mode B decays + token refill (epochic)
        // Decay TinyLFU and dead-gate; refill per-set stream tokens
        extern uint8_t mode_B_pc_freq[];
        extern uint8_t mode_B_pc_dead[];
        extern uint8_t mode_B_tokens[];
        for(uint32_t i=0;i<512;i++){ mode_B_pc_freq[i] >>= 1; mode_B_pc_dead[i] >>= 1; }
        for(uint32_t s=0;s<LLC_SETS;s++){ mode_B_tokens[s] = 2; } // TOKENS_PER_EPOCH
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;
    if(!prefer_mode_B) return false;
    if(get_mode_conf(set) < MODE_B_THRESHOLD) return false;
    if(!follower_is_B(set)){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_set_B(set);
        num_mode_B_followers++;
    }
    return true;
}

//------------------------------------------------------------------------------
// Mode A: Hawkeye-lite (EXCLUSIVE rrpv_A + tiny SHCT)
//------------------------------------------------------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // logically 3-bit

// Tiny SHCT (single table, 512 entries, 5-bit counters)
#define SHCT_ENTRIES 512
static uint8_t SHCT_cnt[SHCT_ENTRIES]; // 0..31 stored in 5 bits
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){
    return (SHCT_cnt[shct_idx(PC)] >= 16);
}
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_cnt[shct_idx(PC)];
    if(hit){ if(c<31) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] < maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

//------------------------------------------------------------------------------
// Mode B: CrossRun-DeadGate TokenBypass (EXCLUSIVE rrpv_B + tiny per-PC)
//------------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];          // 3-bit RRIP
static uint8_t mode_B_hitcnt[LLC_SETS][LLC_WAYS];   // 2-bit hit count (0..3)
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit: stream-quarantined

// PC table (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10-bit (store in 16-bit)
uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit (0..3)
uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4-bit (0..15)
uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2-bit (0..3) noisy/dead gate

// Per-set 2-bit token bucket to cap stream admissions
uint8_t  mode_B_tokens[LLC_SETS]; // 0..3 (we use 0..2)

// Tunables
static constexpr uint8_t MODE_B_INSERT_WARM       = 2; // Near-MRU for hot PCs
static constexpr uint8_t MODE_B_INSERT_COLD       = 6; // Near-tail for cold PCs
static constexpr uint8_t MODE_B_STREAM_TAIL       = 7; // Hard tail for streams/prefetch
static constexpr uint8_t MODE_B_STREAM_ARM        = 2; // Need 2 forward steps to arm stream
static constexpr uint8_t MODE_B_HOT_THRESHOLD     = 8; // TinyLFU hot threshold
static constexpr uint8_t MODE_B_DEAD_THRESHOLD    = 2; // Dead/noisy PC threshold
static constexpr uint8_t MODE_B_PROMOTE_NONSTREAM = 2; // 2 hits promote
static constexpr uint8_t MODE_B_PROMOTE_STREAM    = 3; // 3 hits promote
static constexpr uint8_t MODE_B_TOKENS_PER_EPOCH  = 2; // Refill per epoch
static constexpr uint8_t MODE_B_BYPASS_NUM        = 3; // 3/4 bypass when tokens exhausted
static constexpr uint8_t MODE_B_BYPASS_DEN        = 4;

static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }

static inline uint32_t mode_B_pc_index(uint64_t pc) {
    return CRC32(pc) & (MODE_B_PC_TABLE_SIZE - 1);
}
static inline uint16_t mode_B_line_id(uint64_t paddr) {
    return (uint16_t)((paddr >> 6) & 0x3FF);  // 10-bit cache line ID
}

// Detect forward-only (+1/+2) stream; arm after 2 consecutive forward steps
static bool mode_B_detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool forward = false;
    if (last != 0xFFFF) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        forward = (line_id == exp1) || (line_id == exp2);
    }

    if (forward) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else         mode_B_pc_stride_conf[idx] = 0;

    mode_B_pc_last_line[idx] = line_id;
    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM);
}

// Mode B victim (uses ONLY rrpv_B)
static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (!current_set[w].valid) return w;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;

    for (uint32_t w = 0; w < LLC_WAYS; w++)
        if (rrpv_B[set][w] == maxRRPV) return w;

    return 0;
}

// Mode B insertion/update (updates ONLY rrpv_B + Mode B metadata)
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    uint32_t pc_idx = mode_B_pc_index(PC);

    // Update TinyLFU and dead-gate on demand
    if (is_demand(type)) {
        sat_inc_u4(mode_B_pc_freq[pc_idx]);
        if (hit) sat_dec_u2(mode_B_pc_dead[pc_idx]);
        else     sat_inc_u2(mode_B_pc_dead[pc_idx]);
    }

    // Stream detect (forward-only) and tag line
    bool is_stream = mode_B_detect_stream(PC, paddr);
    mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Update hit counter (demand hits only)
    if (hit && is_demand(type)) sat_inc_u2(mode_B_hitcnt[set][way]);
    else                        mode_B_hitcnt[set][way] = 0;

    // Base gating
    bool pc_hot   = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead  = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Default insertion depth (non-stream)
    uint8_t ins_rrpv = pc_hot ? MODE_B_INSERT_WARM : MODE_B_INSERT_COLD;

    // Stream/prefetch quarantine + token-bucket bypass
    if (is_stream || is_prefetch(type)) {
        bool bypass = false;
        if (mode_B_tokens[set] > 0) {
            mode_B_tokens[set]--; // admit, hard-tail
            ins_rrpv = MODE_B_STREAM_TAIL;
        } else {
            // Probabilistic bypass when tokens exhausted; bias stronger for cold/dead PCs
            uint32_t coin = CRC32(((uint64_t)set<<32) ^ PC ^ paddr ^ selector_epoch) & (MODE_B_BYPASS_DEN-1);
            uint8_t bypass_num = MODE_B_BYPASS_NUM;
            if (!pc_hot || pc_dead) bypass_num = MODE_B_BYPASS_NUM; // keep 3/4 for cold/noisy
            bypass = (coin < bypass_num);
            ins_rrpv = bypass ? maxRRPV : MODE_B_STREAM_TAIL;
        }
        rrpv_B[set][way] = ins_rrpv;
    } else {
        // Non-stream insertion
        rrpv_B[set][way] = ins_rrpv;
    }

    // Multi-hit rescue (demand-only):
    // - Non-stream: promote on 2nd hit
    // - Stream-tagged: require 3 hits to escape quarantine
    if (hit && is_demand(type)) {
        uint8_t need = mode_B_stream_tag[set][way] ? MODE_B_PROMOTE_STREAM : MODE_B_PROMOTE_NONSTREAM;
        if (mode_B_hitcnt[set][way] >= need) {
            rrpv_B[set][way] = 0;           // promote to MRU
            mode_B_stream_tag[set][way] = 0; // clear stream-tag once proven reusable
        }
    }
}

//------------------------------------------------------------------------------
// CHAMPSIM INTERFACE
//------------------------------------------------------------------------------
void InitReplacementState() {
    // Initialize Mode A and B arrays
    for (uint32_t i = 0; i < LLC_SETS; i++) {
        for (uint32_t j = 0; j < LLC_WAYS; j++) {
            rrpv_A[i][j] = maxRRPV;
            rrpv_B[i][j] = maxRRPV;
            mode_B_hitcnt[i][j] = 0;
            mode_B_stream_tag[i][j] = 0;
        }
        mode_B_tokens[i] = MODE_B_TOKENS_PER_EPOCH;
    }

    // Mode A SHCT
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B PC tables
    for (uint32_t i = 0; i < MODE_B_PC_TABLE_SIZE; i++) {
        mode_B_pc_last_line[i]   = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i]        = 0;
        mode_B_pc_dead[i]        = 0;
    }

    // Selector init
    std::memset(mode_conf_nibbles, 0x88 /* encodes 0 */, sizeof(mode_conf_nibbles));
    std::memset(follower_bits, 0, sizeof(follower_bits));
    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) return mode_B_victim(set, current_set);
    return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;  // line-align

    if (is_writeback(type)) return;

    // Update selector (leaders and epochic decays/tokens handled within)
    update_selector(set, hit);

    bool use_mode_B = should_use_mode_B(set);

    if (use_mode_B) {
        // Mode B EXCLUSIVE update
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A EXCLUSIVE update: Hawkeye-lite insertion + training
        bool friendly = hawkeye_is_friendly(PC);
        if (friendly) {
            // friendly: insert near MRU, mild aging if miss to avoid flooding
            if (!hit) {
                for (uint32_t i=0;i<LLC_WAYS;i++) if (rrpv_A[set][i] < maxRRPV-1) rrpv_A[set][i]++;
            }
            rrpv_A[set][way] = 0;
        } else {
            // averse: hard-tail
            rrpv_A[set][way] = maxRRPV;
        }
        hawkeye_train(PC, hit);
    }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}