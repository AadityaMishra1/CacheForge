/*
 * Hawkeye + CrossRun-DeadGate (CR-DG)
 * STRICT ENSEMBLE: EXCLUSIVE state per mode; no shared RRIP arrays.
 *
 * Tunables (telemetry hooks):
 * - Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, SELECTOR_EPOCH_SIZE=4096, MAX_MODE_B_FOLLOWERS=256
 * - Mode B thresholds:
 *     HOT_THRESHOLD=8 (TinyLFU), DEAD_THRESHOLD=2, STREAM_ARM=2 (±1/±2), LONG_RUN_UNIQUE_SETS=6,
 *     INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7, BYPASS_PROB≈50% via per-PC flip.
 * - Epoch decay (each selector epoch): pc_freq>>=1, pc_dead>>=1, pc_run_len>>=1, pc_setmask>>=1, pc_hitstreak>>=1
 *
 * Storage budget: ~30.234 KB total (see breakdown).
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <cstring>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11

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

// lightweight CRC32 for hashing
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
static constexpr int32_t  MODE_A_BIAS         = 50;  // much higher bias to Hawkeye
static constexpr int8_t   MODE_B_THRESHOLD    = 7;   // stricter local per-set threshold

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
#define maxRRPV 7
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
// Mode B: CrossRun-DeadGate (EXCLUSIVE rrpv_B + tiny per-line flag)
//------------------------------------------------------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];        // 3-bit logical
static uint8_t mode_B_line_touched[LLC_SETS][LLC_WAYS]; // 0/1 first-hit flag

// Compact PC table (128 entries)
#define MODE_B_PC_TABLE_SIZE 128
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE]; // 10-bit, 0xFFFF = invalid
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2-bit
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];   // 4-bit TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];   // 2-bit dead score
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE]; // last set low-8
static uint8_t  mode_B_pc_setmask8[MODE_B_PC_TABLE_SIZE];  // 8-bit unique-set mask
static uint8_t  mode_B_pc_run_len[MODE_B_PC_TABLE_SIZE];   // 0-15
static uint8_t  mode_B_pc_flip[MODE_B_PC_TABLE_SIZE];      // 0/1 bypass coin
static uint8_t  mode_B_pc_hitstreak[MODE_B_PC_TABLE_SIZE]; // 0-3

// Tunables - HST-E Logic (Stride + TinyLFU + Stream Quarantine + 2-hit Rescue)
static constexpr uint8_t MODE_B_INSERT_WARM        = 0;  // MRU for hot (TinyLFU >= threshold)
static constexpr uint8_t MODE_B_INSERT_COLD        = 3;  // mid-age for cold
static constexpr uint8_t MODE_B_STREAM_TAIL        = 7;  // LRU for streams/prefetch
static constexpr uint8_t MODE_B_STREAM_ARM         = 2;  // need 2 consecutive ±1/±2 steps
static constexpr uint8_t MODE_B_HOT_THRESHOLD      = 6;  // TinyLFU >=6 (more aggressive promotion)
static constexpr uint8_t MODE_B_DEAD_THRESHOLD     = 2;  // dead >=2 => tail/bypass
static constexpr uint8_t MODE_B_LONGRUN_UNIQUESETS = 4;  // >=4 unique sets (tighter stream detection)
// helpers
static inline uint32_t mode_B_pc_index(uint64_t pc){ return CRC32(pc) & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); }
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline uint8_t popcount8(uint8_t x){ x = x - ((x>>1)&0x55); x = (x&0x33)+((x>>2)&0x33); return (((x + (x>>4)) & 0x0F)); }

// Detect stride/run and update per-PC run tracking (uses previous state)
static inline bool mode_B_detect_stream(uint32_t pc_idx, uint64_t paddr, uint32_t set){
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[pc_idx];
    bool step = false;
    if(last != 0xFFFF){
        // bidirectional ±1/±2
        step = (line_id == (uint16_t)(last+1)) || (line_id == (uint16_t)(last+2)) ||
               (line_id == (uint16_t)(last-1)) || (line_id == (uint16_t)(last-2));
    }
    if(step) {
        sat_inc_u2(mode_B_pc_stride_conf[pc_idx]);
        if(mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM){
            // grow run_len and unique-set mask on armed stream
            if(mode_B_pc_run_len[pc_idx] < 15) mode_B_pc_run_len[pc_idx]++;
            // track low-3-bits of set in 8-bit mask
            uint8_t bit = (uint8_t)(1u << (set & 7));
            mode_B_pc_setmask8[pc_idx] |= bit;
        }
    } else {
        mode_B_pc_stride_conf[pc_idx] = 0;
        mode_B_pc_run_len[pc_idx] = 0;
        // keep setmask; it will decay epochically
    }
    mode_B_pc_last_line[pc_idx] = line_id;
    mode_B_pc_last_set8[pc_idx] = (uint8_t)(set & 0xFFu);
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM);
}

static inline bool mode_B_is_long_run(uint32_t pc_idx){
    return (mode_B_pc_stride_conf[pc_idx] >= MODE_B_STREAM_ARM) &&
           (popcount8(mode_B_pc_setmask8[pc_idx]) >= MODE_B_LONGRUN_UNIQUESETS);
}

static inline bool mode_B_should_bypass(uint32_t set, uint64_t PC, uint64_t paddr, uint32_t type){
    if(is_prefetch(type)) return false; // prefetches are quarantined, not bypassed
    uint32_t pc_idx = mode_B_pc_index(PC);
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run = mode_B_is_long_run(pc_idx);
    bool pc_hot   = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead  = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // HST-E: More conservative bypass - only bypass long streams that are definitively cold AND dead
    if(is_stream && long_run && !pc_hot && pc_dead)
        return true;

    return false;
}

// Mode B insertion/update (updates ONLY rrpv_B and Mode B metadata)
static inline void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pc_idx = mode_B_pc_index(PC);

    // Frequency and dead-gate updates
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[pc_idx]);
    if(hit) {
        // demand hit => reinforce usefulness
        if(mode_B_pc_dead[pc_idx] > 0) mode_B_pc_dead[pc_idx]--;
        if(mode_B_pc_hitstreak[pc_idx] < 3) mode_B_pc_hitstreak[pc_idx]++;
    } else {
        // demand miss => potential dead-on-fill signal
        if(is_demand(type)){
            if(mode_B_pc_hitstreak[pc_idx] == 0 && mode_B_pc_dead[pc_idx] < 3) mode_B_pc_dead[pc_idx]++;
            if(mode_B_pc_hitstreak[pc_idx] > 0) mode_B_pc_hitstreak[pc_idx]--; // temper burstiness
        }
    }

    // Stream detection (uses previous last_line)
    bool is_stream = mode_B_detect_stream(pc_idx, paddr, set);
    bool long_run  = mode_B_is_long_run(pc_idx);
    bool pc_hot    = (mode_B_pc_freq[pc_idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_dead   = (mode_B_pc_dead[pc_idx] >= MODE_B_DEAD_THRESHOLD);

    // Insertion policy (affects ONLY rrpv_B)
    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;     // quarantine prefetches
    } else if(is_stream && long_run){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;     // long-run quarantine
    } else if(pc_dead){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL;     // dead/noisy PCs to tail
    } else if(pc_hot){
        rrpv_B[set][way] = MODE_B_INSERT_WARM;     // warm for hot PCs
    } else {
        rrpv_B[set][way] = MODE_B_INSERT_COLD;     // near-tail for cold PCs
    }

    // Stream-aware multi-hit rescue using a tiny per-line flag + per-PC hitstreak
    if(hit){
        if(mode_B_line_touched[set][way]){
            // Second hit observed: allow promotion; require small hitstreak if stream-like
            if(!is_stream || mode_B_pc_hitstreak[pc_idx] >= 2){
                rrpv_B[set][way] = 0; // MRU promotion
            }
        } else {
            mode_B_line_touched[set][way] = 1; // mark first hit
        }
    } else {
        mode_B_line_touched[set][way] = 0; // reset on fill
    }
}

// Mode B victim (uses ONLY rrpv_B)
static inline uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

//------------------------------------------------------------------------------
// ChampSim interface
//------------------------------------------------------------------------------
void InitReplacementState(){
    // Clear per-line state
    for(uint32_t s=0; s<LLC_SETS; s++){
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_line_touched[s][w] = 0;
        }
    }
    // Selector init
    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false; selector_epoch = 0;
    std::memset(mode_conf_nibbles, 0x88, sizeof(mode_conf_nibbles)); // all zeros (-8..+7) => store +8 => 0x8
    std::memset(follower_bits, 0, sizeof(follower_bits));
    num_mode_B_followers = 0;

    // Hawkeye-lite SHCT
    std::memset(SHCT_cnt, 0, sizeof(SHCT_cnt));

    // Mode B PC tables
    for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_last_set8[i] = 0;
        mode_B_pc_setmask8[i] = 0;
        mode_B_pc_run_len[i] = 0;
        mode_B_pc_flip[i] = 0;
        mode_B_pc_hitstreak[i] = 0;
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu;
    bool use_B = should_use_mode_B(set);

    if(use_B){
        // Optional bypass for Mode B
        if(mode_B_should_bypass(set, PC, paddr, type))
            return LLC_WAYS; // ChampSim bypass
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

    // Update conservative selector
    update_selector(set, hit);

    // Epochic decay for Mode B PC stats piggybacked on selector epoch
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        for(uint32_t i=0; i<MODE_B_PC_TABLE_SIZE; i++){
            mode_B_pc_freq[i]      >>= 1;
            mode_B_pc_dead[i]      >>= 1;
            mode_B_pc_run_len[i]   >>= 1;
            mode_B_pc_setmask8[i]  >>= 1;
            mode_B_pc_hitstreak[i] >>= 1;
        }
    }

    // Mode A: train tiny SHCT (friendly if hits)
    hawkeye_train(PC, hit != 0);

    bool use_B = should_use_mode_B(set);

    // EXCLUSIVE update: only chosen mode updates its RRIP array
    if(use_B){
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Hawkeye-lite insertion
        bool friendly = hawkeye_is_friendly(PC);
        if(is_prefetch(type)){
            rrpv_A[set][way] = maxRRPV; // avoid polluting
        } else {
            rrpv_A[set][way] = friendly ? 0 : maxRRPV;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}