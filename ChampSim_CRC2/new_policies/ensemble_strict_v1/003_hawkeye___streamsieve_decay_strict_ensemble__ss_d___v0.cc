/*
 * Hawkeye + StreamSieve-Decay (SS-D)
 * Strict Dynamic Ensemble: EXCLUSIVE RRIP state per mode; never cross-touch rrpv_A vs rrpv_B.
 *
 * Exploration-mode tunables (telemetry hooks):
 * - Selector: MODE_A_BIAS=33, MODE_B_THRESHOLD=5, EPOCH=4096, MAX_MODE_B_FOLLOWERS=256
 * - Mode B:
 *     HOT_THRESHOLD=8 (TinyLFU 4-bit), DEAD_STRONG=3 (2-bit), STREAM_ARM=2 (±1/±2),
 *     INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7, RUN_BYPASS_LEN=8, SHORT_RUN_CEIL=3,
 *     PROMOTE_NONSTREAM=2 hits, PROMOTE_STREAM=3 hits (and run_meter<=SHORT_RUN_CEIL),
 *     Epoch decay (pc_freq>>=1, pc_dead>>=1, pc_run_meter>>=1).
 *
 * Notes:
 *  - Default: followers use Hawkeye (Mode A). Mode B activates for a capped subset of sets only
 *    when it beats Mode A decisively (global bias + local threshold).
 *  - Bypass is approximated via hard-tail quarantine and extra aging (pushback) on long runs,
 *    keeping state separation intact.
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <map>
#include <vector>
#include <algorithm>
#include <cstring>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11
#define maxRRPV 7

// ChampSim CRC2 types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// Hash
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// Leader/follower sampling: 64 sampled sets
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u)==1u); }

// Saturating helpers
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

// ============================================================================
// Selector (Conservative gating)
// ============================================================================
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;
static constexpr int8_t MODE_B_THRESHOLD = 5;

static int8_t mode_confidence[LLC_SETS];  // -8..+7 (stored as int8_t)

// Follower cap (~12.5% of sets)
static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1

static inline void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int32_t delta = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += delta;
        if(LEADER_B(set)) leaderB_score += delta;
    }

    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Per-set drift towards global gate
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B){ if(mode_confidence[s]<7) mode_confidence[s]++; }
            else             { if(mode_confidence[s]>-8) mode_confidence[s]--; }
        }

        leaderA_score = 0; leaderB_score = 0;
    }
}

static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;
    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    // Enforce follower cap lazily
    if(!follower_is_B[set]){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_is_B[set] = 1;
        num_mode_B_followers++;
    }
    return true;
}

// ============================================================================
// Mode A: Hawkeye-lite (exclusive rrpv_A + tiny SHCT)
// ============================================================================
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS];

// Tiny SHCT (budgeted)
#define SHCT_ENTRIES 512
static uint8_t SHCT_demand[SHCT_ENTRIES];   // 0..15
static uint8_t SHCT_prefetch[SHCT_ENTRIES]; // 0..15
static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC, bool is_pf){
    uint8_t c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    return (c >= 8);
}
static inline void hawkeye_train(uint64_t PC, bool is_pf, bool hit){
    uint8_t &c = is_pf ? SHCT_prefetch[shct_idx(PC)] : SHCT_demand[shct_idx(PC)];
    if(hit){ if(c<15) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;

    // Find maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;

    // Age and retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]==maxRRPV) return w;
    return 0;
}

// ============================================================================
// Mode B: StreamSieve-Decay (exclusive rrpv_B + tiny per-PC tables)
// ============================================================================
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS];          // 3-bit logical
static uint8_t mode_B_hitcnt[LLC_SETS][LLC_WAYS];   // 2-bit (0..3)
static uint8_t mode_B_stream_tag[LLC_SETS][LLC_WAYS]; // 0/1

// Per-PC tiny tables (512 entries)
#define MODE_B_PC_TABLE_SIZE 512
static uint16_t mode_B_pc_last_line[MODE_B_PC_TABLE_SIZE];   // 10b ID (packable)
static uint8_t  mode_B_pc_stride_conf[MODE_B_PC_TABLE_SIZE]; // 2b
static uint8_t  mode_B_pc_freq[MODE_B_PC_TABLE_SIZE];        // 4b TinyLFU
static uint8_t  mode_B_pc_dead[MODE_B_PC_TABLE_SIZE];        // 2b deadness
static uint8_t  mode_B_pc_run_meter[MODE_B_PC_TABLE_SIZE];   // 4b cross-set run
static uint8_t  mode_B_pc_last_set8[MODE_B_PC_TABLE_SIZE];   // 8b last set low8

// Tunables
static constexpr uint8_t MODE_B_HOT_THRESHOLD       = 8; // freq >= 8 => hot
static constexpr uint8_t MODE_B_DEAD_STRONG         = 3; // 2-bit max
static constexpr uint8_t MODE_B_STREAM_ARM          = 2; // need 2 successive ±1/±2
static constexpr uint8_t MODE_B_INSERT_WARM         = 2; // near-MRU for hot
static constexpr uint8_t MODE_B_INSERT_COLD         = 6; // near-tail
static constexpr uint8_t MODE_B_STREAM_TAIL         = 7; // hard tail
static constexpr uint8_t MODE_B_RUN_BYPASS_LEN      = 8; // long-run threshold
static constexpr uint8_t MODE_B_SHORT_RUN_CEIL      = 3; // short-run cutoff
static constexpr uint8_t MODE_B_PROMOTE_NONSTREAM   = 2; // 2nd hit
static constexpr uint8_t MODE_B_PROMOTE_STREAM      = 3; // 3rd hit

static inline uint32_t mode_B_pc_index(uint64_t pc){ return (uint32_t)pc & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t mode_B_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit

// Run/stride detect; also updates cross-set run meter
static bool mode_B_detect_stream_and_runs(uint32_t set, uint64_t PC, uint64_t paddr){
    uint32_t idx = mode_B_pc_index(PC);
    uint16_t line_id = mode_B_line_id(paddr);
    uint16_t last = mode_B_pc_last_line[idx];

    bool step = false;
    if(last != 0xFFFF){
        int16_t diff = (int16_t)(line_id - last);
        step = (diff==1)||(diff==2)||(diff==-1)||(diff==-2);
    }
    if(step) sat_inc_u2(mode_B_pc_stride_conf[idx]);
    else     mode_B_pc_stride_conf[idx] = 0;
    mode_B_pc_last_line[idx] = line_id;

    // Cross-set run meter (use low 8 bits of set index)
    uint8_t cur_set8 = (uint8_t)(set & 0xFF);
    uint8_t prev_set8 = mode_B_pc_last_set8[idx];
    if(prev_set8 != 0xFF){
        if(cur_set8 != prev_set8) sat_inc_u4(mode_B_pc_run_meter[idx]);
        else                      sat_dec_u4(mode_B_pc_run_meter[idx]);
    }
    mode_B_pc_last_set8[idx] = cur_set8;

    return (mode_B_pc_stride_conf[idx] >= MODE_B_STREAM_ARM);
}

// Mode B update: updates ONLY rrpv_B and Mode B metadata
static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t idx = mode_B_pc_index(PC);

    // Frequency + deadness
    if(is_demand(type)) sat_inc_u4(mode_B_pc_freq[idx]);
    if(is_demand(type)){
        if(hit){ mode_B_pc_dead[idx] = 0; }         // hit => alive
        else    { if(mode_B_pc_dead[idx]<3) mode_B_pc_dead[idx]++; }
    }

    // Detect stream and run (also updates pc_last_line, run_meter)
    bool is_stream = mode_B_detect_stream_and_runs(set, PC, paddr);
    mode_B_stream_tag[set][way] = is_stream ? 1 : 0;

    // Update per-line hit count
    if(hit) { if(mode_B_hitcnt[set][way]<3) mode_B_hitcnt[set][way]++; }
    else    { mode_B_hitcnt[set][way] = 0; }

    // Insertion policy (RRIP_B only)
    bool pc_hot   = (mode_B_pc_freq[idx] >= MODE_B_HOT_THRESHOLD);
    bool pc_noisy = (mode_B_pc_dead[idx] >= MODE_B_DEAD_STRONG);
    uint8_t run_m = mode_B_pc_run_meter[idx];

    if(is_prefetch(type)){
        rrpv_B[set][way] = MODE_B_STREAM_TAIL; // quarantine prefetches
    } else if(is_stream){
        // Quarantine streams by default; allow slightly better position if hot and not noisy
        if(pc_hot && !pc_noisy) rrpv_B[set][way] = MODE_B_INSERT_COLD;
        else                    rrpv_B[set][way] = MODE_B_STREAM_TAIL;

        // Long-run pushback: extra aging for the set to hasten eviction of stream clutter
        if(run_m >= MODE_B_RUN_BYPASS_LEN){
            for(uint32_t w=0; w<LLC_WAYS; w++){
                if(rrpv_B[set][w] < maxRRPV) rrpv_B[set][w]++;
            }
            rrpv_B[set][way] = MODE_B_STREAM_TAIL;
        }
    } else {
        // Non-stream: hot inserts warmer, noisy inserts colder
        if(pc_hot && !pc_noisy) rrpv_B[set][way] = MODE_B_INSERT_WARM;
        else                    rrpv_B[set][way] = MODE_B_INSERT_COLD;
    }

    // Multi-hit rescue (demand hits only)
    if(is_demand(type) && hit){
        if(!is_stream && mode_B_hitcnt[set][way] >= MODE_B_PROMOTE_NONSTREAM){
            rrpv_B[set][way] = 0;
        } else if(is_stream && mode_B_hitcnt[set][way] >= MODE_B_PROMOTE_STREAM){
            // Promote only if runs look short (likely short-reuse streams)
            if(mode_B_pc_run_meter[idx] <= MODE_B_SHORT_RUN_CEIL)
                rrpv_B[set][way] = 0;
        }
    }
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;

    // Find maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;

    // Age and retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]==maxRRPV) return w;

    return 0;
}

// ============================================================================
// ChampSim hooks
// ============================================================================
void InitReplacementState(){
    // Mode A/B RRIPs and metadata
    for(uint32_t s=0; s<LLC_SETS; s++){
        mode_confidence[s] = 0;
        follower_is_B[s] = 0;
        for(uint32_t w=0; w<LLC_WAYS; w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_hitcnt[s][w] = 0;
            mode_B_stream_tag[s][w] = 0;
        }
    }

    // Hawkeye-lite predictors
    std::memset(SHCT_demand, 0, sizeof(SHCT_demand));
    std::memset(SHCT_prefetch, 0, sizeof(SHCT_prefetch));

    // Mode B PC tables
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        mode_B_pc_last_line[i] = 0xFFFF;
        mode_B_pc_stride_conf[i] = 0;
        mode_B_pc_freq[i] = 0;
        mode_B_pc_dead[i] = 0;
        mode_B_pc_run_meter[i] = 0;
        mode_B_pc_last_set8[i] = 0xFF;
    }

    leaderA_score = 0; leaderB_score = 0;
    prefer_mode_B = false;
    selector_epoch = 0;
    num_mode_B_followers = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool use_B = should_use_mode_B(set);
    if(use_B) return mode_B_victim(set, current_set);
    else      return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    paddr = (paddr >> 6) << 6;

    // Ignore writebacks
    if(is_writeback(type)) return;

    // Selector accounting
    update_selector(set, hit);

    // Periodic decay for Mode B small tables
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
            mode_B_pc_freq[i] >>= 1;
            mode_B_pc_dead[i] >>= 1;
            mode_B_pc_run_meter[i] >>= 1;
        }
    }

    // Both modes learn their own predictors; EXCLUSIVE RRIP updates
    bool use_B = should_use_mode_B(set);

    if(use_B){
        // Mode B exclusive update
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // Mode A exclusive update (Hawkeye-lite)
        bool friendly = hawkeye_is_friendly(PC, is_prefetch(type));

        // Train SHCT on outcome
        if(is_demand(type) || is_prefetch(type))
            hawkeye_train(PC, is_prefetch(type), hit);

        if(is_prefetch(type)){
            // Prefetches default to averse in Hawkeye-lite
            rrpv_A[set][way] = maxRRPV;
            return;
        }

        if(!friendly){
            rrpv_A[set][way] = maxRRPV;
        } else {
            // Friendly: insert high with slight aging to make room
            for(uint32_t w=0; w<LLC_WAYS; w++){
                if(rrpv_A[set][w] < maxRRPV-1) rrpv_A[set][w]++;
            }
            rrpv_A[set][way] = 0;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}