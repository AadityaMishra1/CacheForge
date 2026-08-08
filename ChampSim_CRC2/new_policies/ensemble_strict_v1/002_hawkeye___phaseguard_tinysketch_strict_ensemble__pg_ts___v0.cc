/*
 * Hawkeye + PhaseGuard-TinySketch (PG-TS)
 * STRICT DYNAMIC ENSEMBLE (EXCLUSIVE STATE PER MODE)
 *
 * Key Tunables (adjust to refine gcc/omnetpp/mcf):
 * - SELECTOR_EPOCH_SIZE = 4096
 * - MODE_A_BIAS = 33, MODE_B_THRESHOLD = +1 (2-bit confidence: -2..+1)
 * - MAX_MODE_B_FOLLOWERS = 256
 * - Mode B inserts: INSERT_WARM=2, INSERT_COLD=6, STREAM_TAIL=7
 * - HOT_THRESHOLD=8 (TinyLFU, 0-15), DEAD_THRESHOLD=2 (0-3)
 * - STREAM_ARM_RUNS=2 (±1/±2), BYPASS_RUN_LEN=8, BYPASS_DENOM=2 (1/2 probability)
 * - Promote: non-stream needs 2 hits (line_seen_once), stream needs “line_seen_once && pc_hitstreak>=2”
 * Telemetry hooks: leader scores, per-set confidence, per-PC freq/dead decay every epoch
 */

#include "../inc/champsim_crc2.h"
#include <cstdint>
#include <vector>
#include <algorithm>

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
#define LLC_SET_BITS 11
#define maxRRPV 7

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;
static inline bool is_demand(uint32_t t){ return (t==ACCESS_LOAD)||(t==ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t==ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t==ACCESS_WRITEBACK); }

// Sampling for leaders (64 sets)
static inline bool SAMPLED_SET(uint32_t set){ return ((set & 63u) == ((set >> (LLC_SET_BITS-6)) & 63u)); }
static inline bool LEADER_A(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set){ return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// CRC32 for compact hashing
static inline uint32_t CRC32(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}

// ------------------------------
// Selector (conservative gating)
// ------------------------------
static int32_t leaderA_score = 0, leaderB_score = 0;
static bool prefer_mode_B = false;
static uint64_t selector_epoch = 0;

static constexpr uint64_t SELECTOR_EPOCH_SIZE = 4096;
static constexpr int32_t MODE_A_BIAS = 33;   // High bias toward Hawkeye
// 2-bit per-set confidence: -2..+1
static int8_t mode_confidence[LLC_SETS];     // store -2..+1 (clamped)
static constexpr int8_t MODE_B_THRESHOLD = 1;

static constexpr uint32_t MAX_MODE_B_FOLLOWERS = 256;
static uint32_t num_mode_B_followers = 0;
static uint8_t follower_is_B[LLC_SETS]; // 0/1

static inline void conf_inc(int8_t &c){ if(c < 1) c++; }
static inline void conf_dec(int8_t &c){ if(c > -2) c--; }

// ------------------------------
// Mode A: Hawkeye-lite (EXCLUSIVE rrpv_A + tiny SHCT)
// ------------------------------
static uint8_t rrpv_A[LLC_SETS][LLC_WAYS]; // 3-bit logical (stored in byte)

// Tiny SHCT (single table for simplicity)
#define SHCT_ENTRIES 256
static uint8_t SHCT_table[SHCT_ENTRIES]; // 4-bit counter, 0..15

static inline uint32_t shct_idx(uint64_t PC){ return CRC32(PC) & (SHCT_ENTRIES-1); }
static inline bool hawkeye_is_friendly(uint64_t PC){
    return (SHCT_table[shct_idx(PC)] >= 8);
}
static inline void hawkeye_train(uint64_t PC, bool hit){
    uint8_t &c = SHCT_table[shct_idx(PC)];
    if(hit){ if(c<15) c++; } else { if(c>0) c--; }
}

static uint32_t mode_A_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    // Look for maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    // Age all and retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w]<maxRRPV) rrpv_A[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_A[set][w] == maxRRPV) return w;
    return 0;
}

// ------------------------------
// Mode B: PhaseGuard-TinySketch (EXCLUSIVE rrpv_B)
// ------------------------------
static uint8_t rrpv_B[LLC_SETS][LLC_WAYS]; // 3-bit logical
// Minimal per-line rescue bit
static uint8_t mode_B_hit_seen[LLC_SETS][LLC_WAYS]; // 0/1

// Tiny per-set "keeper": rescue the last-hit way once
static uint8_t keeper_way[LLC_SETS]; // 0..15
static uint8_t keeper_ttl[LLC_SETS]; // 0/1

// PC TinySketch (256 entries)
#define MODE_B_PC_TABLE_SIZE 256
static uint16_t b_last_line[MODE_B_PC_TABLE_SIZE];  // 10-bit used, 0xFFFF unset
static uint8_t  b_stride_conf[MODE_B_PC_TABLE_SIZE]; // 0..3
static uint8_t  b_freq[MODE_B_PC_TABLE_SIZE];        // 0..15
static uint8_t  b_dead[MODE_B_PC_TABLE_SIZE];        // 0..3
static uint8_t  b_last_set8[MODE_B_PC_TABLE_SIZE];   // 0..255
static uint8_t  b_crossrun[MODE_B_PC_TABLE_SIZE];    // 0..15
static uint8_t  b_hitstreak[MODE_B_PC_TABLE_SIZE];   // 0..3

// Tunables
static constexpr uint8_t INSERT_WARM = 2;
static constexpr uint8_t INSERT_COLD = 6;
static constexpr uint8_t STREAM_TAIL = 7;
static constexpr uint8_t HOT_THRESHOLD = 8;
static constexpr uint8_t DEAD_THRESHOLD = 2;
static constexpr uint8_t STREAM_ARM_RUNS = 2;     // need 2 consecutive ±1/±2 strides
static constexpr uint8_t BYPASS_RUN_LEN = 8;      // long cross-set run
static constexpr uint8_t BYPASS_DENOM = 2;        // 1/2 prob tail-drop

// Helpers
static inline uint32_t b_pc_idx(uint64_t pc){ return CRC32(pc) & (MODE_B_PC_TABLE_SIZE-1); }
static inline uint16_t b_line_id(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x3FF); } // 10-bit
static inline void sat_inc_u2(uint8_t &x){ if(x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if(x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if(x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if(x>0) x--; }

// Stream/run detector (±1/±2)
static inline bool b_detect_stream(uint32_t pcidx, uint64_t paddr){
    uint16_t cur = b_line_id(paddr);
    uint16_t last = b_last_line[pcidx];
    bool stride = false;
    if(last != 0xFFFF){
        int16_t d = (int16_t)cur - (int16_t)last;
        stride = (d==1)||(d==2)||(d==-1)||(d==-2);
    }
    if(stride) sat_inc_u2(b_stride_conf[pcidx]); else b_stride_conf[pcidx]=0;
    b_last_line[pcidx] = cur;
    return (b_stride_conf[pcidx] >= STREAM_ARM_RUNS);
}

// Cross-set run tracking (for probabilistic tail-drop)
static inline void b_update_crossrun(uint32_t pcidx, uint32_t set){
    uint8_t s8 = (uint8_t)(set & 0xFF);
    int d = (int)s8 - (int)b_last_set8[pcidx];
    if((d==1)||(d==-1)) { if(b_crossrun[pcidx]<15) b_crossrun[pcidx]++; }
    else { b_crossrun[pcidx] = 0; }
    b_last_set8[pcidx] = s8;
}

static uint32_t mode_B_victim(uint32_t set, const BLOCK* current_set){
    // Prefer invalid
    for(uint32_t w=0; w<LLC_WAYS; w++) if(!current_set[w].valid) return w;
    // Look for maxRRPV
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV){
        // Keeper veto once
        if(keeper_ttl[set] && (w == keeper_way[set]) && current_set[w].valid){
            // Age all and continue search
            for(uint32_t i=0;i<LLC_WAYS;i++) if(rrpv_B[set][i]<maxRRPV) rrpv_B[set][i]++;
            keeper_ttl[set]=0; // consume veto
            break;
        }
        return w;
    }
    // Age and retry
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w]<maxRRPV) rrpv_B[set][w]++;
    for(uint32_t w=0; w<LLC_WAYS; w++) if(rrpv_B[set][w] == maxRRPV) return w;
    return 0;
}

static void mode_B_update(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit){
    uint32_t pcidx = b_pc_idx(PC);

    // TinyLFU: update on demand
    if(is_demand(type)) sat_inc_u4(b_freq[pcidx]);
    // Deadness: increment on cold alloc miss, decrement on hits
    if(!hit && is_demand(type)) sat_inc_u2(b_dead[pcidx]);
    if(hit) sat_dec_u2(b_dead[pcidx]);

    // Stream/run detection
    bool is_stream = b_detect_stream(pcidx, paddr);
    b_update_crossrun(pcidx, set);

    bool pc_hot  = (b_freq[pcidx] >= HOT_THRESHOLD);
    bool pc_dead = (b_dead[pcidx] >= DEAD_THRESHOLD);

    // Decide insertion depth (exclusive rrpv_B)
    uint8_t ins = INSERT_COLD;
    if(is_prefetch(type)) ins = STREAM_TAIL;                     // quarantine prefetch
    else if(is_stream)    ins = STREAM_TAIL;                     // stream quarantine
    else if(pc_dead)      ins = INSERT_COLD;                     // dead-ish -> cold
    else if(pc_hot)       ins = INSERT_WARM;                     // hot -> warm
    else                  ins = INSERT_COLD;

    // Probabilistic tail-drop during long runs (avoid polluting cache)
    if((b_crossrun[pcidx] >= BYPASS_RUN_LEN) && !pc_hot){
        // 1/BYPASS_DENOM probability
        if((CRC32(paddr) & (BYPASS_DENOM-1)) == 0) ins = STREAM_TAIL;
    }

    rrpv_B[set][way] = ins;

    // Stream-aware multi-hit rescue
    if(hit){
        // Record line seen once
        if(!mode_B_hit_seen[set][way]){
            mode_B_hit_seen[set][way] = 1;
        } else {
            // Non-stream: promote on 2nd line hit
            // Stream: require per-PC hitstreak evidence to be >=2
            if(!is_stream || (b_hitstreak[pcidx] >= 2)){
                rrpv_B[set][way] = 0; // MRU
            }
        }
        // Per-PC hitstreak (approx reuse intensity)
        if(b_hitstreak[pcidx] < 3) b_hitstreak[pcidx]++;
        // Keeper: protect this way once
        keeper_way[set] = (uint8_t)way;
        keeper_ttl[set] = 1;
    } else {
        mode_B_hit_seen[set][way] = 0;
    }
}

// ------------------------------
// Selector mechanics
// ------------------------------
static inline bool should_use_mode_B(uint32_t set){
    if(LEADER_A(set)) return false;
    if(LEADER_B(set)) return true;

    if(!prefer_mode_B) return false;
    if(mode_confidence[set] < MODE_B_THRESHOLD) return false;

    // Enforce follower cap
    if(!follower_is_B[set]){
        if(num_mode_B_followers >= MAX_MODE_B_FOLLOWERS) return false;
        follower_is_B[set] = 1;
        num_mode_B_followers++;
    }
    return true;
}

static void update_selector(uint32_t set, uint8_t hit){
    if(SAMPLED_SET(set)){
        int d = hit ? 1 : -1;
        if(LEADER_A(set)) leaderA_score += d;
        if(LEADER_B(set)) leaderB_score += d;
    }
    selector_epoch++;
    if(selector_epoch % SELECTOR_EPOCH_SIZE == 0){
        // Global gate: Mode B only if decisively better
        prefer_mode_B = (leaderB_score > (leaderA_score + MODE_A_BIAS));

        // Drift per-set confidence toward global preference
        for(uint32_t s=0; s<LLC_SETS; s++){
            if(SAMPLED_SET(s)) continue;
            if(prefer_mode_B) conf_inc(mode_confidence[s]);
            else              conf_dec(mode_confidence[s]);
        }

        // Epochal decay for Mode B TinySketch
        for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
            b_freq[i] = (uint8_t)(b_freq[i] >> 1);
            b_dead[i] = (uint8_t)(b_dead[i] >> 1);
            b_hitstreak[i] = (uint8_t)(b_hitstreak[i] >> 1);
            b_crossrun[i] = (uint8_t)(b_crossrun[i] >> 1);
        }

        // Reset leader scores
        leaderA_score = 0; leaderB_score = 0;
    }
}

// ------------------------------
// ChampSim Hooks
// ------------------------------
void InitReplacementState(){
    for(uint32_t s=0;s<LLC_SETS;s++){
        for(uint32_t w=0;w<LLC_WAYS;w++){
            rrpv_A[s][w] = maxRRPV;
            rrpv_B[s][w] = maxRRPV;
            mode_B_hit_seen[s][w] = 0;
        }
        mode_confidence[s] = 0;
        follower_is_B[s] = 0;
        keeper_way[s] = 0;
        keeper_ttl[s] = 0;
    }
    for(uint32_t i=0;i<SHCT_ENTRIES;i++) SHCT_table[i]=8; // neutral
    for(uint32_t i=0;i<MODE_B_PC_TABLE_SIZE;i++){
        b_last_line[i]=0xFFFF; b_stride_conf[i]=0;
        b_freq[i]=0; b_dead[i]=0; b_last_set8[i]=0;
        b_crossrun[i]=0; b_hitstreak[i]=0;
    }
    leaderA_score=leaderB_score=0;
    prefer_mode_B=false;
    selector_epoch=0;
    num_mode_B_followers=0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
    (void)cpu; (void)PC; (void)paddr; (void)type;
    bool useB = should_use_mode_B(set);
    if(useB) return mode_B_victim(set, current_set);
    else     return mode_A_victim(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
    (void)cpu; (void)victim_addr;
    if(is_writeback(type)) return;

    // Selector bookkeeping
    update_selector(set, hit);

    // Always keep Mode A predictor warm (but never touch rrpv_A if Mode B chosen)
    if(is_demand(type)) hawkeye_train(PC, hit);

    bool useB = should_use_mode_B(set);

    if(useB){
        // MODE B EXCLUSIVE UPDATE
        mode_B_update(set, way, paddr, PC, type, hit);
    } else {
        // MODE A EXCLUSIVE UPDATE
        if(!hawkeye_is_friendly(PC)){
            rrpv_A[set][way] = maxRRPV; // averse -> tail
        } else {
            // friendly -> MRU; slight aging on miss to shape recency stack
            if(!hit){
                for(uint32_t i=0;i<LLC_WAYS;i++) if(rrpv_A[set][i]<maxRRPV-1) rrpv_A[set][i]++;
            }
            rrpv_A[set][way] = 0;
        }
    }
}

void PrintStats_Heartbeat(){}
void PrintStats(){}