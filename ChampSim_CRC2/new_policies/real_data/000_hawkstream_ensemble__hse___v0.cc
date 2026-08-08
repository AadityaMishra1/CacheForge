#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ---------------- Access types (CRC2) ----------------
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
// PC-indexed tables
static constexpr uint8_t  PC_BITS        = 6;                // 64-entry tables
static constexpr uint32_t PC_SIZE        = (1u << PC_BITS);

// Expected-use predictor (EUP) 2-bit counter per PC
static constexpr uint8_t  EUP_INIT       = 1;                // slightly cold
static constexpr uint8_t  EUP_WARM_TH    = 2;                // >=2 => warm

// RunGuard: per-PC stride±1/±2 forward run detector
static constexpr uint8_t  LINE_LOW_BITS  = 12;               // compare low line bits
static constexpr uint8_t  RG_BYPASS_TH   = 2;                // >=2 => treat as stream

// Mode B insertion/promotion
static constexpr uint8_t  AGE_MAX        = 3;                // 0..3, 3=tail
static constexpr uint8_t  AGE_YOUNG      = 1;                // warm insert
static constexpr uint8_t  HITS_TO_PROMOTE= 2;                // promote on 2nd+ demand hit

// Mode A (Hawkeye-like) RRIP
static constexpr uint8_t  RRPV_MAX       = 7;                // 3-bit RRIP (0..7)
static constexpr uint8_t  RRPV_FRIENDLY  = 2;                // young-ish insert
static constexpr uint8_t  RRPV_COLD      = 7;                // hard tail

// Selector: leader/follower, epoch and margin
static constexpr uint32_t EPOCH_LENGTH   = 1u << 14;         // accesses/epoch (tunable)
static constexpr uint32_t SWITCH_MARGIN  = 8;                // min hit delta to switch followers

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline uint32_t pc_index(uint64_t pc) {
    // tiny hash for small PC table
    uint64_t x = pc ^ (pc >> 5) ^ (pc >> 11) ^ (pc >> 17);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
template <typename T> static inline void sat_inc(T &x, T maxv) { if (x < maxv) x++; }
template <typename T> static inline void sat_dec(T &x)        { if (x > 0) x--; }

// ---------------- Mode B per-line state ----------------
static uint8_t AGE   [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3 (age/recency)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];   // 2 bits: 0=NORMAL,1=STREAM,2=QUAR
static uint8_t PCSIG [LLC_SETS][LLC_WAYS];   // 6 bits: PC index

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Mode A (Hawkeye-like) per-line state ----------------
static uint8_t RRPV  [LLC_SETS][LLC_WAYS];   // 3 bits: 0..7

// ---------------- Global predictors (tiny) ----------------
// Expected-use by PC
static uint8_t EUP[PC_SIZE];                 // 2-bit 0..3

// RunGuard per-PC
static uint16_t RG_LAST_LINE[PC_SIZE];       // 12 bits stored in 16
static uint8_t  RG_CONF[PC_SIZE];            // 2 bits (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];      // 1 bit
static uint8_t  RG_LAST_FWD[PC_SIZE];        // 1 bit

// ---------------- Selector: leader/follower ----------------
// 32 A-leaders, 32 B-leaders using a deterministic mapping
static inline bool is_sampled(uint32_t set) {
    // select 64 sets: lower 6 bits equal upper 6-overlap bits (LLC_SETS=2048 => 11 bits, use shift 5)
    return ((set & 0x3F) == ((set >> 5) & 0x3F));
}
static inline bool is_leader_A(uint32_t set) { return is_sampled(set) && ((set & 0x1u) == 0); }
static inline bool is_leader_B(uint32_t set) { return is_sampled(set) && ((set & 0x1u) == 1); }

// Per-set chosen mode bit for followers (0=A, 1=B); leaders ignore this bit
static uint8_t SET_SELECT[LLC_SETS];         // 1 bit used

// Epoch accounting
static uint32_t epoch_accesses = 0;
static uint32_t leader_hits_A  = 0;
static uint32_t leader_hits_B  = 0;
static int32_t  global_bias    = -1;         // start favoring Mode A

// ---------------- RunGuard update ----------------
static inline uint8_t update_runguard(uint32_t pc_idx, uint64_t paddr) {
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[pc_idx]));
    int16_t ad    = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12    = (ad == 1) || (ad == 2);
    bool fwd      = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[pc_idx] && RG_LAST_FWD[pc_idx]) {
            sat_inc<uint8_t>(RG_CONF[pc_idx], 3); // two consecutive fwd ±1/±2
        } else {
            if (RG_CONF[pc_idx] == 0) RG_CONF[pc_idx] = 1;
            RG_LAST_ABS12[pc_idx] = 1;
            RG_LAST_FWD[pc_idx]   = 1;
        }
    } else {
        sat_dec<uint8_t>(RG_CONF[pc_idx]);       // decay when pattern breaks
        RG_LAST_ABS12[pc_idx] = 0;
        RG_LAST_FWD[pc_idx]   = 0;
    }
    RG_LAST_LINE[pc_idx] = curr;
    return RG_CONF[pc_idx];
}

static inline void age_all_B(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
}

// Compare Mode B evictability
static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS[set][a] != ST_NORMAL);
    bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict stream/quarantine first

    uint8_t ea = EUP[ PCSIG[set][a] ];
    uint8_t eb = EUP[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // evict colder PCs

    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb); // fewer observed hits

    return AGE[set][a] > AGE[set][b]; // older age
}

// ---------------- Init ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
            RRPV[s][w]   = RRPV_MAX;
        }
        SET_SELECT[s] = 0; // default followers to Mode A
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]          = EUP_INIT;
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS12[i]= 0;
        RG_LAST_FWD[i]  = 0;
    }
    epoch_accesses = 0;
    leader_hits_A  = 0;
    leader_hits_B  = 0;
    global_bias    = -1; // prefer A initially
}

// ---------------- Mode choice helpers ----------------
static inline bool use_mode_B(uint32_t set) {
    if (is_leader_A(set)) return false;
    if (is_leader_B(set)) return true;
    return (SET_SELECT[set] != 0); // follower uses current selection
}

// ---------------- Victim selection ----------------
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    if (!use_mode_B(set)) {
        // Mode A: RRIP victim (max RRPV or highest)
        uint32_t best = 0;
        uint8_t  best_v = RRPV[set][0];
        if (best_v == RRPV_MAX) return 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            uint8_t v = RRPV[set][w];
            if (v == RRPV_MAX) return w;
            if (v > best_v) { best_v = v; best = w; }
        }
        return best;
    } else {
        // Mode B: composite victim
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable_B(set, w, best)) best = w;
        }
        return best;
    }
}

// ---------------- State update ----------------
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    // Never react to writebacks
    if (type == ACCESS_WRITEBACK) return;

    // Leader accounting (hits only)
    if (hit) {
        if (is_leader_A(set)) leader_hits_A++;
        else if (is_leader_B(set)) leader_hits_B++;
    }

    // Epoch and follower selection
    epoch_accesses++;
    if (epoch_accesses >= EPOCH_LENGTH) {
        // Require margin to switch to B; otherwise default A
        int32_t delta = static_cast<int32_t>(leader_hits_B) - static_cast<int32_t>(leader_hits_A);
        if (delta > static_cast<int32_t>(SWITCH_MARGIN)) global_bias = 1;
        else if (delta < -static_cast<int32_t>(SWITCH_MARGIN)) global_bias = -1;
        // Apply to followers
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!is_leader_A(s) && !is_leader_B(s)) {
                SET_SELECT[s] = (global_bias > 0) ? 1 : 0;
            }
        }
        // Reset epoch
        epoch_accesses = 0;
        leader_hits_A  = 0;
        leader_hits_B  = 0;
    }

    const bool demand = is_demand(type);
    const uint32_t pc_idx = pc_index(PC);

    if (!use_mode_B(set)) {
        // ---------------- Mode A: Hawkeye-like RRIP with PC-biased insertion ----------------
        if (hit) {
            // Demand/prefetch hit => make MRU
            RRPV[set][way] = 0;
        } else {
            // Train on eviction of previous line in this way: penalize low-reuse
            uint8_t old_hits = HITCNT[set][way];
            uint8_t old_pc   = PCSIG [set][way];
            if (old_hits < HITS_TO_PROMOTE) sat_dec<uint8_t>(EUP[old_pc]);

            // Insert new line
            uint8_t ins_v = RRPV_COLD; // default cold
            if (demand && EUP[pc_idx] >= EUP_WARM_TH) ins_v = RRPV_FRIENDLY;
            if (type == ACCESS_PREFETCH) ins_v = RRPV_COLD; // quarantine prefetch
            RRPV[set][way] = ins_v;

            // Also initialize shared fields so Mode B can observe subsequent behavior
            AGE[set][way]    = AGE_MAX;
            HITCNT[set][way] = 0;
            STATUS[set][way] = (type == ACCESS_PREFETCH) ? ST_QUAR : ST_NORMAL;
            PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
        }
    } else {
        // ---------------- Mode B: Stream+Dead-PC Boost ----------------
        // Age set mildly to maintain recency ordering
        age_all_B(set);

        // Update RunGuard on demand
        uint8_t rg_conf = 0;
        if (demand) rg_conf = update_runguard(pc_idx, paddr);
        const bool stream_like = (rg_conf >= RG_BYPASS_TH);

        if (hit) {
            if (demand) {
                // Count demand hits, gate promotion until 2nd hit
                if (HITCNT[set][way] < HITS_TO_PROMOTE) HITCNT[set][way]++;
                if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                    AGE[set][way] = 0;             // make MRU
                    STATUS[set][way] = ST_NORMAL;  // clear stream/quarantine after proven reuse
                    // Reinforce the PC as warm
                    sat_inc<uint8_t>(EUP[ PCSIG[set][way] ], 3);
                }
            } else {
                // Prefetch hit: do not promote unless already proven reuser
                if (HITCNT[set][way] >= HITS_TO_PROMOTE) AGE[set][way] = 0;
            }
        } else {
            // Miss/fill: train on the evicted line before overwrite
            uint8_t old_hits = HITCNT[set][way];
            uint8_t old_pc   = PCSIG [set][way];
            if (old_hits < HITS_TO_PROMOTE) sat_dec<uint8_t>(EUP[old_pc]);

            // Initialize new line metadata
            PCSIG[set][way]  = static_cast<uint8_t>(pc_idx);
            HITCNT[set][way] = 0;

            if (type == ACCESS_PREFETCH) {
                STATUS[set][way] = ST_QUAR;           // quarantine prefetches
                AGE[set][way]    = AGE_MAX;
            } else if (stream_like) {
                STATUS[set][way] = ST_STREAM;         // stream near-bypass
                AGE[set][way]    = AGE_MAX;           // hard tail
            } else {
                STATUS[set][way] = ST_NORMAL;
                AGE[set][way]    = (EUP[pc_idx] >= EUP_WARM_TH) ? AGE_YOUNG : AGE_MAX;
            }

            // Keep RRIP coherent enough for Mode A fallback
            RRPV[set][way] = (STATUS[set][way] == ST_NORMAL && EUP[pc_idx] >= EUP_WARM_TH) ? RRPV_FRIENDLY : RRPV_COLD;
        }
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}