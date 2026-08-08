#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

// HESG: HawkEnsemble-StreamGuard
// - Mode A: Hawkeye-like RRIP-friendly path (simple RRIP core with PC-expected-use guided insertion).
// - Mode B: StreamGuard-MHP with PC Expected-Use + RunGuard stream detector and multi-hit promotion.
// - Selector: set-dueling leaders and a global epoch bandit; followers default to Mode A unless B leaders win.

// -------------------- Configuration (tunables) --------------------
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// RRIP
static constexpr uint8_t RRPV_MAX = 7;         // 3-bit RRIP

// PC-indexing
static constexpr uint8_t PC_BITS   = 8;        // 256-entry PC tables
static constexpr uint32_t PC_SIZE  = (1u << PC_BITS);

// Expected-Use Predictor (EUP), 2-bit 0..3; >=2 => warm
static constexpr uint8_t EUP_INIT   = 1;
static constexpr uint8_t EUP_WARM_TH= 2;

// RunGuard: stream detector on ±1/±2 forward runs (2-step confidence)
static constexpr uint8_t LINE_LOW_BITS = 12;
static constexpr uint8_t RG_CONF_TH = 2;       // >=2 => stream

// Multi-hit promotion gate (only 2nd+ demand hit promotes to MRU)
static constexpr uint8_t HITS_TO_PROMOTE = 2;

// Selector: leaders and epoch
static constexpr uint32_t EPOCH_LENGTH = 8192; // accesses per decision epoch
static constexpr uint32_t HYSTERESIS   = 16;   // miss margin to switch to Mode B

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// Status encodings
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// -------------------- Replacement metadata --------------------
// Shared per-line state (counted tightly in storage accounting)
static uint8_t rrpv   [LLC_SETS][LLC_WAYS];  // 3 bits logical (0..7)
static uint8_t hitcnt [LLC_SETS][LLC_WAYS];  // 2 bits logical (0,1,2=2+)
static uint8_t status [LLC_SETS][LLC_WAYS];  // 2 bits logical (ST_*)
static uint8_t pcsig  [LLC_SETS][LLC_WAYS];  // 8-bit PC index (hashed)

// PC Expected-Use predictor
static uint8_t EUP[PC_SIZE];                 // 2-bit counters

// RunGuard per-PC state (packed in storage accounting)
static uint16_t rg_last_line[PC_SIZE];       // low LINE_LOW_BITS of line number
static uint8_t  rg_conf[PC_SIZE];            // 2-bit (0..3)
static uint8_t  rg_last_abs12[PC_SIZE];      // 1-bit
static uint8_t  rg_last_fwd[PC_SIZE];        // 1-bit

// Set-dueling selector
static uint8_t set_mode[LLC_SETS];           // 0=Mode A (Hawk), 1=Mode B (StreamGuard)
static uint8_t global_mode;                  // followers use this (0=A default)

// Leader miss counters + epoch
static uint32_t leaderA_misses = 0;
static uint32_t leaderB_misses = 0;
static uint32_t global_tick    = 0;

// -------------------- Helpers --------------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix to 8 bits
    uint64_t x = pc ^ (pc >> 17) ^ (pc >> 37);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline bool is_leader_A(uint32_t set) { return ((set & 0x3Fu) == 0u); }
static inline bool is_leader_B(uint32_t set) { return ((set & 0x3Fu) == 1u); }

static inline bool select_modeB(uint32_t set) {
    if (is_leader_A(set)) return false;
    if (is_leader_B(set)) return true;
    return (set_mode[set] ? true : (global_mode != 0));
}

// Update RunGuard on demand access; returns new confidence
static inline uint8_t update_runguard(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(rg_last_line[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (rg_last_abs12[idx] && rg_last_fwd[idx]) {
            sat_inc(rg_conf[idx], 3); // two consecutive forward steps
        } else {
            if (rg_conf[idx] == 0) rg_conf[idx] = 1;
            rg_last_abs12[idx] = 1;
            rg_last_fwd[idx]   = 1;
        }
    } else {
        sat_dec(rg_conf[idx]); // decay on break/back/irregular
        rg_last_abs12[idx] = 0;
        rg_last_fwd[idx]   = 0;
    }
    rg_last_line[idx] = curr;
    return rg_conf[idx];
}

// StreamGuard victim comparison: prefer evicting sentinels, then colder PCs, then fewer hits, then older (higher RRPV)
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (status[set][a] != ST_NORMAL);
    bool b_s = (status[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s; // evict sentinel first

    uint8_t ea = EUP[ pcsig[set][a] ];
    uint8_t eb = EUP[ pcsig[set][b] ];
    if (ea != eb) return (ea < eb); // colder PC first

    uint8_t ha = hitcnt[set][a];
    uint8_t hb = hitcnt[set][b];
    if (ha != hb) return (ha < hb); // fewer hits first

    return rrpv[set][a] > rrpv[set][b]; // older first
}

// -------------------- CRC2 Hooks --------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w]   = RRPV_MAX; // old by default
            hitcnt[s][w] = 0;
            status[s][w] = ST_NORMAL;
            pcsig[s][w]  = 0;
        }
        set_mode[s] = 0; // default Mode A
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        EUP[i]          = EUP_INIT;
        rg_last_line[i] = 0;
        rg_conf[i]      = 0;
        rg_last_abs12[i]= 0;
        rg_last_fwd[i]  = 0;
    }
    global_mode     = 0; // default to Mode A
    leaderA_misses  = 0;
    leaderB_misses  = 0;
    global_tick     = 0;

    std::cout << "Initialize HESG (HawkEnsemble-StreamGuard)" << std::endl;
}

// If any invalid way exists, return it immediately; otherwise pick victim per active mode.
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    const bool useB = select_modeB(set);

    if (useB) {
        uint32_t best = 0;
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (more_evictable(set, w, best)) best = w;
        }
        return best;
    } else {
        // Mode A: Hawkeye-like RRIP victim: prefer lines at RRPV_MAX; else highest RRPV
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == RRPV_MAX) return w;
        }
        uint32_t best = 0;
        uint8_t  best_rrpv = rrpv[set][0];
        for (uint32_t w = 1; w < LLC_WAYS; w++) {
            if (rrpv[set][w] >= best_rrpv) { best_rrpv = rrpv[set][w]; best = w; }
        }
        return best;
    }
}

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
    // Ignore writebacks
    if (type == ACCESS_WRITEBACK) return;

    // Epoch accounting + leader miss tracking
    if (!hit) {
        if (is_leader_A(set)) leaderA_misses++;
        else if (is_leader_B(set)) leaderB_misses++;
    }
    global_tick++;
    if (global_tick % EPOCH_LENGTH == 0) {
        // Update global winner; followers adopt global_mode; leaders remain fixed
        if (leaderB_misses + HYSTERESIS < leaderA_misses) global_mode = 1; else global_mode = 0;
        leaderA_misses = leaderB_misses = 0;
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (!is_leader_A(s) && !is_leader_B(s)) set_mode[s] = global_mode;
        }
    }

    const bool demand = is_demand(type);
    const bool useB = select_modeB(set);
    const uint32_t pc_idx = pc_index(PC);

    // Update RunGuard on demand accesses (helps next insertion)
    if (demand) (void)update_runguard(PC, paddr);

    // On hit: apply mode-specific promotions
    if (hit) {
        // Multi-hit promotion discipline
        if (demand) {
            if (hitcnt[set][way] < 2) hitcnt[set][way]++;
            if (hitcnt[set][way] >= HITS_TO_PROMOTE) {
                // Promotion on 2nd+ demand hit
                rrpv[set][way] = 0;
                if (status[set][way] != ST_NORMAL) status[set][way] = ST_NORMAL; // shed sentinels
                // Positive reinforcement for PC
                sat_inc(EUP[ pcsig[set][way] ], 3);
            } else {
                // First demand hit: do not promote to MRU
                if (rrpv[set][way] > 0) rrpv[set][way]--; // mild benefit without MRU
            }
        } else {
            // Prefetch hit: never promote on prefetch
            // Keep quarantine/stream sentinels intact
        }
        return;
    }

    // Miss path: train on the line being evicted from 'way' (old metadata), then initialize new line
    // Train EUP based on observed reuse of the evicted line (if it was valid before)
    {
        uint8_t old_pc = pcsig[set][way];
        uint8_t old_hits = hitcnt[set][way];
        if (old_pc || old_hits) {
            if (old_hits >= HITS_TO_PROMOTE) sat_inc(EUP[old_pc], 3);
            else sat_dec(EUP[old_pc]);
        }
    }

    // Initialize new line metadata
    pcsig[set][way]  = static_cast<uint8_t>(pc_idx);
    hitcnt[set][way] = 0;

    if (type == ACCESS_PREFETCH) {
        // Always quarantine prefetches at the hard tail
        status[set][way] = ST_QUAR;
        rrpv[set][way]   = RRPV_MAX;
        return;
    }

    // Demand insertion policy per mode
    if (useB) {
        // Mode B: StreamGuard-MHP
        bool stream = (rg_conf[pc_idx] >= RG_CONF_TH);
        if (stream) {
            status[set][way] = ST_STREAM;
            rrpv[set][way]   = RRPV_MAX;   // near-bypass at tail; no early promote
        } else {
            status[set][way] = ST_NORMAL;
            // PC Expected-Use guided insertion
            if (EUP[pc_idx] >= EUP_WARM_TH) rrpv[set][way] = 2; // slightly young
            else                            rrpv[set][way] = 6; // cold PCs: old
        }
    } else {
        // Mode A: Hawkeye-like RRIP-friendly insertion (using EUP as friendliness proxy)
        status[set][way] = ST_NORMAL;
        if (EUP[pc_idx] >= EUP_WARM_TH) rrpv[set][way] = 2; // friendly PC inserts young
        else                            rrpv[set][way] = 6; // averse PC inserts old
    }
}

void PrintStats() {}
void PrintStats_Heartbeat() {}