#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables ----------------
static constexpr uint8_t  PC_BITS            = 8;     // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE            = (1u << PC_BITS);

static constexpr uint8_t  LINE_LOW_BITS      = 12;    // for stream run detection
static constexpr uint8_t  STREAM_TH          = 2;     // >=2 => near-bypass/quarantine

static constexpr uint8_t  HITS_TO_PROMOTE    = 2;     // second+ demand hit promotes
static constexpr uint8_t  WARM_INIT          = 1;     // PC warmth start (0..3)
static constexpr uint8_t  WARM_FRIENDLY_TH   = 2;     // >=2 => insert young

// Epoch arithmetic (2-bit per-set epoch, 2-bit per-line fill epoch)
static constexpr uint8_t  EPOCH_MOD          = 4;

// Revival Sketch (ghost of recent evictions/bypasses)
static constexpr uint32_t GHOST_IDX_BITS     = 10;    // 1024 entries
static constexpr uint32_t GHOST_SIZE         = (1u << GHOST_IDX_BITS);
static constexpr uint8_t  GHOST_TAG_BITS     = 12;    // compact tag

// ---------------- Per-line state ----------------
// 2-bit fill epoch (mod-4)
static uint8_t FILL_EPOCH[LLC_SETS][LLC_WAYS];
// 2-bit hit counter: 0,1,2(=2+)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];
// 1-bit sentinel: 1 => stream/quarantine near-bypass, deprioritized for promotion
static uint8_t SENTINEL[LLC_SETS][LLC_WAYS];
// 8-bit PC signature index (PC_BITS)
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];
// 1-bit valid shadow (to safely train on eviction)
static uint8_t VALID_SH[LLC_SETS][LLC_WAYS];

// ---------------- Per-set state ----------------
static uint8_t SET_EPOCH[LLC_SETS]; // 2-bit rolling epoch

// ---------------- Global predictors ----------------
// PC warmth: 2-bit per PC
static uint8_t WARM[PC_SIZE];

// StreamShield per-PC run detector: last line low, conf, last flags
static uint16_t RG_LAST_LINE[PC_SIZE]; // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];      // 2-bit conf (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];// 1-bit
static uint8_t  RG_LAST_FWD[PC_SIZE];  // 1-bit

// Revival Sketch: direct-mapped small ghost
static uint16_t G_TAG[GHOST_SIZE];     // GHOST_TAG_BITS tag
static uint8_t  G_VAL[GHOST_SIZE];     // valid bit

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline uint32_t pc_index(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint64_t line_addr(uint64_t paddr) {
    return (paddr >> 6); // 64B lines
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>(line_addr(paddr) & ((1u << LINE_LOW_BITS) - 1));
}
static inline uint32_t ghost_index(uint64_t paddr) {
    uint64_t la = line_addr(paddr);
    uint64_t h = la ^ (la >> 11) ^ (la >> 23);
    return static_cast<uint32_t>(h) & (GHOST_SIZE - 1);
}
static inline uint16_t ghost_tag(uint64_t paddr) {
    uint64_t la = line_addr(paddr);
    uint64_t h = la ^ (la >> 17) ^ (la >> 29);
    return static_cast<uint16_t>(h) & ((1u << GHOST_TAG_BITS) - 1);
}
static inline uint8_t epoch_age(uint8_t set_epoch, uint8_t fill_epoch) {
    return static_cast<uint8_t>((set_epoch - fill_epoch) & (EPOCH_MOD - 1));
}

// Update StreamShield and return confidence (0..3); update only on demand
static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    uint16_t prev = RG_LAST_LINE[idx];
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(prev));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3);
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Eviction comparator: true if (a) is more evictable than (b)
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Evict sentinels first
    if (SENTINEL[set][a] != SENTINEL[set][b]) return SENTINEL[set][a] > SENTINEL[set][b];

    // 2) PCs predicted colder first
    uint8_t wa = WARM[ PCSIG[set][a] ];
    uint8_t wb = WARM[ PCSIG[set][b] ];
    if (wa != wb) return wa < wb;

    // 3) Fewer observed demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return ha < hb;

    // 4) Older epoch
    uint8_t age_a = epoch_age(SET_EPOCH[set], FILL_EPOCH[set][a]);
    uint8_t age_b = epoch_age(SET_EPOCH[set], FILL_EPOCH[set][b]);
    if (age_a != age_b) return age_a > age_b;

    // 5) Tie-break by way index (deterministic)
    return a > b;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        SET_EPOCH[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            FILL_EPOCH[s][w] = 0;
            HITCNT[s][w]     = 0;
            SENTINEL[s][w]   = 0;
            PCSIG[s][w]      = 0;
            VALID_SH[s][w]   = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        WARM[i]          = WARM_INIT;
        RG_LAST_LINE[i]  = 0;
        RG_CONF[i]       = 0;
        RG_LAST_ABS12[i] = 0;
        RG_LAST_FWD[i]   = 0;
    }
    for (uint32_t i = 0; i < GHOST_SIZE; i++) {
        G_TAG[i] = 0;
        G_VAL[i] = 0;
    }
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return an invalid way if available
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Choose most-evictable way
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, best)) best = w;
    }
    return best;
}

// Update replacement state
void UpdateReplacementState(
    uint32_t /*cpu*/,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);

    // StreamShield update on all demand accesses
    uint8_t stream_conf = 0;
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    if (hit) {
        // Demand hit handling with multi-hit promotion
        if (demand) {
            if (HITCNT[set][way] < 2) HITCNT[set][way]++; // count only demand hits
            if ((HITCNT[set][way] >= HITS_TO_PROMOTE)) {
                // Promote: clear sentinel and reset epoch to youngest
                SENTINEL[set][way] = 0;
                FILL_EPOCH[set][way] = SET_EPOCH[set];
                // Reward warm PC
                sat_inc(WARM[ PCSIG[set][way] ], 3);
            }
        }
        // No promotion on prefetch-only hits (implicitly enforced)
        return;
    }

    // Miss: train previous line in this way if it was valid
    if (VALID_SH[set][way]) {
        uint8_t prev_pc = PCSIG[set][way];
        uint8_t prev_hits = HITCNT[set][way];
        if (prev_hits >= HITS_TO_PROMOTE) sat_inc(WARM[prev_pc], 3);
        else sat_dec(WARM[prev_pc]);

        // Insert evicted line into ghost
        uint32_t gi = ghost_index(victim_addr);
        G_TAG[gi] = ghost_tag(victim_addr);
        G_VAL[gi] = 1;
    }

    // Decide insertion policy for new line
    uint32_t pc_idx = pc_index(PC);
    bool ghost_hit = false;
    {
        uint32_t gi = ghost_index(paddr);
        if (G_VAL[gi] && (G_TAG[gi] == ghost_tag(paddr))) {
            ghost_hit = true;
            // Nudge PC warmer on revival
            sat_inc(WARM[pc_idx], 3);
            G_VAL[gi] = 0; // consume entry to avoid persistent bias
        }
    }

    // Classify
    bool streamy = demand && (stream_conf >= STREAM_TH);
    bool is_pref = (type == ACCESS_PREFETCH);
    bool warm_pc = (WARM[pc_idx] >= WARM_FRIENDLY_TH);

    // Write new line metadata
    PCSIG[set][way]    = static_cast<uint8_t>(pc_idx);
    HITCNT[set][way]   = 0;             // demand-only counter
    VALID_SH[set][way] = 1;

    // Insertion depth via epochs (younger => current epoch)
    uint8_t cur_epoch = SET_EPOCH[set];
    uint8_t ins_epoch;

    if (is_pref) {
        SENTINEL[set][way] = 1;         // quarantine prefetches
        ins_epoch = static_cast<uint8_t>((cur_epoch - 3) & (EPOCH_MOD - 1));
    } else if (streamy) {
        SENTINEL[set][way] = 1;         // near-bypass for stream
        ins_epoch = static_cast<uint8_t>((cur_epoch - 3) & (EPOCH_MOD - 1));
    } else if (ghost_hit) {
        SENTINEL[set][way] = 0;         // revive young
        ins_epoch = cur_epoch;
    } else if (warm_pc) {
        SENTINEL[set][way] = 0;         // friendly PC inserts young
        ins_epoch = cur_epoch;
    } else {
        SENTINEL[set][way] = 0;         // slightly old for cold PCs
        ins_epoch = static_cast<uint8_t>((cur_epoch - 2) & (EPOCH_MOD - 1));
    }

    FILL_EPOCH[set][way] = ins_epoch;

    // Advance set epoch after each fill (both demand and prefetch and writeback)
    SET_EPOCH[set] = static_cast<uint8_t>((SET_EPOCH[set] + 1) & (EPOCH_MOD - 1));
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}