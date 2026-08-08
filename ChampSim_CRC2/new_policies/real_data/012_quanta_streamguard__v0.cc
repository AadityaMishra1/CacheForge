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

// ---------------- Tunables (knobs) ----------------
static constexpr uint8_t  PC_BITS         = 8;                 // 256-entry PC-indexed tables
static constexpr uint32_t PC_SIZE         = (1u << PC_BITS);

// PC coldness predictor: 2-bit (0..3)
static constexpr uint8_t  COLD_INIT       = 1;                 // slightly cold
static constexpr uint8_t  WARM_TH         = 2;                 // >=2 => warm (insert young)

// StreamGuard: per-PC forward run detector (±1/±2), two consecutive forward steps
static constexpr uint8_t  LINE_LOW_BITS   = 12;                // compare low bits of line#
static constexpr uint8_t  STREAM_BYP_TH   = 2;                 // >=2 => near-bypass

// Age field (small recency): 0=MRU .. 3=LRU
static constexpr uint8_t  AGE_MAX         = 3;
static constexpr uint8_t  AGE_YOUNG       = 1;                 // warm-PC insertion depth

// Multi-hit promotion gate
static constexpr uint8_t  HITS_TO_PROMOTE = 2;                 // promote on 2nd+ demand hit

// Per-line status
enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Per-line state (compact) ----------------
static uint8_t  AGE    [LLC_SETS][LLC_WAYS];   // 2 bits (0..3)
static uint8_t  HITCNT [LLC_SETS][LLC_WAYS];   // 0,1,2(=2+)
static uint8_t  STATUS [LLC_SETS][LLC_WAYS];   // ST_*
static uint8_t  PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index

// ---------------- Global predictors ----------------
// PC coldness (2-bit saturating) and StreamGuard (PC-indexed)
static uint8_t  PC_COLD[PC_SIZE];              // 2-bit (0..3)

static uint16_t RG_LAST_LINE[PC_SIZE];         // low LINE_LOW_BITS of line number
static uint8_t  RG_CONF[PC_SIZE];              // 2-bit confidence (0..3)
static uint8_t  RG_LAST_ABS12[PC_SIZE];        // 1-bit: last step had |delta| in {1,2}
static uint8_t  RG_LAST_FWD[PC_SIZE];          // 1-bit: last step forward

// ---------------- Helpers ----------------
static inline uint32_t pc_index(uint64_t pc) {
    // simple 8-bit hash
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec(uint8_t &x) { if (x > 0) x--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}

// Update StreamGuard on demand access; returns updated confidence
static inline uint8_t update_stream_guard(uint64_t pc, uint64_t paddr) {
    const uint32_t idx = pc_index(pc);
    const uint16_t curr = line_lowN(paddr);
    const int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE[idx]));
    const int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    const bool abs12 = (ad == 1) || (ad == 2);
    const bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12[idx] && RG_LAST_FWD[idx]) {
            sat_inc(RG_CONF[idx], 3);     // two consecutive forward ±1/±2 confirms streaming
        } else {
            if (RG_CONF[idx] == 0) RG_CONF[idx] = 1;
            RG_LAST_ABS12[idx] = 1;
            RG_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec(RG_CONF[idx]);            // decay on break/backward/irregular
        RG_LAST_ABS12[idx] = 0;
        RG_LAST_FWD[idx]   = 0;
    }
    RG_LAST_LINE[idx] = curr;
    return RG_CONF[idx];
}

// Eviction comparator
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting sentinels (streams/prefetch quarantine)
    const bool a_s = (STATUS[set][a] != ST_NORMAL);
    const bool b_s = (STATUS[set][b] != ST_NORMAL);
    if (a_s != b_s) return a_s;

    // 2) Prefer PCs predicted colder
    const uint8_t ca = PC_COLD[ PCSIG[set][a] ];
    const uint8_t cb = PC_COLD[ PCSIG[set][b] ];
    if (ca != cb) return (ca < cb);

    // 3) Prefer lower observed hits (0 < 1 < 2+)
    const uint8_t ha = HITCNT[set][a];
    const uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older more evictable)
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            STATUS[s][w] = ST_NORMAL;
            PCSIG[s][w]  = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_COLD[i]      = COLD_INIT;
        RG_LAST_LINE[i] = 0;
        RG_CONF[i]      = 0;
        RG_LAST_ABS12[i]= 0;
        RG_LAST_FWD[i]  = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Choose most evictable way by composite priority
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
    uint64_t /*victim_addr*/,
    uint32_t type,
    uint8_t hit
) {
    const bool demand = is_demand(type);

    // Update stream detector on demand accesses
    uint8_t stream_conf = 0;
    if (demand) stream_conf = update_stream_guard(PC, paddr);

    // Age the set each access
    age_all(set);

    if (hit) {
        // Demand hit bookkeeping and gated promotion
        if (demand) {
            const uint8_t prev = HITCNT[set][way];
            if (HITCNT[set][way] < 2) HITCNT[set][way]++;

            // Reinforce PC warmness when real reuse emerges
            if (prev < 2 && HITCNT[set][way] >= 2) sat_inc(PC_COLD[ pc_index(PC) ], 3);

            // Streaming/quarantine lines only promote after 2+ demand hits
            if (HITCNT[set][way] >= HITS_TO_PROMOTE) {
                AGE[set][way] = 0;                // MRU
                if (STATUS[set][way] == ST_STREAM || STATUS[set][way] == ST_QUAR)
                    STATUS[set][way] = ST_NORMAL; // shed sentinel after proven reuse
            }
        }
        // Prefetch and writeback hits: do not promote
        return;
    }

    // Miss path: train on the evicted line before overwrite (best-effort)
    {
        const uint8_t ev_pcidx = PCSIG[set][way];
        const uint8_t ev_hits  = HITCNT[set][way];
        // Heuristic to avoid training on uninitialized lines: require any activity signature
        if (ev_pcidx != 0 || ev_hits != 0 || STATUS[set][way] != ST_NORMAL) {
            if (ev_hits >= 2) sat_inc(PC_COLD[ev_pcidx], 3);
            else              sat_dec(PC_COLD[ev_pcidx]);
        }
    }

    // Decide insertion for the new line
    const uint32_t pcidx = pc_index(PC);

    if (type == ACCESS_WRITEBACK) {
        // Never bypass writebacks; quarantine at tail so they self-evict if unused
        AGE[set][way]     = AGE_MAX;
        STATUS[set][way]  = ST_QUAR;
        HITCNT[set][way]  = 0;
        PCSIG[set][way]   = static_cast<uint8_t>(pcidx);
        return;
    }

    if (type == ACCESS_PREFETCH) {
        // Prefetches always insert at hard tail under quarantine
        AGE[set][way]     = AGE_MAX;
        STATUS[set][way]  = ST_QUAR;
        HITCNT[set][way]  = 0;
        PCSIG[set][way]   = static_cast<uint8_t>(pcidx);
        return;
    }

    // Demand fill
    const bool streamish = (stream_conf >= STREAM_BYP_TH);
    if (streamish) {
        // Near-bypass for streaming demand fills: hard tail, stream sentinel, no promote
        AGE[set][way]     = AGE_MAX;
        STATUS[set][way]  = ST_STREAM;
        HITCNT[set][way]  = 0;
        PCSIG[set][way]   = static_cast<uint8_t>(pcidx);
        return;
    }

    // Non-stream demand: insertion guided by PC coldness
    const bool warm_pc = (PC_COLD[pcidx] >= WARM_TH);
    AGE[set][way]     = warm_pc ? AGE_YOUNG : AGE_MAX;
    STATUS[set][way]  = ST_NORMAL;
    HITCNT[set][way]  = 0;
    PCSIG[set][way]   = static_cast<uint8_t>(pcidx);
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}