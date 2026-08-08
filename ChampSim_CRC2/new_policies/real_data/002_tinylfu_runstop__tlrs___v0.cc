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
// Per-line age (0=young/MRU .. 3=old/LRU)
static constexpr uint8_t  AGE_MAX          = 3;
static constexpr uint8_t  AGE_YOUNG        = 1;

// Multi-hit promotion gate (promote only on 2nd+ demand hit)
static constexpr uint8_t  HITS_TO_PROMOTE  = 2;  // 2 promotes; 1st hit never promotes

// RunStop stream detector (per-PC): stride ±1/±2, two consecutive forward steps => streaming
static constexpr uint8_t  LINE_LOW_BITS    = 12; // compare low bits of line number
static constexpr uint8_t  STREAM_CONF_TH   = 2;  // >=2 => stream near-bypass

// TinyLFU admission (count-min sketch on PC⊕region)
// 3 rows x 1024 counters x 6-bit each; periodic halving
static constexpr uint8_t  CMS_ROWS         = 3;
static constexpr uint32_t CMS_SIZE         = 1024; // power of two
static constexpr uint8_t  CMS_BITS         = 6;    // 0..63
static constexpr uint8_t  CMS_FREQ_TH      = 2;    // <2 considered cold for admission

// Tiny per-PC frequency (eviction bias)
static constexpr uint8_t  PC_BITS          = 8;         // 256-entry tables
static constexpr uint32_t PC_SIZE          = (1u << PC_BITS);
static constexpr uint8_t  PCF_BITS         = 3;         // 0..7
static constexpr uint8_t  PCF_WARM_TH      = 2;         // >=2 => warm

// Global decay cadence (both CMS + per-PC freq)
static constexpr uint32_t DECAY_LOG2       = 13;        // every 8192 demand updates

// ---------------- Per-line state (compact; report minimal bits in storage section) ----------------
static uint8_t AGE   [LLC_SETS][LLC_WAYS];   // 2 bits (0..3)
static uint8_t HITCT [LLC_SETS][LLC_WAYS];   // 2 bits: 0,1,2(=2+)
static uint8_t STAT  [LLC_SETS][LLC_WAYS];   // 2 bits: 0=normal,1=stream,2=quarantine
static uint8_t PCSIG [LLC_SETS][LLC_WAYS];   // 8-bit pc index

enum : uint8_t { ST_NORMAL=0, ST_STREAM=1, ST_QUAR=2 };

// ---------------- Global predictors ----------------
// Count-Min Sketch (TinyLFU)
static uint8_t CMS[CMS_ROWS][CMS_SIZE];      // 6-bit conceptual (0..63)

// Per-PC RunStop state and tiny frequency
static uint16_t RS_LAST_LINE[PC_SIZE];       // low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];            // 0..3
static uint8_t  RS_LAST_ABS12[PC_SIZE];      // 0/1
static uint8_t  RS_LAST_FWD[PC_SIZE];        // 0/1
static uint8_t  PC_FREQ[PC_SIZE];            // 0..7 (3-bit)

// Global tick for periodic decay
static uint32_t G_TICKS = 0;

// ---------------- Helpers ----------------
static inline void sat_inc_u8(uint8_t &v, uint8_t maxv) { if (v < maxv) v++; }
static inline void sat_dec_u8(uint8_t &v) { if (v > 0) v--; }

static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // Simple mix to spread low bits
    uint64_t x = pc ^ (pc >> 13) ^ (pc << 7);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline uint64_t region_id(uint64_t paddr) { // 4KB region
    return (paddr >> 12);
}

// 64-bit mix/hash
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
static inline uint32_t cms_index(uint64_t key, uint32_t row) {
    static constexpr uint64_t SALT[CMS_ROWS] = {
        0x9e3779b97f4a7c15ULL, 0xc2b2ae3d27d4eb4fULL, 0x165667b19e3779f9ULL
    };
    uint64_t h = mix64(key ^ SALT[row]);
    return static_cast<uint32_t>(h) & (CMS_SIZE - 1);
}
static inline uint8_t cms_query(uint64_t key) {
    uint8_t m = 0xFF;
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index(key, r);
        uint8_t val = CMS[r][idx];
        if (val < m) m = val;
    }
    return (m == 0xFF) ? 0 : m;
}
static inline void cms_increment(uint64_t key) {
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index(key, r);
        if (CMS[r][idx] < ((1u << CMS_BITS) - 1)) CMS[r][idx]++;
    }
}
static inline void decay_all() {
    // Decay per-PC frequency
    for (uint32_t i = 0; i < PC_SIZE; i++) PC_FREQ[i] >>= 1;
    // Decay CMS
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_SIZE; i++) CMS[r][i] >>= 1;
    }
}

// Update RunStop on demand access; returns updated confidence
static inline uint8_t update_runstop(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc_u8(RS_CONF[idx], 3); // two consecutive forward small strides
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u8(RS_CONF[idx]);        // decay on irregular/backward/large stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

// Compare candidates: true if a is more evictable than b
static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Prefer evicting stream/quarantine sentinels
    bool sa = (STAT[set][a] != ST_NORMAL);
    bool sb = (STAT[set][b] != ST_NORMAL);
    if (sa != sb) return sa;

    // 2) Prefer lower per-PC frequency (PC-friendly lines stay)
    uint8_t pfa = PC_FREQ[ PCSIG[set][a] ];
    uint8_t pfb = PC_FREQ[ PCSIG[set][b] ];
    if (pfa != pfb) return (pfa < pfb);

    // 3) Prefer fewer observed demand hits (0 < 1 < 2+)
    uint8_t ha = HITCT[set][a];
    uint8_t hb = HITCT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Tie-break by age (older more evictable)
    return AGE[set][a] > AGE[set][b];
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]   = AGE_MAX;
            HITCT[s][w] = 0;
            STAT[s][w]  = ST_NORMAL;
            PCSIG[s][w] = 0;
        }
    }
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_SIZE; i++) CMS[r][i] = 0;
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        RS_LAST_LINE[i]  = 0;
        RS_CONF[i]       = 0;
        RS_LAST_ABS12[i] = 0;
        RS_LAST_FWD[i]   = 0;
        PC_FREQ[i]       = 0;
    }
    G_TICKS = 0;
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

    // Composite priority selection
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
    const bool pref   = (type == ACCESS_PREFETCH);
    // Train TinyLFU + RunStop on demand references only
    uint8_t rs_conf = 0;
    if (demand) {
        // Update run detector and frequency structures
        rs_conf = update_runstop(PC, paddr);
        uint64_t adm_key = (static_cast<uint64_t>(PC) << 16) ^ region_id(paddr);
        cms_increment(adm_key);
        uint32_t pidx = pc_index(PC);
        if (PC_FREQ[pidx] < ((1u << PCF_BITS) - 1)) PC_FREQ[pidx]++;
        // Periodic global decay
        G_TICKS++;
        if ((G_TICKS & ((1u << DECAY_LOG2) - 1)) == 0) decay_all();
    }

    if (hit) {
        // On hits, enforce multi-hit gate and avoid promoting on first hit/prefetch
        age_all(set);
        if (demand) {
            if (HITCT[set][way] < 2) HITCT[set][way]++;
            if (HITCT[set][way] >= HITS_TO_PROMOTE) {
                STAT[set][way] = ST_NORMAL; // shed any sentinel on promotion
                AGE[set][way]  = 0;         // MRU on confirmed reuse
            }
            // else: no promotion on first demand hit
        } else {
            // Prefetch hit: never promote
            // Keep sentinel/quarantine if present
        }
        return;
    }

    // Miss: decide insertion
    // Record PC signature
    uint8_t pcs = static_cast<uint8_t>(pc_index(PC));
    PCSIG[set][way] = pcs;

    // Choose insertion depth and sentinel
    age_all(set);

    // Writebacks: never bypass; insert old, neutral
    if (type == ACCESS_WRITEBACK) {
        AGE[set][way]   = AGE_MAX;
        HITCT[set][way] = 0;
        STAT[set][way]  = ST_NORMAL;
        return;
    }

    // Prefetch: quarantine at tail
    if (pref) {
        AGE[set][way]   = AGE_MAX;
        HITCT[set][way] = 0;
        STAT[set][way]  = ST_QUAR;
        return;
    }

    // Demand fill: combine RunStop stream detection and TinyLFU admission
    const bool streamish = (rs_conf >= STREAM_CONF_TH);
    uint64_t adm_key = (static_cast<uint64_t>(PC) << 16) ^ region_id(paddr);
    uint8_t cms_freq = cms_query(adm_key);
    uint8_t pcfreq   = PC_FREQ[pcs];

    if (streamish) {
        // Near-bypass for streams: hard tail, stream sentinel, never promote until 2nd+ demand hit
        AGE[set][way]   = AGE_MAX;
        HITCT[set][way] = 0;
        STAT[set][way]  = ST_STREAM;
        return;
    }

    // TinyLFU admission: cold => tail; warm => slightly young
    if ((cms_freq < CMS_FREQ_TH) && (pcfreq < PCF_WARM_TH)) {
        AGE[set][way]   = AGE_MAX;    // cold admission at hard tail
        HITCT[set][way] = 0;
        STAT[set][way]  = ST_NORMAL;  // no sentinel; first hit still won't promote
    } else {
        AGE[set][way]   = AGE_YOUNG;  // slightly young for warm keys
        HITCT[set][way] = 0;
        STAT[set][way]  = ST_NORMAL;
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}