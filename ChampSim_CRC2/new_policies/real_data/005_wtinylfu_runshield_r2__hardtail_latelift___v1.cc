#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access type tags (CRC2 convention)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Tunables (R2 retune) ----------------
static constexpr uint8_t  PC_BITS              = 8;       // 256-entry PC-indexed structures
static constexpr uint32_t PC_SIZE              = (1u << PC_BITS);

static constexpr uint8_t  AGE_MAX              = 3;       // 0 (MRU/young) .. 3 (LRU/tail)
static constexpr uint8_t  AGE_YOUNG            = 1;

static constexpr uint8_t  HITS_TO_PROMOTE_BASE = 2;       // non-stream: lift on 2nd demand hit
static constexpr uint8_t  HITS_TO_PROMOTE_STRM = 3;       // stream-tagged: lift on 3rd demand hit

static constexpr uint8_t  EPOCH_INIT           = 1;       // 2-bit epoch start
static constexpr uint8_t  EPOCH_WARM_TH        = 2;       // >=2 = warm

static constexpr uint8_t  STREAM_CONF_TH       = 2;       // RunShield confidence to quarantine
static constexpr uint8_t  LINE_LOW_BITS        = 12;      // line# low bits for run detection

// TinyLFU Count-Min Sketch parameters
static constexpr uint8_t  CMS_ROWS             = 3;
static constexpr uint16_t CMS_WIDTH            = 256;     // match PC index width
static constexpr uint8_t  CMS_BITS             = 4;       // 0..15 saturating
static constexpr uint8_t  CMS_MAX              = (1u << CMS_BITS) - 1;
static constexpr uint32_t CMS_AGING_PERIOD     = 2048;    // faster aging (R2)
static constexpr uint8_t  ADMIT_BIAS           = 0;       // lower bias (R2)

// Quarantine demotion accelerator
static constexpr uint8_t  FAST_DEMOTE_ON_TOUCH = 1;       // extra aging step for quarantined lines

// ---------------- Per-line state ----------------
// Stored in bytes; report minimal information bits separately.
static uint8_t AGE    [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3
static uint8_t HITCNT [LLC_SETS][LLC_WAYS];   // 2 bits: 0..3 (we cap at 3)
static uint8_t SENT   [LLC_SETS][LLC_WAYS];   // 2 bits: 0=none, 1=stream, 2=prefetch, 3=cold-quarantine
static uint8_t PCSIG  [LLC_SETS][LLC_WAYS];   // 8-bit PC index of last toucher/filler
static uint8_t LVALID [LLC_SETS][LLC_WAYS];   // 1 bit: our mirror of "line had been valid before fill"

// ---------------- Global predictors ----------------
// PC epoch (deadness bias), 2-bit per PC
static uint8_t PC_EPOCH[PC_SIZE];             // 0..3

// RunShield per PC: ±1/±2 forward-run detector
static uint16_t RS_LAST_LINE[PC_SIZE];        // store low LINE_LOW_BITS of line#
static uint8_t  RS_CONF[PC_SIZE];             // 2-bit confidence (0..3)
static uint8_t  RS_LAST_ABS12[PC_SIZE];       // 1-bit
static uint8_t  RS_LAST_FWD[PC_SIZE];         // 1-bit

// TinyLFU Count-Min Sketch: CMS_ROWS x CMS_WIDTH, 4-bit counters
static uint8_t CMS[CMS_ROWS][CMS_WIDTH];      // use lower CMS_BITS bits
static uint32_t cms_tick = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline uint32_t pc_index(uint64_t pc) {
    // Lightweight mixer to decorrelate low bits
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 21);
    return static_cast<uint32_t>(x) & (PC_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS) - 1));
}
static inline void sat_inc_u2(uint8_t &v) { if (v < 3) v++; }
static inline void sat_dec_u2(uint8_t &v) { if (v > 0) v--; }
static inline void age_all(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE[set][w] < AGE_MAX) AGE[set][w]++;
    }
}
static inline uint32_t cms_hash(uint8_t row, uint32_t key) {
    static constexpr uint32_t SALT[CMS_ROWS] = {0x9e3779b9u, 0x7f4a7c15u, 0x94d049bbu};
    uint32_t x = key ^ (SALT[row] >> (row + 1));
    x ^= (x >> 7);
    x ^= (x >> 13);
    return x & (CMS_WIDTH - 1);
}
static inline uint8_t cms_query(uint32_t key) {
    uint8_t m = 0xFF;
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_hash(r, key);
        uint8_t c = CMS[r][idx] & CMS_MAX;
        if (c < m) m = c;
    }
    return (m == 0xFF) ? 0 : m;
}
static inline void cms_add(uint32_t key) {
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_hash(r, key);
        uint8_t c = CMS[r][idx] & CMS_MAX;
        if (c < CMS_MAX) CMS[r][idx] = static_cast<uint8_t>(c + 1);
    }
    cms_tick++;
    if (cms_tick % CMS_AGING_PERIOD == 0) {
        for (uint8_t r = 0; r < CMS_ROWS; r++) {
            for (uint32_t i = 0; i < CMS_WIDTH; i++) {
                CMS[r][i] = static_cast<uint8_t>((CMS[r][i] & CMS_MAX) >> 1);
            }
        }
    }
}
// Update RunShield on demand access; returns updated confidence
static inline uint8_t update_runshield(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RS_LAST_LINE[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RS_LAST_ABS12[idx] && RS_LAST_FWD[idx]) {
            sat_inc_u2(RS_CONF[idx]); // two consecutive forward small strides
        } else {
            if (RS_CONF[idx] == 0) RS_CONF[idx] = 1;
            RS_LAST_ABS12[idx] = 1;
            RS_LAST_FWD[idx]   = 1;
        }
    } else {
        sat_dec_u2(RS_CONF[idx]);     // decay on irregular/backwards/large stride
        RS_LAST_ABS12[idx] = 0;
        RS_LAST_FWD[idx]   = 0;
    }
    RS_LAST_LINE[idx] = curr;
    return RS_CONF[idx];
}

static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Quarantined (stream/prefetch/cold) first
    bool qa = (SENT[set][a] != 0);
    bool qb = (SENT[set][b] != 0);
    if (qa != qb) return qa;

    // 2) Colder PC epoch
    uint8_t ea = PC_EPOCH[ PCSIG[set][a] ];
    uint8_t eb = PC_EPOCH[ PCSIG[set][b] ];
    if (ea != eb) return (ea < eb); // lower epoch => colder => more evictable

    // 3) Fewer demand hits
    uint8_t ha = HITCNT[set][a];
    uint8_t hb = HITCNT[set][b];
    if (ha != hb) return (ha < hb);

    // 4) Older age
    if (AGE[set][a] != AGE[set][b]) return (AGE[set][a] > AGE[set][b]);

    // 5) Tie-breaker: lower way index loses (arbitrary but deterministic)
    return (a < b);
}

// ---------------- CRC2 hooks ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE[s][w]    = AGE_MAX;
            HITCNT[s][w] = 0;
            SENT[s][w]   = 0;
            PCSIG[s][w]  = 0;
            LVALID[s][w] = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE; i++) {
        PC_EPOCH[i]     = EPOCH_INIT;
        RS_LAST_LINE[i] = 0;
        RS_CONF[i]      = 0;
        RS_LAST_ABS12[i]= 0;
        RS_LAST_FWD[i]  = 0;
    }
    for (uint8_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_WIDTH; i++) CMS[r][i] = 0;
    }
    cms_tick = 0;
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // 1) Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // 2) Choose the most-evictable line according to quarantine/epoch/hits/age
    uint32_t victim = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable(set, w, victim)) victim = w;
    }
    return victim;
}

void UpdateReplacementState(
    uint32_t cpu,
    uint32_t set,
    uint32_t way,
    uint64_t paddr,
    uint64_t PC,
    uint64_t victim_addr,
    uint32_t type,
    uint8_t hit
) {
    uint32_t pcidx = pc_index(PC);

    // Update TinyLFU and RunShield on demand accesses
    if (is_demand(type)) {
        cms_add(pcidx);
        update_runshield(PC, paddr);
    }

    // Age everyone; quarantined get extra demotion to protect cache residency
    age_all(set);
    if (FAST_DEMOTE_ON_TOUCH) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (SENT[set][w] != 0 && AGE[set][w] < AGE_MAX) AGE[set][w]++;
        }
    }

    if (hit) {
        // On hit: multi-hit gated promotion; prefetch/writeback do not count as demand hits
        if (is_demand(type)) {
            if (HITCNT[set][way] < 3) HITCNT[set][way]++;

            // Determine promotion threshold
            uint8_t thr = (SENT[set][way] == 1) ? HITS_TO_PROMOTE_STRM : HITS_TO_PROMOTE_BASE;

            if (HITCNT[set][way] >= thr) {
                AGE[set][way]  = 0;       // LateLift to MRU
                SENT[set][way] = 0;       // shed quarantine
                // Warm the PC epoch on confirmed reuse
                if (PC_EPOCH[pcidx] < 3) PC_EPOCH[pcidx]++;
            } else {
                // Light youth on first demand hit without full promotion
                if (AGE[set][way] > AGE_YOUNG) AGE[set][way] = AGE_YOUNG;
            }
            // Keep last toucher PC for frequency/victim comparison
            PCSIG[set][way] = static_cast<uint8_t>(pcidx & 0xFF);
        } else {
            // Non-demand hit: minimal effect, keep it from falling off a cliff
            if (AGE[set][way] > AGE_YOUNG) AGE[set][way] = AGE_YOUNG;
        }
        return;
    }

    // Miss path (fill): train victim's PC epoch on eviction using our mirror
    if (LVALID[set][way]) {
        uint8_t v_psig  = PCSIG[set][way];
        uint8_t v_hits  = HITCNT[set][way];
        uint8_t v_sent  = SENT[set][way];
        // Dead vs live feedback
        if (v_hits == 0) {
            if (PC_EPOCH[v_psig] > 0) PC_EPOCH[v_psig]--;
        } else if (v_hits >= 2) {
            if (PC_EPOCH[v_psig] < 3) PC_EPOCH[v_psig]++;
        } else {
            // Single-hit: if stream-quarantined, treat as deadish; else neutral
            if (v_sent == 1 && PC_EPOCH[v_psig] > 0) PC_EPOCH[v_psig]--;
        }
    }

    // Decide insertion policy for the incoming line
    uint8_t epoch   = PC_EPOCH[pcidx];
    uint8_t rs_conf = RS_CONF[pcidx]; // already updated on demand above
    bool    stream  = (rs_conf >= STREAM_CONF_TH);
    bool    demand  = is_demand(type);

    // WTinyLFU admission: incoming vs victim PC frequency
    uint8_t freq_new = cms_query(pcidx);
    uint8_t freq_old = cms_query(PCSIG[set][way]);
    bool admit = (static_cast<int>(freq_new) + ADMIT_BIAS >= static_cast<int>(freq_old));

    // Initialize state
    HITCNT[set][way] = 0;
    PCSIG[set][way]  = static_cast<uint8_t>(pcidx & 0xFF);
    LVALID[set][way] = 1;

    if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine: always hard-tail
        AGE[set][way]  = AGE_MAX;
        SENT[set][way] = 2; // prefetch
        return;
    }

    // Demand (LOAD/RFO) or WRITEBACK
    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; keep low priority and no quarantine tag
        AGE[set][way]  = AGE_MAX;
        SENT[set][way] = 0;
        return;
    }

    // Demand: combine RunShield, PC epoch, and TinyLFU admission
    if (stream) {
        // Stream quarantine: aggressive hard-tail; LateLift if reuse shows up later
        AGE[set][way]  = AGE_MAX;
        SENT[set][way] = 1; // stream
    } else if (!admit || (epoch < EPOCH_WARM_TH)) {
        // Cold/noisy PCs or not admitted: cold quarantine
        AGE[set][way]  = AGE_MAX;
        SENT[set][way] = 3; // cold
    } else {
        // Warm and admitted: mild youth insert
        AGE[set][way]  = AGE_YOUNG;
        SENT[set][way] = 0;
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}