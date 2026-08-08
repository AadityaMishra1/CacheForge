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

static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}

// ---------------- Tunables ----------------
// RRIP parameters
static constexpr uint8_t  RRPV_DISTANT = 3;

// StreamSentinel thresholds
static constexpr uint32_t STREAM_PC_BITS = 10;                // 1024-entry stream table
static constexpr uint32_t STREAM_SIZE    = (1u << STREAM_PC_BITS);
static constexpr uint8_t  STREAM_NEAR_TAIL_TH = 2;            // >=2 => insert tail with stream sentinel
static constexpr uint8_t  STREAM_BYPASS_TH    = 3;            // >=3 => also tail sentinel (no real bypass due to API)

// Track low line bits for stride (10 bits -> 1K wrap)
static constexpr uint8_t  STREAM_LINE_BITS = 10;

// CMS dead-block predictor (PC-keyed via 8-bit signature)
static constexpr uint8_t  PC_SIG_BITS = 8;                    // per-line stored signature
static constexpr uint32_t CMS_ROWS    = 3;
static constexpr uint32_t CMS_SIZE    = (1u << PC_SIG_BITS);  // 256 counters per row
static constexpr uint8_t  CMS_INC     = 1;                    // increment on 2nd+ demand hit
static constexpr uint8_t  CMS_DEC     = 1;                    // decrement on dead-on-evict
static constexpr uint8_t  CMS_MAX     = 255;

static constexpr uint8_t  CMS_WARM_TH = 3;                    // >=3 => warm
static constexpr uint8_t  CMS_COLD_TH = 1;                    // <=1 => cold

// Insertion depths
static constexpr uint8_t  INSERT_WARM_RRPV = 1;               // warm inserts younger
static constexpr uint8_t  INSERT_COLD_RRPV = RRPV_DISTANT;    // cold/stream/prefetch/WB insert tail

// ---------------- Per-line state ----------------
// RRPV: 2 bits (0..3)
static uint8_t RRPV[LLC_SETS][LLC_WAYS];
// REGION: 0=probation, 1=protected
static uint8_t REGION[LLC_SETS][LLC_WAYS];
// STATUS: 0=normal, 1=stream-sentinel (no promote/train), 2=pf/wb-sentinel (no promote/train)
static uint8_t STATUS[LLC_SETS][LLC_WAYS];
// HITCNT: 0=none, 1=one demand hit, 2=two-or-more (saturates)
static uint8_t HITCNT[LLC_SETS][LLC_WAYS];
// PC signature (8-bit) captured at fill for training
static uint8_t PCSIG[LLC_SETS][LLC_WAYS];

// ---------------- Global predictors ----------------
// StreamSentinel: per-PC last line + confidence + last_abs1
static uint16_t ST_LAST_LINE[STREAM_SIZE]; // 10-bit value stored in 16-bit slot
static uint8_t  ST_CONF[STREAM_SIZE];      // 0..3
static uint8_t  ST_LAST_ABS1[STREAM_SIZE]; // 0/1

// Count-Min Sketch: 3 rows x 256 8-bit counters
static uint8_t CMS[CMS_ROWS][CMS_SIZE];

// ---------------- Helpers ----------------
static inline uint32_t stream_index(uint64_t pc) {
    // simple mix to 10 bits
    uint64_t x = pc ^ (pc >> 9) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (STREAM_SIZE - 1);
}
static inline uint16_t line_lowN(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << STREAM_LINE_BITS) - 1));
}
static inline uint8_t pc_sig8(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 8) ^ (pc >> 16) ^ (pc >> 24);
    return static_cast<uint8_t>(x & 0xFFu);
}
static inline uint32_t cms_index_from_pc(uint64_t pc, uint32_t row) {
    uint8_t sig = pc_sig8(pc);
    // row-dependent salt
    static const uint8_t salt[CMS_ROWS] = {0x5B, 0xA7, 0xD3};
    return static_cast<uint32_t>((sig ^ salt[row]) & (CMS_SIZE - 1));
}
static inline uint32_t cms_index_from_sig(uint8_t sig, uint32_t row) {
    static const uint8_t salt[CMS_ROWS] = {0x5B, 0xA7, 0xD3};
    return static_cast<uint32_t>((sig ^ salt[row]) & (CMS_SIZE - 1));
}
static inline void cms_inc_by_sig(uint8_t sig) {
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index_from_sig(sig, r);
        uint8_t &c = CMS[r][idx];
        uint16_t tmp = static_cast<uint16_t>(c) + CMS_INC;
        c = static_cast<uint8_t>(tmp > CMS_MAX ? CMS_MAX : tmp);
    }
}
static inline void cms_dec_by_sig(uint8_t sig) {
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index_from_sig(sig, r);
        uint8_t &c = CMS[r][idx];
        c = (c > CMS_DEC) ? static_cast<uint8_t>(c - CMS_DEC) : 0;
    }
}
static inline uint8_t cms_query_pc(uint64_t pc) {
    uint8_t m = 255;
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index_from_pc(pc, r);
        uint8_t c = CMS[r][idx];
        if (c < m) m = c;
    }
    return m;
}
static inline uint8_t cms_query_sig(uint8_t sig) {
    uint8_t m = 255;
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        uint32_t idx = cms_index_from_sig(sig, r);
        uint8_t c = CMS[r][idx];
        if (c < m) m = c;
    }
    return m;
}

static inline uint8_t update_stream_conf(uint64_t pc, uint64_t paddr) {
    uint32_t idx = stream_index(pc);
    uint16_t curr = line_lowN(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(ST_LAST_LINE[idx]));
    bool abs1 = (delta == 1) || (delta == -1);

    if (abs1) {
        if (ST_LAST_ABS1[idx]) {
            if (ST_CONF[idx] < 3) ST_CONF[idx]++;
        } else {
            if (ST_CONF[idx] == 0) ST_CONF[idx] = 1;
            ST_LAST_ABS1[idx] = 1;
        }
    } else {
        if (ST_CONF[idx] > 0) ST_CONF[idx]--;
        ST_LAST_ABS1[idx] = 0;
    }
    ST_LAST_LINE[idx] = curr;
    return ST_CONF[idx];
}

static inline bool more_evictable(uint32_t set, uint32_t a, uint32_t b) {
    // 1) Stream/prefetch sentinels first
    bool sa = (STATUS[set][a] != 0);
    bool sb = (STATUS[set][b] != 0);
    if (sa != sb) return sa; // true means 'a' more evictable

    // 2) CMS-cold lines next (based on stored fill PC signature)
    bool cold_a = (cms_query_sig(PCSIG[set][a]) <= CMS_COLD_TH);
    bool cold_b = (cms_query_sig(PCSIG[set][b]) <= CMS_COLD_TH);
    if (cold_a != cold_b) return cold_a;

    // 3) Prefer probation region
    bool prob_a = (REGION[set][a] == 0);
    bool prob_b = (REGION[set][b] == 0);
    if (prob_a != prob_b) return prob_a;

    // 4) Larger RRPV
    if (RRPV[set][a] != RRPV[set][b]) return RRPV[set][a] > RRPV[set][b];

    // 5) Fewer hits
    if (HITCNT[set][a] != HITCNT[set][b]) return HITCNT[set][a] < HITCNT[set][b];

    return a < b; // stable tie-breaker
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            RRPV[s][w]   = RRPV_DISTANT;
            REGION[s][w] = 0;   // probation
            STATUS[s][w] = 0;   // normal
            HITCNT[s][w] = 0;
            PCSIG[s][w]  = 0;
        }
    }
    for (uint32_t i = 0; i < STREAM_SIZE; i++) {
        ST_LAST_LINE[i] = 0;
        ST_CONF[i]      = 0;
        ST_LAST_ABS1[i] = 0;
    }
    for (uint32_t r = 0; r < CMS_ROWS; r++) {
        for (uint32_t i = 0; i < CMS_SIZE; i++) {
            CMS[r][i] = 0; // cold start
        }
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
    // Prefer invalid ways
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Composite selection per 'more_evictable'
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
    if (demand) {
        stream_conf = update_stream_conf(PC, paddr);
    }

    if (hit) {
        // Hits on sentinels: never promote, never train
        if (STATUS[set][way] != 0) {
            return;
        }

        // Demand hit handling
        if (demand) {
            uint8_t prev = HITCNT[set][way];
            if (prev < 3) HITCNT[set][way] = static_cast<uint8_t>(prev + 1);

            // Multi-hit promotion: only on 2nd+ demand hit
            if (prev >= 1) {
                // Promote to protected and MRU
                REGION[set][way] = 1;    // protected
                RRPV[set][way] = 0;

                // Train CMS toward warm using fill PC signature
                cms_inc_by_sig(PCSIG[set][way]);
            }
            // First hit: no promotion
        } else {
            // Non-demand hit (shouldn't occur, but guard): do nothing
        }
        return;
    }

    // Miss path: train on victim before overwrite
    // Dead-on-evict: 0 demand hits and normal status
    if (STATUS[set][way] == 0 && HITCNT[set][way] == 0) {
        cms_dec_by_sig(PCSIG[set][way]);
    }

    // New line insertion policy
    uint8_t new_status = 0;
    uint8_t new_rrpv   = INSERT_COLD_RRPV; // default tail
    uint8_t new_region = 0;                // probation
    uint8_t new_hitcnt = 0;
    uint8_t new_pcsig  = pc_sig8(PC);

    if (type == ACCESS_PREFETCH) {
        // Prefetch: always tail, sentinel, no-train/no-promote
        new_status = 2;
        new_rrpv   = INSERT_COLD_RRPV;
    } else if (type == ACCESS_WRITEBACK) {
        // Writeback: never bypass; treat like sentinel at tail
        new_status = 2;
        new_rrpv   = INSERT_COLD_RRPV;
    } else {
        // Demand fill
        if (stream_conf >= STREAM_NEAR_TAIL_TH) {
            // StreamSentinel: aggressive tail, no-train/no-promote
            new_status = 1;
            new_rrpv   = INSERT_COLD_RRPV;
        } else {
            // Use CMS prediction for non-stream demand
            uint8_t score = cms_query_pc(PC);
            if (score <= CMS_COLD_TH) {
                // near-bypass tail
                new_status = 0;
                new_rrpv   = INSERT_COLD_RRPV;
            } else if (score >= CMS_WARM_TH) {
                // warm insert
                new_status = 0;
                new_rrpv   = INSERT_WARM_RRPV;
            } else {
                // uncertain: mid-depth
                new_status = 0;
                new_rrpv   = 2;
            }
        }
    }

    // Overwrite line metadata for the new block
    STATUS[set][way] = new_status;
    RRPV[set][way]   = new_rrpv;
    REGION[set][way] = new_region;
    HITCNT[set][way] = new_hitcnt;
    PCSIG[set][way]  = new_pcsig;
}

// Print end-of-simulation statistics
void PrintStats() {
    // (intentionally blank)
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // (intentionally blank)
}