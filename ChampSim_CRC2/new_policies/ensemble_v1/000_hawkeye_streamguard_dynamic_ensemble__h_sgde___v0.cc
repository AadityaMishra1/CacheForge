#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ============================================================================
// Leader-Follower Sampling (64 sets)
// ============================================================================
#define LLC_SET_BITS 11
static inline bool SAMPLED_SET(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SAMPLED_SET(set) && ((set & 1u) == 1u); }

// ============================================================================
// Tunables (can be tweaked per-surrogate guidance)
// ============================================================================
static const int RRPV_BITS         = 3;
static const int RRPV_MAX          = (1 << RRPV_BITS) - 1; // 7
static const int SHCT_SIZE         = 2048;  // Hawkeye predictor size
static const int SHCT_CTR_BITS     = 5;     // 5-bit counters
static const int SHCT_MAX          = (1 << SHCT_CTR_BITS) - 1;
static const int SHCT_INIT         = 1;     // initial counter value
static const int SHCT_FRIENDLY_TH  = 2;     // predict friendly if >= 2

static const int SAMPLER_SET_COUNT = 64;    // fixed by SAMPLED_SET mapping
static const int SAMPLER_WAYS      = 8;     // sampler associativity
static const int SAMPLER_TAG_BITS  = 16;    // partial tag to train reuse
static const int SIG_BITS          = 11;    // PC signature bits -> 2048 entries

// StreamGuard tunables
static const int PC_TABLE_SIZE     = 512;   // stride+freq tables
static const int STREAM_CONF_MAX   = 3;
static const int STREAM_CONF_TH    = 2;     // >=2 means stream detected
static const int PROMOTE_HITS      = 2;     // multi-hit gate to promote
static const int PC_FREQ_BITS      = 4;     // 0..15
static const int PC_FREQ_TH        = 4;     // cold PCs below 4 are demoted
static const int PREFETCH_TYPE     = 2;     // Champsim: LOAD=0, RFO=1, PREFETCH=2, WRITEBACK=3
static const int WRITEBACK_TYPE    = 3;
static const int PREFETCH_QUAR_RRPV= RRPV_MAX; // quarantine depth for prefetches

// Selector tunables
static const int32_t BIAS          = 4;
static const int CONF_MAX          = 127;
static const int CONF_TH           = 64;     // per-set threshold to use Mode B
static const uint64_t SELECTOR_PERIOD = 4096;
static const uint64_t DECAY_PERIOD    = 16384;

// ============================================================================
// Global State
// ============================================================================

// Per-line metadata
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];        // 2-bit (store in 0..3)
static uint8_t modeB_stream_tag[LLC_SETS][LLC_WAYS]; // 1-bit flag

// Mode selector
static int32_t leaderA_score = 0;
static int32_t leaderB_score = 0;
static bool prefer_mode_B = false;
static int8_t mode_confidence[LLC_SETS];
static uint64_t access_count = 0;

// ===================== Hawkeye-Like (Mode A) ================================
// SHCT: PC-signature history counter
static uint8_t SHCT[SHCT_SIZE];

// Sampler (64 sets x 8 ways)
struct SamplerEntry {
    uint16_t tag16;
    uint16_t sig : SIG_BITS;
    uint16_t used: 1;
    uint16_t valid: 1;
};
static SamplerEntry sampler[SAMPLER_SET_COUNT][SAMPLER_WAYS];
static uint8_t sampler_rrptr[SAMPLER_SET_COUNT]; // round-robin insert

static inline uint32_t get_sig(uint64_t PC) {
    // simple hash: CRC32 of PC then fold to SIG_BITS or just low bits
    return (champsim_crc32(0, &PC, sizeof(PC)) ^ (PC >> 2)) & (SHCT_SIZE - 1);
}
static inline uint16_t line_tag16(uint64_t paddr) {
    // 64B lines -> line address, take low 16 bits as partial tag
    return (uint16_t)((paddr >> 6) & 0xFFFFu);
}
static inline uint32_t sampler_set_index(uint32_t set) {
    // Map actual LLC set to [0..63] sampler index deterministically
    // Use the low 6 bits of set (same that SAMPLED_SET uses)
    return (set & 63u);
}

static void hawkeye_sampler_train(uint32_t set, uint64_t PC, uint64_t paddr) {
    if (!SAMPLED_SET(set)) return;

    uint32_t ss = sampler_set_index(set);
    uint16_t tg = line_tag16(paddr);
    uint32_t sig = get_sig(PC);

    // Lookup hit in sampler
    int hit_way = -1;
    for (int w = 0; w < SAMPLER_WAYS; w++) {
        if (sampler[ss][w].valid && sampler[ss][w].tag16 == tg) {
            hit_way = w;
            break;
        }
    }

    if (hit_way >= 0) {
        // Reuse observed -> increment SHCT for old signature
        if (SHCT[sampler[ss][hit_way].sig] < SHCT_MAX) SHCT[sampler[ss][hit_way].sig]++;
        sampler[ss][hit_way].used = 1;
        // Refresh tag/signature (track latest PC)
        sampler[ss][hit_way].sig = sig;
        sampler[ss][hit_way].tag16 = tg;
        sampler[ss][hit_way].valid = 1;
        return;
    }

    // Miss in sampler: insert/replace via round-robin
    int w = sampler_rrptr[ss] % SAMPLER_WAYS;
    sampler_rrptr[ss] = (sampler_rrptr[ss] + 1) & 0x7;

    // If evicting an entry that has not been reused, decrement its SHCT
    if (sampler[ss][w].valid && sampler[ss][w].used == 0) {
        if (SHCT[sampler[ss][w].sig] > 0) SHCT[sampler[ss][w].sig]--;
    }

    sampler[ss][w].tag16 = tg;
    sampler[ss][w].sig   = sig;
    sampler[ss][w].used  = 0;
    sampler[ss][w].valid = 1;
}

static inline bool hawkeye_predict_friendly(uint64_t PC) {
    return (SHCT[get_sig(PC)] >= SHCT_FRIENDLY_TH);
}

// ===================== StreamGuard (Mode B) =================================
// PC-stride stream detector (+1/+2 forward runs) and frequency gate
static uint16_t pc_last_line[PC_TABLE_SIZE];
static uint8_t  pc_stream_conf[PC_TABLE_SIZE]; // 0..3
static uint8_t  pc_freq[PC_TABLE_SIZE];        // 4-bit frequency

static inline uint32_t pc_index(uint64_t PC) { return (PC ^ (PC >> 4)) & (PC_TABLE_SIZE - 1); }
static inline uint32_t line_addr(uint64_t paddr) { return (uint32_t)(paddr >> 6); }

static bool detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint32_t line = line_addr(paddr) & 0xFFFFu;

    bool is_stream = false;
    if (pc_last_line[idx] != 0) {
        int32_t delta = (int32_t)line - (int32_t)pc_last_line[idx];
        if (delta == 1 || delta == 2) {
            if (pc_stream_conf[idx] < STREAM_CONF_MAX) pc_stream_conf[idx]++;
        } else {
            // reset on non-forward small strides
            pc_stream_conf[idx] = 0;
        }
    }
    pc_last_line[idx] = (uint16_t)line;
    is_stream = (pc_stream_conf[idx] >= STREAM_CONF_TH);
    return is_stream;
}

static inline bool pc_is_cold(uint64_t PC) {
    uint32_t idx = pc_index(PC);
    // increment freq (saturating)
    if (pc_freq[idx] < ((1u << PC_FREQ_BITS) - 1)) pc_freq[idx]++;
    return (pc_freq[idx] < PC_FREQ_TH);
}

// ===================== Selector ============================================
static void update_selector(uint32_t set, uint8_t hit) {
    access_count++;

    if (SAMPLED_SET(set)) {
        int delta = hit ? 1 : -1;
        if (LEADER_A(set)) leaderA_score += delta;
        if (LEADER_B(set)) leaderB_score += delta;
    }

    if ((access_count % SELECTOR_PERIOD) == 0) {
        prefer_mode_B = (leaderB_score > (leaderA_score + BIAS));

        // Drift per-set confidence
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (prefer_mode_B) {
                if (mode_confidence[s] < CONF_MAX) mode_confidence[s]++;
            } else {
                if (mode_confidence[s] > 0) mode_confidence[s]--;
            }
        }
        // Gentle decay of PC frequency to adapt to phases
        if ((access_count % DECAY_PERIOD) == 0) {
            for (int i = 0; i < PC_TABLE_SIZE; i++) {
                pc_freq[i] = (pc_freq[i] >> 1);
            }
        }
    }
}

static inline bool should_use_mode_B(uint32_t set) {
    if (LEADER_A(set)) return false; // Force Mode A
    if (LEADER_B(set)) return true;  // Force Mode B
    return prefer_mode_B && (mode_confidence[set] >= CONF_TH);
}

// ===================== Common RRIP Victim ===================================
static uint32_t find_victim_rrip(uint32_t set, const BLOCK *current_set) {
    // 1) Invalid way first
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (current_set[way].valid == 0) return way;
    }
    // 2) RRIP selection
    while (true) {
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] >= RRPV_MAX) return way;
        }
        for (uint32_t way = 0; way < LLC_WAYS; way++) {
            if (rrpv[set][way] < RRPV_MAX) rrpv[set][way]++;
        }
    }
}

// ===================== Mode A (Hawkeye-like) behaviors ======================
static uint32_t hawkeye_victim(uint32_t set, const BLOCK *current_set) {
    return find_victim_rrip(set, current_set);
}
static void hawkeye_insertion(uint32_t set, uint32_t way, uint64_t PC, uint32_t type) {
    // Never bypass writebacks; treat writebacks as friendly (keep data for potential RFO)
    if (type == (uint32_t)WRITEBACK_TYPE) {
        rrpv[set][way] = 0;
        return;
    }
    bool friendly = hawkeye_predict_friendly(PC);
    rrpv[set][way] = friendly ? 0 : RRPV_MAX; // friendly->MRU, unfriendly->tail
}
static void hawkeye_on_hit(uint32_t set, uint32_t way) {
    rrpv[set][way] = 0; // promote to MRU on hits
}

// ===================== Mode B (StreamGuard) behaviors =======================
static uint32_t mode_B_victim(uint32_t set, const BLOCK *current_set) {
    return find_victim_rrip(set, current_set);
}
static void mode_B_insertion(uint32_t set, uint32_t way, uint64_t PC, uint64_t paddr, uint32_t type) {
    // Prefetch quarantine (always tail)
    if (type == (uint32_t)PREFETCH_TYPE) {
        rrpv[set][way] = PREFETCH_QUAR_RRPV;
        modeB_stream_tag[set][way] = 1;
        hitcnt[set][way] = 0;
        return;
    }
    // Never bypass on writeback
    if (type == (uint32_t)WRITEBACK_TYPE) {
        rrpv[set][way] = 0;
        modeB_stream_tag[set][way] = 0;
        hitcnt[set][way] = 0;
        return;
    }

    bool is_stream = detect_stream(PC, paddr);
    bool is_cold_pc = pc_is_cold(PC);

    if (is_stream) {
        // Stream/scans -> hard-tail insert and gate promotions
        rrpv[set][way] = RRPV_MAX;
        modeB_stream_tag[set][way] = 1;
        hitcnt[set][way] = 0;
    } else if (is_cold_pc) {
        // Cold/noisy PCs -> conservative admission (tail), rescued by multi-hit gate
        rrpv[set][way] = RRPV_MAX;
        modeB_stream_tag[set][way] = 0;
        hitcnt[set][way] = 0;
    } else {
        // Warm/hot non-stream demand -> mild protection
        rrpv[set][way] = (RRPV_MAX > 1) ? (RRPV_MAX - 1) : 0; // near-MRU
        modeB_stream_tag[set][way] = 0;
        hitcnt[set][way] = 0;
    }
}
static void mode_B_on_hit(uint32_t set, uint32_t way) {
    // Multi-hit promotion gate for stream/quarantined/cold lines
    if (modeB_stream_tag[set][way]) {
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        if (hitcnt[set][way] >= PROMOTE_HITS) {
            rrpv[set][way] = 0;            // promote to MRU and clear stream tag
            modeB_stream_tag[set][way] = 0;
            hitcnt[set][way] = 0;
        } else {
            // Gentle promotion but keep gate
            if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
    } else {
        // Non-stream: normal promotion
        rrpv[set][way] = 0;
    }
}

// ===================== APIs required by template ============================

void InitReplacementState() {
    // Per-line state
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = RRPV_MAX;
            hitcnt[s][w] = 0;
            modeB_stream_tag[s][w] = 0;
        }
        mode_confidence[s] = 0;
    }
    // Selector
    leaderA_score = 0;
    leaderB_score = 0;
    prefer_mode_B = false;
    access_count = 0;

    // Hawkeye-like predictor
    for (int i = 0; i < SHCT_SIZE; i++) SHCT[i] = SHCT_INIT;
    for (int ss = 0; ss < SAMPLER_SET_COUNT; ss++) {
        memset(sampler[ss], 0, sizeof(sampler[ss]));
        sampler_rrptr[ss] = 0;
    }

    // StreamGuard tables
    memset(pc_last_line, 0, sizeof(pc_last_line));
    memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    memset(pc_freq, 0, sizeof(pc_freq));
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    bool use_mode_B = should_use_mode_B(set);
    if (use_mode_B) {
        return mode_B_victim(set, current_set);
    } else {
        return hawkeye_victim(set, current_set);
    }
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
    (void)cpu; (void)victim_addr;

    // Update selector accounting first
    update_selector(set, hit);

    // Always train Mode A sampler/SHCT (safety net always learning)
    hawkeye_sampler_train(set, PC, paddr);

    bool use_mode_B = should_use_mode_B(set);

    if (hit) {
        // On hit, apply chosen mode's hit policy
        if (use_mode_B) {
            mode_B_on_hit(set, way);
        } else {
            hawkeye_on_hit(set, way);
        }
    } else {
        // On miss: we are installing a new block at (set, way)
        // Reset per-line aux
        hitcnt[set][way] = 0;
        modeB_stream_tag[set][way] = 0;

        if (use_mode_B) {
            mode_B_insertion(set, way, PC, paddr, type);
        } else {
            hawkeye_insertion(set, way, PC, type);
        }
    }
}

void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}