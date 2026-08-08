#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"
#include "optgen.h"

// HawkFusion-StreamShield++
// Mode A: Hawkeye-like (OPTgen-trained small PC predictor) driving RRIP insertion
// Mode B: StreamShield (PC forward-run stream quarantine + multi-hit gating)
// Selector: per-set 4-bit bandit with Hawkeye-safe fallback

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16
static constexpr uint8_t MAX_RRPV = 7;

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK);
}

// ---------------- Per-line metadata (packed conceptually) ----------------
// rrpv: 3b, hitcnt: 2b (0,1,2; 3 used as "prefetch sentinel"), b_tag: 1b (inserted by Mode B stream quarantine)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t b_tag[LLC_SETS][LLC_WAYS]; // 1 if inserted by Mode B stream quarantine (for bandit attribution)

// ---------------- Per-set selector (4-bit bandit; enable Mode B if score >= 2) ----------------
static int8_t bandit_score[LLC_SETS];
static constexpr int8_t BANDIT_MIN   = -7;
static constexpr int8_t BANDIT_MAX   = 7;
static constexpr int8_t BANDIT_THRES = 2;

// ---------------- Hawkeye-like: small OPTgen + small PC predictor ----------------
// Sampled sets: 16 (very light)
// OPTgen occupancy vectors per set (used only on sampled sets)
static OPTgen optvec[LLC_SETS];
static uint64_t set_timer[LLC_SETS];

static constexpr uint32_t LLC_SET_BITS = 11;
static inline uint64_t bitmask_u64(uint32_t l) { return (l == 64) ? ~0ULL : ((1ULL << l) - 1ULL); }
static inline uint64_t bits_u64(uint64_t x, uint32_t i, uint32_t l) { return (x >> i) & bitmask_u64(l); }
// 16 sampled sets: lower 4 bits match upper 4 bits of set index
static inline bool SAMPLED_SET(uint32_t set) {
    return bits_u64(set, 0, 4) == bits_u64(set, LLC_SET_BITS - 4, 4);
}
static constexpr uint32_t OPTGEN_VEC_SIZE = 128;

// Small PC predictor (1K entries, 5-bit counter, friendly if ctr >= 16)
static constexpr uint32_t PRED_SIZE = 1024;
static uint8_t pc_pred[PRED_SIZE];
static inline uint32_t pred_idx(uint64_t pc) { return (uint32_t)pc & (PRED_SIZE - 1); }
static inline void pred_inc(uint64_t pc) { uint8_t &c = pc_pred[pred_idx(pc)]; if (c < 31) c++; }
static inline void pred_dec(uint64_t pc) { uint8_t &c = pc_pred[pred_idx(pc)]; if (c > 0) c--; }
static inline bool pred_friend(uint64_t pc) { return pc_pred[pred_idx(pc)] >= 16; }

// History sampler: 1000 entries, 8-way, 125 sets
#define SAMPLER_WAYS 8
#define SAMPLER_ENTRIES 1000
#define SAMPLER_SETS (SAMPLER_ENTRIES / SAMPLER_WAYS)
struct SampEntry {
    uint8_t  valid;
    uint8_t  tag;          // CRC(paddr>>12) % 256
    uint8_t  lru;          // 0..7
    uint8_t  pref;         // 1 if prefetched
    uint8_t  pc_sig;       // 8-bit PC signature (index into pc_pred)
    uint8_t  last_q;       // last quanta (mod 128)
};
static SampEntry sampler[SAMPLER_SETS][SAMPLER_WAYS];

static inline void sampler_touch_lru(uint32_t ss, uint32_t way, uint8_t new_lru = 0) {
    uint8_t old = sampler[ss][way].lru;
    for (uint32_t w = 0; w < SAMPLER_WAYS; w++) {
        if (!sampler[ss][w].valid) continue;
        if (sampler[ss][w].lru < old) sampler[ss][w].lru++;
    }
    sampler[ss][way].lru = new_lru;
}
static inline int32_t sampler_find(uint32_t ss, uint8_t tag) {
    for (uint32_t w = 0; w < SAMPLER_WAYS; w++)
        if (sampler[ss][w].valid && sampler[ss][w].tag == tag)
            return (int32_t)w;
    return -1;
}
static inline uint32_t sampler_victim(uint32_t ss) {
    // return invalid or LRU
    for (uint32_t w = 0; w < SAMPLER_WAYS; w++)
        if (!sampler[ss][w].valid)
            return w;
    for (uint32_t w = 0; w < SAMPLER_WAYS; w++)
        if (sampler[ss][w].lru == (SAMPLER_WAYS - 1))
            return w;
    return 0;
}

static inline void hawk_sample(uint32_t set, uint64_t paddr, uint64_t PC, uint32_t type) {
    if (!SAMPLED_SET(set)) return;
    uint64_t line = paddr >> 6;
    uint32_t ss = (uint32_t)(line % SAMPLER_SETS);
    uint8_t stag = (uint8_t)(CRC(paddr >> 12) & 0xFF);

    uint8_t curr_q = (uint8_t)(set_timer[set] % OPTGEN_VEC_SIZE);
    set_timer[set]++;

    int32_t hit_way = sampler_find(ss, stag);
    if (hit_way >= 0 && type != ACCESS_PREFETCH) {
        // Determine reuse vs dead using OPTgen
        uint8_t last_q = sampler[ss][hit_way].last_q;
        uint8_t dq = (curr_q >= last_q) ? (curr_q - last_q) : (uint8_t)(curr_q + OPTGEN_VEC_SIZE - last_q);
        bool wrap = (dq > OPTGEN_VEC_SIZE);
        if (!wrap && optvec[set].should_cache(curr_q, last_q))
            pred_inc((uint64_t)sampler[ss][hit_way].pc_sig);
        else
            pred_dec((uint64_t)sampler[ss][hit_way].pc_sig);
        sampler_touch_lru(ss, (uint32_t)hit_way, 0);
        sampler[ss][hit_way].pref = 0;
        sampler[ss][hit_way].last_q = curr_q;
        sampler[ss][hit_way].pc_sig = (uint8_t)pred_idx(PC);
    } else {
        // Allocate/replace
        uint32_t v = sampler_victim(ss);
        sampler[ss][v].valid  = 1;
        sampler[ss][v].tag    = stag;
        sampler[ss][v].lru    = 0;
        sampler[ss][v].pref   = (type == ACCESS_PREFETCH) ? 1 : 0;
        sampler[ss][v].last_q = curr_q;
        sampler[ss][v].pc_sig = (uint8_t)pred_idx(PC);
        // Age others
        for (uint32_t w = 0; w < SAMPLER_WAYS; w++) {
            if (w == v || !sampler[ss][w].valid) continue;
            sampler[ss][w].lru = (sampler[ss][w].lru < (SAMPLER_WAYS - 1)) ? (sampler[ss][w].lru + 1) : sampler[ss][w].lru;
        }
    }

    optvec[set].add_access(curr_q);
}

// ---------------- PC forward-run stream detector ----------------
// 512-entry PC table: last line (12b) + 2-bit confidence
static constexpr uint32_t PC_TBL_SZ = 512;
static uint16_t pc_last_line[PC_TBL_SZ];
static uint8_t  pc_stream_conf[PC_TBL_SZ]; // 0..3
static inline uint32_t pc_idx(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SZ - 1); }
static inline uint16_t line_low12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }

static inline void update_stream_detector(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_idx(PC);
    uint16_t cur = line_low12(paddr);
    uint16_t prev = pc_last_line[idx];
    int16_t delta = (int16_t)(cur - prev);
    if (prev != 0 && (delta == 1 || delta == 2)) {
        if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
    } else {
        if (pc_stream_conf[idx] > 0) pc_stream_conf[idx]--;
    }
    pc_last_line[idx] = cur;
}

// ---------------- Utilities ----------------
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // If invalid exists, it should be returned by GetVictimInSet
    for (int iter = 0; iter < 8; iter++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (rrpv[set][w] == MAX_RRPV) return w;
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (rrpv[set][w] < MAX_RRPV) rrpv[set][w]++;
    }
    // Fallback: pick the most aged
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// ---------------- ChampSim hooks ----------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = MAX_RRPV;
            hitcnt[s][w] = 0;
            b_tag[s][w] = 0;
        }
        bandit_score[s] = 0;
        set_timer[s] = 0;
        optvec[s].init(LLC_WAYS - 2);
    }
    std::memset(pc_pred, 0, sizeof(pc_pred));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    for (uint32_t ss = 0; ss < SAMPLER_SETS; ss++) {
        for (uint32_t w = 0; w < SAMPLER_WAYS; w++) {
            sampler[ss][w] = {0,0,0,0,0,0};
        }
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim with bounded aging
    return rrip_victim_and_age(set, current_set);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    // Always update stream detector; used to decide streaming quarantine on fills
    update_stream_detector(PC, paddr);

    // WRITEBACKs do not affect policy
    if (type == ACCESS_WRITEBACK) return;

    // Train Hawkeye-like predictor with sampler on every access (only sampled sets matter)
    hawk_sample(set, paddr, PC, type);

    // On hit: multi-hit gating
    if (hit) {
        if (hitcnt[set][way] == 3) {
            // First demand after prefetch: don't promote yet
            hitcnt[set][way] = 0;
            return;
        }
        if (is_demand(type)) {
            if (hitcnt[set][way] == 0) {
                // first demand hit: slight age reduction but no MRU
                if (rrpv[set][way] > 0) rrpv[set][way]--;
                hitcnt[set][way] = 1;
            } else {
                // 2nd+ demand hit: promote to MRU
                hitcnt[set][way] = 2;
                rrpv[set][way] = 0;
            }
        }
        return;
    }

    // Miss/fill path: attribute bandit score for evicted line before overwrite
    // (only lines inserted by Mode B stream quarantine contribute)
    if (b_tag[set][way]) {
        if (hitcnt[set][way] >= 2) {
            if (bandit_score[set] < BANDIT_MAX) bandit_score[set]++;
        } else {
            if (bandit_score[set] > BANDIT_MIN) bandit_score[set]--;
        }
    }

    // Decide mode by per-set bandit (fallback to Hawkeye unless B is clearly better)
    bool use_modeB = (bandit_score[set] >= BANDIT_THRES);

    // Determine streaming confidence for this PC
    uint32_t pcid = pc_idx(PC);
    bool is_stream = (pc_stream_conf[pcid] >= 2);

    // Insertion policy
    uint8_t new_rrpv = MAX_RRPV;
    uint8_t new_hitcnt = 0;
    uint8_t new_btag = 0;

    if (type == ACCESS_PREFETCH) {
        // Always tail insert; first demand hit does not promote
        new_rrpv = MAX_RRPV;
        new_hitcnt = 3; // prefetch sentinel
        new_btag = (use_modeB && is_stream) ? 1 : 0; // streamed prefetches contribute to B stats
    } else if (use_modeB) {
        if (is_stream) {
            // StreamShield quarantine
            new_rrpv = MAX_RRPV;
            new_hitcnt = 0;
            new_btag = 1;
        } else {
            // Non-stream under Mode B: conservative middle insertion
            new_rrpv = (pred_friend(PC) ? 2 : 6);
            new_hitcnt = 0;
            new_btag = 0;
        }
    } else {
        // Mode A (Hawkeye-like): friendlies higher, averse low
        bool friendly = pred_friend(PC);
        new_rrpv = friendly ? 2 : MAX_RRPV;
        new_hitcnt = 0;
        new_btag = 0;
    }

    // Commit insertion state
    rrpv[set][way] = new_rrpv;
    hitcnt[set][way] = new_hitcnt;
    b_tag[set][way]  = new_btag;
}

void PrintStats_Heartbeat() {}
void PrintStats() {}