#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"
#include "optgen.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (ChampSim CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK);
}

static constexpr uint8_t MAX_RRPV = 7;

// ---------------- Per-line metadata (packed conceptually to 7 bits/line) ----------------
// rrpv: 3b; hitcnt: 2b (0,1,2; 3 used as "prefetch sentinel"); b_tag: 1b (inserted under Mode B stream quarantine);
// young_snap: 1b (epoch snapshot at fill for short-reuse window)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t b_tag[LLC_SETS][LLC_WAYS];
static uint8_t young_snap[LLC_SETS][LLC_WAYS];

// ---------------- Per-set selector (2-bit bandit; enable Mode B if score >= 1) ----------------
static int8_t bandit_score[LLC_SETS]; // store as int8_t, conceptually 2 bits with clamp [-2,+1]
static constexpr int8_t BANDIT_MIN   = -2;
static constexpr int8_t BANDIT_MAX   =  1;
static constexpr int8_t BANDIT_THRES =  1;

// ---------------- Per-set short-reuse epoch (1-bit epoch + 2-bit counter) ----------------
static uint8_t set_epoch_bit[LLC_SETS];  // 0/1
static uint8_t set_epoch_ctr[LLC_SETS];  // 2 bits (0..3)
static constexpr uint8_t EPOCH_PERIOD = 4; // increment bit every 4 touches to this set

// ---------------- Hawkeye-like: small OPTgen + small PC predictor ----------------
// Sampled sets: 16 (very light).
static OPTgen optvec[LLC_SETS];
static uint64_t set_timer[LLC_SETS];

static constexpr uint32_t LLC_SET_BITS = 11;
static inline uint64_t bitmask_u64(uint32_t l) { return (l == 64) ? ~0ULL : ((1ULL << l) - 1ULL); }
static inline uint64_t bits_u64(uint64_t x, uint32_t i, uint32_t l) { return (x >> i) & bitmask_u64(l); }
// 16 sampled sets: lower 4 bits match upper 4 bits of set index
static inline bool SAMPLED_SET(uint32_t set) {
    return bits_u64(set, 0, 4) == bits_u64(set, LLC_SET_BITS - 4, 4);
}
static constexpr uint32_t OPTGEN_VEC_SIZE = 64; // 6-bit quanta

// Small PC predictor (512 entries, 5-bit counter, friendly if ctr >= 16)
static constexpr uint32_t PRED_SIZE = 512;
static uint8_t pc_pred[PRED_SIZE];
static inline uint32_t pred_idx(uint64_t pc) { return (uint32_t)pc & (PRED_SIZE - 1); }
static inline void pred_inc(uint64_t pc) { uint8_t &c = pc_pred[pred_idx(pc)]; if (c < 31) c++; }
static inline void pred_dec(uint64_t pc) { uint8_t &c = pc_pred[pred_idx(pc)]; if (c > 0)  c--; }
static inline bool pred_friend(uint64_t pc) { return pc_pred[pred_idx(pc)] >= 16; }

// History sampler: 512 entries, 8-way, 64 sets
#define SAMPLER_WAYS 8
#define SAMPLER_ENTRIES 512
#define SAMPLER_SETS (SAMPLER_ENTRIES / SAMPLER_WAYS)
struct SampEntry {
    uint8_t  valid;       // 1b
    uint8_t  tag;         // 8b: CRC(paddr>>12) % 256
    uint8_t  lru;         // 3b: 0..7
    uint8_t  pref;        // 1b
    uint16_t pc_sig;      // 9b: index into pc_pred (mask 9 bits)
    uint8_t  last_q;      // 6b: last quanta (mod 64)
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

    uint8_t curr_q = (uint8_t)(set_timer[set] & 0x3F); // mod 64
    set_timer[set]++;

    int32_t hit_way = sampler_find(ss, stag);
    if (hit_way >= 0 && type != ACCESS_PREFETCH) {
        uint8_t last_q = sampler[ss][hit_way].last_q;
        uint8_t dq = (uint8_t)((curr_q >= last_q) ? (curr_q - last_q) : (curr_q + OPTGEN_VEC_SIZE - last_q));
        bool reuse = optvec[set].should_cache(curr_q, last_q) && (dq < OPTGEN_VEC_SIZE);
        if (reuse)
            pred_inc((uint64_t)sampler[ss][hit_way].pc_sig);
        else
            pred_dec((uint64_t)sampler[ss][hit_way].pc_sig);
        sampler_touch_lru(ss, (uint32_t)hit_way, 0);
        sampler[ss][hit_way].pref = 0;
        sampler[ss][hit_way].last_q = curr_q;
        sampler[ss][hit_way].pc_sig = (uint16_t)(pred_idx(PC) & (PRED_SIZE - 1));
    } else {
        uint32_t v = sampler_victim(ss);
        sampler[ss][v].valid  = 1;
        sampler[ss][v].tag    = stag;
        sampler[ss][v].lru    = 0;
        sampler[ss][v].pref   = (type == ACCESS_PREFETCH) ? 1 : 0;
        sampler[ss][v].last_q = curr_q;
        sampler[ss][v].pc_sig = (uint16_t)(pred_idx(PC) & (PRED_SIZE - 1));
        for (uint32_t w = 0; w < SAMPLER_WAYS; w++) {
            if (w == v || !sampler[ss][w].valid) continue;
            sampler[ss][w].lru = (uint8_t)std::min<int>(sampler[ss][w].lru + 1, SAMPLER_WAYS - 1);
        }
    }

    optvec[set].add_access(curr_q);
}

// ---------------- PC forward-run stream detector ----------------
// 256-entry PC table: last line low 10b + 2-bit confidence + 2-bit run + valid
static constexpr uint32_t STREAM_TBL_SIZE = 256;
struct StreamEnt {
    uint16_t last_low; // low 10 bits of line addr
    uint8_t  conf;     // 0..3
    uint8_t  run;      // 0..3 (consecutive +1/+2)
    uint8_t  valid;    // 0/1
};
static StreamEnt stream_tab[STREAM_TBL_SIZE];
static inline uint32_t stream_idx(uint64_t pc) { return (uint32_t)pc & (STREAM_TBL_SIZE - 1); }
static constexpr uint8_t STREAM_RUN_THRES = 2;

static inline bool stream_update(uint64_t PC, uint64_t line_id) {
    uint32_t idx = stream_idx(PC);
    StreamEnt &e = stream_tab[idx];
    uint16_t low = (uint16_t)(line_id & 0x3FF); // low 10 bits
    bool forward = false;
    if (!e.valid) {
        e.valid = 1; e.last_low = low; e.conf = 0; e.run = 0;
    } else {
        // compute delta in small domain (best-effort)
        uint16_t prev = e.last_low;
        uint16_t d = (uint16_t)((low - prev) & 0x3FF);
        if (d == 1 || d == 2) {
            e.run = (uint8_t)std::min<int>(e.run + 1, 3);
            if (e.run >= STREAM_RUN_THRES && e.conf < 3) e.conf++;
            forward = true;
        } else {
            // break the run; decay confidence slightly on clear non-forward
            e.run = 0;
            if (d != 0 && e.conf > 0) e.conf--;
        }
        e.last_low = low;
    }
    return forward;
}

static inline bool stream_confident(uint64_t PC) {
    return stream_tab[stream_idx(PC)].valid && (stream_tab[stream_idx(PC)].conf >= STREAM_RUN_THRES);
}

// ---------------- Tunables (calibrated) ----------------
static constexpr uint8_t HAWK_FRIEND_INSERT = 1; // middle-ish
static constexpr uint8_t HAWK_ENEMY_INSERT  = 7; // tail
static constexpr uint8_t PROBE_MASK = 7;         // 1/8 deterministic probe when Mode B disabled

// Initialize replacement state
void InitReplacementState() {
    std::memset(rrpv, MAX_RRPV, sizeof(rrpv));
    std::memset(hitcnt, 0, sizeof(hitcnt));
    std::memset(b_tag, 0, sizeof(b_tag));
    std::memset(young_snap, 0, sizeof(young_snap));
    std::memset(bandit_score, 0, sizeof(bandit_score));
    std::memset(set_epoch_bit, 0, sizeof(set_epoch_bit));
    std::memset(set_epoch_ctr, 0, sizeof(set_epoch_ctr));
    std::memset(pc_pred, 0, sizeof(pc_pred));
    std::memset(stream_tab, 0, sizeof(stream_tab));
    std::memset(sampler, 0, sizeof(sampler));
    std::memset(set_timer, 0, sizeof(set_timer));
}

// Helper: age RRPV in a set by +1 (saturate at MAX)
static inline void age_set(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] < MAX_RRPV) rrpv[set][w]++;
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
    // SRRIP victim selection: look for MAX_RRPV; if none, age and retry
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == MAX_RRPV) return w;
        }
        age_set(set);
    }
    // unreachable
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
    // Per-set epoch maintenance
    set_epoch_ctr[set] = (uint8_t)((set_epoch_ctr[set] + 1) & (EPOCH_PERIOD - 1));
    if (set_epoch_ctr[set] == 0) set_epoch_bit[set] ^= 1;

    // Train Hawkeye components (cheap)
    hawk_sample(set, paddr, PC, type);

    uint64_t line_id = (paddr >> 6);
    bool fwd = stream_update(PC, line_id);
    bool sconf = stream_confident(PC);

    // On hit: multi-hit gate with short-reuse escape
    if (hit) {
        if (type == ACCESS_WRITEBACK) return; // don't touch RRIP on writeback hits

        // Prefetch sentinel: treat first demand hit as gated (no promotion)
        if (hitcnt[set][way] == 3) {
            hitcnt[set][way] = 1; // quarantine: require another hit before MRU
            if (rrpv[set][way] > 0) rrpv[set][way]--; // mild nudge
            return;
        }

        uint8_t hc = hitcnt[set][way];
        if (hc == 0) {
            hitcnt[set][way] = 1;
            if (rrpv[set][way] > 0) rrpv[set][way]--; // gentle aging
            return;
        } else if (hc == 1) {
            bool stream_ins = (b_tag[set][way] != 0);
            bool young_ok = (young_snap[set][way] == set_epoch_bit[set]); // short-reuse window
            hitcnt[set][way] = 2;

            // Bandit reward: Mode B quarantined line achieved 2nd demand hit
            if (stream_ins && bandit_score[set] < BANDIT_MAX) bandit_score[set]++;

            if (stream_ins && !young_ok) {
                // Stream line, but reuse is late: partial promotion only
                rrpv[set][way] = std::min<uint8_t>(rrpv[set][way], (uint8_t)1);
            } else {
                // Promote to MRU
                rrpv[set][way] = 0;
            }
            return;
        } else {
            // 3rd+ demand hit: strong reuse -> MRU
            hitcnt[set][way] = 2;
            rrpv[set][way] = 0;
            return;
        }
    }

    // Miss/Fill path
    // Before overwriting the victim's metadata, apply bandit penalty if a Mode B line died cold
    if (b_tag[set][way]) {
        if (hitcnt[set][way] < 2 && bandit_score[set] > BANDIT_MIN) bandit_score[set]--;
    }

    // Decide insertion policy
    uint8_t ins_rrpv = MAX_RRPV;
    uint8_t ins_hitcnt = 0;
    uint8_t ins_btag = 0;

    if (type == ACCESS_WRITEBACK) {
        // Never bypass on writeback; insert at tail
        ins_rrpv = MAX_RRPV;
        ins_hitcnt = 0;
        ins_btag = 0;
    } else if (type == ACCESS_PREFETCH) {
        // Prefetch quarantine: always hard-tail; first demand hit will not promote
        ins_rrpv = MAX_RRPV;
        ins_hitcnt = 3; // sentinel
        ins_btag = 0;
    } else {
        // Demand insert
        bool modeB_enabled = (bandit_score[set] >= BANDIT_THRES);
        bool try_modeB = modeB_enabled;

        // Deterministic sparse probing to allow learning when disabled (1/8)
        if (!modeB_enabled) {
            if (((PC ^ set) & PROBE_MASK) == 0) try_modeB = true;
        }

        if (try_modeB && sconf && fwd) {
            // Stream quarantine
            ins_rrpv = MAX_RRPV;
            ins_hitcnt = 0;
            ins_btag = 1;
        } else {
            // Hawkeye-guided insertion
            if (pred_friend(PC)) ins_rrpv = HAWK_FRIEND_INSERT;
            else                 ins_rrpv = HAWK_ENEMY_INSERT;
            ins_hitcnt = 0;
            ins_btag = 0;
        }
    }

    // Commit new line metadata
    rrpv[set][way] = ins_rrpv;
    hitcnt[set][way] = ins_hitcnt;
    b_tag[set][way] = ins_btag;
    young_snap[set][way] = set_epoch_bit[set];
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}