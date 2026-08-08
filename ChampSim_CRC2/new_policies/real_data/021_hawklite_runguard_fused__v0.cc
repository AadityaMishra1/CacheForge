#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access helpers
static inline bool is_demand(uint32_t t)    { return (t == LOAD) || (t == RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == WRITEBACK); }

// ---------------- Tunables ----------------
static constexpr uint8_t maxRRPV            = 7;  // 3b
static constexpr uint8_t INSERT_WARM_DEPTH  = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH  = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH  = 7;  // hard tail quarantine
static constexpr uint8_t HITS_PROMOTE_NS    = 2;  // promote on 2nd demand hit (non-stream)
static constexpr uint8_t HITS_PROMOTE_STR   = 3;  // stream escape on 3rd demand hit
static constexpr uint8_t STREAM_CONF_THRESH = 2;  // +1/+2 forward steps
static constexpr uint8_t PC_USE_HOT_THRESH  = 7;  // TinyLFU hot threshold (0..15)
static constexpr int8_t  MODEB_ENABLE_THRESH= 2;  // bandit gate for Mode B
static constexpr uint32_t LFU_DECAY_PERIOD  = 4096;

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Per-line metadata (1 byte/line) -----------------
// Layout: bits [2:0]=rrpv, [4:3]=hitcnt (0..3), [5]=stream_tag, [7:6]=unused
static uint8_t meta[LLC_SETS][LLC_WAYS];
static inline uint8_t get_rrpv(uint8_t m)    { return m & 0x7u; }
static inline uint8_t set_rrpv(uint8_t m, uint8_t v){ return (uint8_t)((m & ~0x7u) | (v & 0x7u)); }
static inline uint8_t get_hit(uint8_t m)     { return (m >> 3) & 0x3u; }
static inline uint8_t set_hit(uint8_t m, uint8_t v){ return (uint8_t)((m & ~(0x3u<<3)) | ((v & 0x3u) << 3)); }
static inline bool get_stream(uint8_t m)     { return (m >> 5) & 0x1u; }
static inline uint8_t set_stream(uint8_t m, bool v){ return (uint8_t)((m & ~(1u<<5)) | ((v?1u:0u) << 5)); }

// ---------------- Mode A: Hawkeye-lite SHiP -----------------------
#define SHCT_SIZE_BITS 10
#define SHCT_SIZE (1u << SHCT_SIZE_BITS)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b sig
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// ---------------- Mode B: RunGuard-LFU-lite -----------------------
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15

// ---------------- Selector ----------------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7]
static bool    prefer_B = false;       // global gate from leader scores
static int32_t leaderA_score = 0;
static int32_t leaderB_score = 0;
static uint64_t access_count = 0;

static inline bool modeB_enabled(uint32_t set) {
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline bool detect_stream(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    bool fwd = false;
    if (last != 0xFFFFu) {
        uint16_t exp1 = (uint16_t)(last + 1);
        uint16_t exp2 = (uint16_t)(last + 2);
        fwd = (ln == exp1) || (ln == exp2);
    }
    pc_last_line10[idx] = ln;
    if (fwd) sat_inc_u2(pc_stream_conf[idx]); else pc_stream_conf[idx] = 0;
    return pc_stream_conf[idx] >= STREAM_CONF_THRESH;
}

static inline void decay_lfu() {
    access_count++;
    if ((access_count & (LFU_DECAY_PERIOD - 1u)) == 0) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
    }
}

// ---------------- Init -------------------------------------------
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        bandit_score[s] = 0;
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w] = set_rrpv(0, maxRRPV);
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    std::memset(hawk_sig, 0, sizeof(hawk_sig));
    std::memset(hawk_used, 0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
        pc_last_line10[i] = 0xFFFFu;
        pc_stream_conf[i] = 0;
        pc_use4[i] = 0;
    }
    prefer_B = false;
    leaderA_score = 0;
    leaderB_score = 0;
    access_count = 0;
}

// ---------------- Victim selection -------------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    (void)cpu; (void)PC; (void)paddr; (void)type;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP search with bounded aging
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (get_rrpv(meta[set][w]) == maxRRPV) return w;
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            uint8_t m = meta[set][w];
            uint8_t r = get_rrpv(m);
            if (r < maxRRPV) meta[set][w] = set_rrpv(m, (uint8_t)(r + 1));
        }
    }
    // Fallback: pick highest RRPV
    uint32_t vic = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint8_t r = get_rrpv(meta[set][w]);
        if (r >= best) { best = r; vic = w; }
    }
    return vic;
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    (void)cpu; (void)victim_addr;

    decay_lfu();
    bool leaderA = LEADER_A(set);
    bool leaderB = LEADER_B(set);
    uint32_t lslot = LEADER_SLOT(set);
    bool mB = modeB_enabled(set);
    uint32_t pidx = pc_index(PC);
    bool is_stream = !is_writeback(type) && detect_stream(PC, paddr);

    // Leader training on eviction (before overwrite)
    if (!hit && (leaderA || leaderB)) {
        uint16_t prev_sig = hawk_sig[lslot][way];
        uint8_t prev_used = hawk_used[lslot][way];
        uint8_t prev_pref = hawk_is_pref[lslot][way];
        if (prev_sig) {
            if (prev_pref) {
                if (prev_used) shct_inc(shct_prefetch[shct_idx(prev_sig)]);
                else           shct_dec(shct_prefetch[shct_idx(prev_sig)]);
            } else {
                if (prev_used) shct_inc(shct_demand[shct_idx(prev_sig)]);
                else           shct_dec(shct_demand[shct_idx(prev_sig)]);
            }
            // Selector feedback
            if (leaderA) {
                if (prev_used) leaderA_score++; else leaderA_score--;
            } else {
                if (prev_used) leaderB_score++; else leaderB_score--;
            }
        }
        hawk_sig[lslot][way] = 0;
        hawk_used[lslot][way] = 0;
        hawk_is_pref[lslot][way] = 0;
    }

    // Update bandit/global gate periodically
    if ((access_count & (BANDIT_EPOCH - 1u)) == 0) {
        prefer_B = (leaderB_score > leaderA_score);
        leaderA_score = leaderB_score = 0;
    }
    if (!SEL_SAMPLED(set)) {
        if (prefer_B) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);
    }

    uint8_t m = meta[set][way];

    if (hit) {
        // Demand hits: track reuse
        if (is_demand(type)) {
            uint8_t hc = get_hit(m);
            if (hc < 3) hc++;
            m = set_hit(m, hc);
            // TinyLFU warmup
            sat_inc_u4(pc_use4[pidx]);
        }

        if (mB) {
            // Stream demotion on touch
            if (get_stream(m) && STREAM_DEMOTE_TOUCH) {
                uint8_t r = get_rrpv(m);
                if (r < maxRRPV) m = set_rrpv(m, (uint8_t)(r + 1));
            }
            // Promotion gate
            uint8_t need = get_stream(m) ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
            if (is_demand(type) && get_hit(m) >= need) {
                m = set_rrpv(m, 0);
                m = set_stream(m, false);
            }
        } else {
            // Mode A promote on demand hit
            if (is_demand(type)) m = set_rrpv(m, 0);
        }
        if (leaderA || leaderB) hawk_used[lslot][way] = 1;
        meta[set][way] = m;
        return;
    }

    // Miss path: reset per-line fields
    m = set_hit(0, 0);
    m = set_stream(m, false);

    if (mB) {
        if (type == PREFETCH) {
            m = set_rrpv(m, STREAM_TAIL_DEPTH);
            m = set_stream(m, true);
        } else if (is_stream) {
            m = set_rrpv(m, STREAM_TAIL_DEPTH);
            m = set_stream(m, true);
        } else if (is_writeback(type)) {
            m = set_rrpv(m, INSERT_WARM_DEPTH);
        } else {
            bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            m = set_rrpv(m, hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
        }
    } else {
        // Mode A Hawkeye-lite: use SHCT demand prediction (prefetch separate)
        if (type == PREFETCH) {
            m = set_rrpv(m, STREAM_TAIL_DEPTH);
        } else {
            uint16_t sig = pc_sig12(PC);
            uint8_t conf = shct_demand[shct_idx(sig)];
            m = set_rrpv(m, (conf >= (SHCT_MAX >> 1)) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH);
        }
    }

    // Leader metadata install
    if (leaderA || leaderB) {
        uint16_t sig = pc_sig12(PC);
        hawk_sig[lslot][way] = sig;
        hawk_used[lslot][way] = 0;
        hawk_is_pref[lslot][way] = (type == PREFETCH) ? 1 : 0;
    }

    meta[set][way] = m;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}
