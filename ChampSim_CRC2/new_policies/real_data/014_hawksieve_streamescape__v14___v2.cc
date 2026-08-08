#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader sets: 64 sampled sets for dueling -------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    // 64 leaders: simple bit-mix sampling
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye leader
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamEscape leader
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables (v14) --------------------------------------------
// RRIP
static constexpr uint8_t  maxRRPV              = 7;   // 3b RRIP window

// Mode A (Hawkeye-like insertion)
static constexpr uint8_t  INSERT_WARM_DEPTH    = 2;
static constexpr uint8_t  INSERT_COLD_DEPTH    = 6;
static constexpr uint8_t  FRIENDLY_THRESH      = 16;  // SHCT threshold (0..31)

// Mode B (StreamEscape)
static constexpr uint8_t  STREAM_CONF_THRESH   = 2;   // need 2 forward steps (+1/+2)
static constexpr uint8_t  STREAM_DEMOTE_TOUCH  = 1;   // demote while locked per touch
static constexpr uint8_t  HITS_PROMOTE_NS      = 3;   // non-stream: promote on 3rd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR     = 2;   // stream: escape+promote on 2nd demand hit

// PC tables
static constexpr uint8_t  PC_USE_HOT_THRESH    = 6;   // TinyLFU hot if >=
static constexpr uint8_t  PC_COLD_HIGH         = 2;   // cold if >=

// Selector/decay
static constexpr uint32_t LFU_DECAY_PERIOD     = 2048;
static constexpr uint32_t BANDIT_EPOCH         = 4096;
static constexpr uint8_t  GSEL_MAX             = 31;  // 5-bit selector
static constexpr uint8_t  GSEL_THRES           = 20;  // followers prefer Mode B if >=20 (bias to A)

// ---------------- Per-line metadata (packed conceptually: 3+2+1 bits) -------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];        // 3b RRIP value (0..7)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 2b demand hit count (0..3)
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1b RunLock (stream/pref quarantine)

// ---------------- Global selector -------------------------------------------
static uint8_t GSEL = 0; // 0..31 (defaults to Mode A)
static uint32_t leader_a_acc = 0, leader_b_acc = 0;
static uint32_t leader_a_hits = 0, leader_b_hits = 0;

// ---------------- Hawkeye-like SHCTs (Mode A safety net) --------------------
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1u << SHCT_SIZE_BITS) // 2048
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

// Per-line training in leader-A sets only (64*16 = 1024 slots)
static uint16_t hawk_sig[64][LLC_WAYS];     // store 12b signature in 16b; 0xFFFF == invalid
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch

// ---------------- Mode B PC tables (512-entry) ------------------------------
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)(pc) & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x3FFu); }

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b effective (stored in 16b)
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 2b (0..3)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 4b TinyLFU (0..15)
static uint8_t  pc_cold2[PC_TBL_SIZE];       // 2b (0..3)

// ---------------- Helpers ---------------------------------------------------
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x) { if (x > 0) x--; }
static inline void sat_inc_u5(uint8_t& x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_u5(uint8_t& x) { if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way) {
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline uint16_t pc_sig12(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 4) ^ (pc >> 9) ^ (pc >> 17);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }

// Forward-run detector (check only; update elsewhere)
static inline bool forward_run_now(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t ln  = line10(paddr);
    uint16_t last = pc_last_line10[idx];
    return (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
}
static inline bool stream_confident_now(uint64_t PC) {
    return pc_stream_conf[pc_index(PC)] >= STREAM_CONF_THRESH;
}

// Decay TinyLFU periodically
static uint32_t lfu_decay_ctr = 0;
static inline void maybe_decay_lfu() {
    lfu_decay_ctr++;
    if (lfu_decay_ctr >= LFU_DECAY_PERIOD) {
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
        lfu_decay_ctr = 0;
    }
}

// Update selector at epoch boundaries
static inline void maybe_update_selector() {
    uint32_t total = leader_a_acc + leader_b_acc;
    if (total >= BANDIT_EPOCH) {
        // Compare hit rates without floating point: B better if hits_B * acc_A > hits_A * acc_B
        uint64_t lhs = (uint64_t)leader_b_hits * (uint64_t)(leader_a_acc ? leader_a_acc : 1);
        uint64_t rhs = (uint64_t)leader_a_hits * (uint64_t)(leader_b_acc ? leader_b_acc : 1);
        if (lhs > rhs) {
            sat_inc_u5(GSEL, GSEL_MAX);
        } else {
            sat_dec_u5(GSEL);
        }
        leader_a_acc = leader_b_acc = 0;
        leader_a_hits = leader_b_hits = 0;
    }
}

// ---------------- CRC2 Hooks -------------------------------------------------
void InitReplacementState() {
    // Initialize per-line metadata
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    // Hawkeye tables
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0xFFFFu;
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    // PC tables
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));

    // Selector
    GSEL = 0;
    leader_a_acc = leader_b_acc = 0;
    leader_a_hits = leader_b_hits = 0;
    lfu_decay_ctr = 0;
}

uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    // Prefer any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection: search for maxRRPV, else age and repeat
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) {
                return w;
            }
        }
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
    // Unreachable
    // return 0;
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

    maybe_decay_lfu();

    // Update leader accounting
    if (SEL_SAMPLED(set)) {
        if (LEADER_A(set)) { leader_a_acc++; if (hit) leader_a_hits++; }
        else if (LEADER_B(set)) { leader_b_acc++; if (hit) leader_b_hits++; }
        maybe_update_selector();
    }

    const bool demand = is_demand(type);
    const bool prefetch = is_prefetch(type);
    const bool writeback = is_writeback(type);

    // Stream detection snapshot before updates
    const uint32_t pidx = pc_index(PC);
    const bool fwd = forward_run_now(PC, paddr);
    const bool stream_conf = stream_confident_now(PC);

    // Update PC stream confidence and TinyLFU
    if (demand) {
        if (fwd) sat_inc_u2(pc_stream_conf[pidx]);
        else      sat_dec_u2(pc_stream_conf[pidx]);
        sat_inc_u4(pc_use4[pidx]);
    }

    if (hit) {
        // Leader-A reuse training mark
        if (LEADER_A(set)) {
            hawk_used[LEADER_SLOT(set)][way] = 1;
        }

        // Demand hit reduces coldness; keep scan resistance during lock
        if (demand && pc_cold2[pidx] > 0) pc_cold2[pidx]--;

        if (stream_lock[set][way]) {
            for (uint8_t i = 0; i < STREAM_DEMOTE_TOUCH; i++) rrpv_demote(set, way);
        }

        // Multi-hit promotion gating
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        const uint8_t need = stream_lock[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;

        if (hitcnt[set][way] >= need && demand) {
            stream_lock[set][way] = 0;     // escape if it was locked
            rrpv_set(set, way, 0);         // promote hard to MRU on confirmation
        } else {
            // Do not promote on first hit; keep conservative
        }

        // Update last line10
        pc_last_line10[pidx] = line10(paddr);
        return;
    }

    // Miss path: train Hawkeye on the evicted line in leader-A sets
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t olds = hawk_sig[slot][way];
        if (olds != 0xFFFFu) {
            uint32_t idx = shct_idx(olds);
            if (hawk_used[slot][way]) {
                if (hawk_is_pref[slot][way]) { if (shct_prefetch[idx] < SHCT_MAX) shct_prefetch[idx]++; }
                else                         { if (shct_demand[idx]   < SHCT_MAX) shct_demand[idx]++; }
            } else {
                if (hawk_is_pref[slot][way]) { if (shct_prefetch[idx] > 0) shct_prefetch[idx]--; }
                else                         { if (shct_demand[idx]   > 0) shct_demand[idx]--; }
            }
        }
        // Install new signature metadata
        hawk_sig[slot][way] = pc_sig12(PC);
        hawk_used[slot][way] = 0;
        hawk_is_pref[slot][way] = prefetch ? 1 : 0;
    }

    // Decide which mode governs insertion for this access
    bool use_mode_b;
    if (LEADER_B(set))      use_mode_b = true;
    else if (LEADER_A(set)) use_mode_b = false;
    else                    use_mode_b = (GSEL >= GSEL_THRES);

    // Per-access stream override
    if (fwd || stream_conf) use_mode_b = true;

    // PC-based heat/cold
    const bool pc_hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
    const bool pc_cold = (pc_cold2[pidx] >= PC_COLD_HIGH);

    // Choose insertion RRPV and quarantine
    uint8_t ins_rrpv = maxRRPV;
    uint8_t lock = 0;

    if (writeback) {
        // Never bypass on WB: keep low priority
        ins_rrpv = maxRRPV;
        lock = 0;
    } else if (prefetch) {
        // Prefetch quarantine at tail
        ins_rrpv = maxRRPV;
        lock = 1;
    } else if (!use_mode_b) {
        // Mode A (Hawkeye-like)
        uint8_t depth = INSERT_COLD_DEPTH;
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        if (shct_demand[idx] >= FRIENDLY_THRESH) depth = INSERT_WARM_DEPTH;
        ins_rrpv = depth;
        lock = 0;
    } else {
        // Mode B (StreamEscape)
        if ((fwd && stream_conf) || pc_cold) {
            // Strong stream or cold PC: hard tail with lock
            ins_rrpv = maxRRPV;
            lock = 1;
        } else if (fwd || stream_conf) {
            // developing run: still quarantine
            ins_rrpv = maxRRPV;
            lock = 1;
        } else {
            // not streamy; respect PC heat
            ins_rrpv = pc_hot ? 2 : 6;
            lock = 0;
        }
    }

    rrpv_set(set, way, ins_rrpv);
    hitcnt[set][way] = 0;
    stream_lock[set][way] = lock;

    // Demand miss increases coldness slightly (helps dead-guard for noisy PCs)
    if (demand) sat_inc_u2(pc_cold2[pidx]);

    // Update last observed line for run detection
    pc_last_line10[pidx] = line10(paddr);
}

void PrintStats() {
    // intentionally empty
}

void PrintStats_Heartbeat() {
    // intentionally empty
}