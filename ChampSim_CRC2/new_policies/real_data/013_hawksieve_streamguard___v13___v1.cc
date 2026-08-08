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

// ---------- Leader sets: 64 sampled sets for dueling ----------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye leader
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamGuard+ leader
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables ----------------
static constexpr uint8_t  maxRRPV              = 7;  // 3b RRIP window
// Mode A (Hawkeye-like)
static constexpr uint8_t  INSERT_WARM_DEPTH    = 2;
static constexpr uint8_t  INSERT_COLD_DEPTH    = 6;
static constexpr uint8_t  FRIENDLY_THRESH      = 16; // SHCT threshold (0..31)
// Mode B (StreamGuard+)
static constexpr uint8_t  STREAM_CONF_THRESH   = 2;  // 2 forward steps (+1/+2)
static constexpr uint8_t  STREAM_DEMOTE_TOUCH  = 1;  // demote by 1 while locked
static constexpr uint8_t  HITS_PROMOTE_NS      = 2;  // 2nd demand hit to promote (non-stream)
static constexpr uint8_t  HITS_PROMOTE_STR     = 2;  // 2nd demand hit to unlock+promote (stream)
// PC tables
static constexpr uint8_t  PC_USE_HOT_THRESH    = 6;  // TinyLFU hot if >=
static constexpr uint8_t  PC_COLD_HIGH         = 2;  // cold if >=
// Selector/decay
static constexpr uint32_t LFU_DECAY_PERIOD     = 2048;
static constexpr uint32_t BANDIT_EPOCH         = 4096;
static constexpr uint8_t  GSEL_MAX             = 31;
static constexpr uint8_t  GSEL_THRES           = 16; // followers prefer Mode B if >=16

// ---------------- Per-line metadata (packed conceptually: 3+2+1 bits) -------
static uint8_t rrpv[LLC_SETS][LLC_WAYS];        // 3b RRIP value (0..7)
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];      // 2b demand hit count (0..3)
static uint8_t stream_lock[LLC_SETS][LLC_WAYS]; // 1b RunLock (stream/pref quarantine)

// ---------------- Global selector ----------------
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

// ---------------- Helpers ----------------
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
        if (lhs > rhs) sat_inc_u5(GSEL, GSEL_MAX);
        else           sat_dec_u5(GSEL);
        leader_a_acc = leader_b_acc = leader_a_hits = leader_b_hits = 0;
    }
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            hitcnt[s][w] = 0;
            stream_lock[s][w] = 0;
        }
    }
    std::memset(shct_demand, 0, sizeof(shct_demand));
    std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
    for (uint32_t ls = 0; ls < 64; ls++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            hawk_sig[ls][w] = 0xFFFFu; // invalid
            hawk_used[ls][w] = 0;
            hawk_is_pref[ls][w] = 0;
        }
    }
    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4, 0, sizeof(pc_use4));
    std::memset(pc_cold2, 0, sizeof(pc_cold2));
    GSEL = 0;
    leader_a_acc = leader_b_acc = leader_a_hits = leader_b_hits = 0;
    lfu_decay_ctr = 0;
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
    // Return an invalid way immediately if present
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // RRIP victim selection: search for maxRRPV, aging if needed
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age all ways toward eviction (bounded)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
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
    bool demand = is_demand(type);
    bool prefetch = is_prefetch(type);
    bool writeback = is_writeback(type);

    // Leader accounting for selector
    if (LEADER_A(set)) {
        leader_a_acc++;
        if (hit) leader_a_hits++;
    } else if (LEADER_B(set)) {
        leader_b_acc++;
        if (hit) leader_b_hits++;
    }
    maybe_update_selector();

    // PC tables: TinyLFU and coldness update on demand accesses
    if (demand) {
        sat_inc_u4(pc_use4[pc_index(PC)]);
        if (hit) sat_dec_u2(pc_cold2[pc_index(PC)]);
        else     sat_inc_u2(pc_cold2[pc_index(PC)]);
        maybe_decay_lfu();
    }

    // Stream detector update (on demand only)
    if (demand) {
        uint32_t idx = pc_index(PC);
        uint16_t ln  = line10(paddr);
        bool fwd = forward_run_now(PC, paddr);
        if (fwd) {
            if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
        } else {
            if (pc_stream_conf[idx] > 0) pc_stream_conf[idx]--;
        }
        pc_last_line10[idx] = ln;
    }

    // Promotion on hit with multi-hit gating and RunLock for streams/prefetch
    if (hit) {
        // Training reuse in leader-A sets
        if (LEADER_A(set)) {
            uint32_t ls = LEADER_SLOT(set);
            hawk_used[ls][way] = 1;
        }

        if (demand) {
            uint8_t& hc = hitcnt[set][way];
            bool locked = (stream_lock[set][way] != 0);
            // Increment hit counter (saturate 2b)
            if (hc < 3) hc++;

            if (locked) {
                if (hc >= HITS_PROMOTE_STR) {
                    stream_lock[set][way] = 0;
                    rrpv_set(set, way, 0); // escape quarantine
                } else {
                    if (STREAM_DEMOTE_TOUCH) rrpv_demote(set, way);
                    // no promotion on first hit while locked
                }
            } else {
                if (hc >= HITS_PROMOTE_NS) {
                    rrpv_set(set, way, 0); // MRU on confirmed reuse
                } else {
                    // mild age reduction discouraged: keep as-is to enforce 2nd-hit policy
                }
            }
        }
        return;
    }

    // Miss/Fill path: decide mode and insertion depth
    bool forced_stream = demand && stream_confident_now(PC) && forward_run_now(PC, paddr);
    bool use_modeB;
    if (LEADER_B(set)) use_modeB = true;
    else if (LEADER_A(set)) use_modeB = false;
    else if (forced_stream) use_modeB = true;      // stream override
    else                    use_modeB = (GSEL >= GSEL_THRES);

    uint8_t new_rrpv = maxRRPV; // default tail
    uint8_t new_lock = 0;
    uint8_t new_hitcnt = 0;

    if (use_modeB) {
        if (prefetch) {
            new_rrpv = maxRRPV;
            new_lock = 1; // quarantine prefetches
        } else if (forced_stream && !writeback) {
            new_rrpv = maxRRPV; // hard-tail
            new_lock = 1;       // RunLock: forbid 1st-hit promotion
        } else if (writeback) {
            new_rrpv = INSERT_COLD_DEPTH; // never bypass writebacks
            new_lock = 0;
        } else {
            // Demand, non-stream: PC-guided hot/cold
            uint32_t pidx = pc_index(PC);
            bool pc_hot  = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
            bool pc_cold = (pc_cold2[pidx] >= PC_COLD_HIGH);
            if (pc_hot)      new_rrpv = INSERT_WARM_DEPTH; // near-MRU for hot PCs
            else if (pc_cold) new_rrpv = INSERT_COLD_DEPTH; // deep insertion for noisy PCs
            else              new_rrpv = INSERT_COLD_DEPTH - 1; // moderate tail
        }
    } else {
        // Mode A (Hawkeye-like)
        if (prefetch) {
            new_rrpv = maxRRPV;
            new_lock = 1; // quarantine prefetches
        } else if (writeback) {
            new_rrpv = INSERT_COLD_DEPTH;
        } else {
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            bool friendly = (shct_demand[idx] >= FRIENDLY_THRESH);
            new_rrpv = friendly ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        }
    }

    // Train SHCT on eviction for leader-A sets (use previous occupant's record)
    if (LEADER_A(set)) {
        uint32_t ls = LEADER_SLOT(set);
        uint16_t psig = hawk_sig[ls][way];
        if (psig != 0xFFFFu) {
            uint32_t idx = shct_idx(psig);
            if (hawk_is_pref[ls][way]) {
                if (hawk_used[ls][way]) { if (shct_prefetch[idx] < SHCT_MAX) shct_prefetch[idx]++; }
                else { if (shct_prefetch[idx] > 0) shct_prefetch[idx]--; }
            } else {
                if (hawk_used[ls][way]) { if (shct_demand[idx] < SHCT_MAX) shct_demand[idx]++; }
                else { if (shct_demand[idx] > 0) shct_demand[idx]--; }
            }
        }
        // Record new fill's signature and type
        hawk_sig[ls][way] = pc_sig12(PC);
        hawk_used[ls][way] = 0;
        hawk_is_pref[ls][way] = prefetch ? 1 : 0;
    }

    // Commit insertion metadata
    rrpv_set(set, way, new_rrpv);
    stream_lock[set][way] = new_lock;
    hitcnt[set][way] = new_hitcnt;
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}