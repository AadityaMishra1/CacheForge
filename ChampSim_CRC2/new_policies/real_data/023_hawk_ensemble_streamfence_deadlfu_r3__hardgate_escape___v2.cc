#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types per CRC2
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t){ return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t){ return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ----------------
static constexpr uint32_t LLC_SET_BITS = 11;
static inline bool SEL_SAMPLED(uint32_t set){
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 0u); } // Hawkeye-like
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u) == 1u); } // StreamFence

static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ---------------- Tunables (R3) --------------------------------
static constexpr uint8_t  maxRRPV               = 7;   // 3-bit SRRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;   // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;   // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;   // hard tail for confident streams
static constexpr uint8_t  STREAM_CONF_THRESH    = 2;   // need two forward steps (+1/+2)
static constexpr uint8_t  HITS_PROMOTE_NS       = 2;   // non-stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;   // stream: promote on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR_HOT  = 1;   // hot stream: promote on 1st demand hit
static constexpr uint8_t  PC_USE_HOT_THRESH     = 7;   // TinyLFU hot threshold (0..15)
static constexpr uint8_t  PC_DEAD_GATE_TH       = 3;   // 3-bit dead gate (>=3 => dead/cold)
static constexpr uint32_t LFU_DECAY_PERIOD      = 2048;// faster decay for stability
static constexpr uint32_t BANDIT_EPOCH          = 8192;// shorter selector epoch
static constexpr uint8_t  MODEB_ENABLE_THRESH   = 3;   // per-set score to enable Mode B

// ---------------- Per-line metadata (bit-packed logically) -----
// rrpv:3b, hitcnt:2b, stream_tag:1b, was_pref:1b => 7 bits/line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS];
static uint8_t was_pref[LLC_SETS][LLC_WAYS];

// ---------------- Per-set bandit selector (2-bit logical) ------
static int8_t bandit_score[LLC_SETS]; // 0..3 used, default 0

// ---------------- Mode A (Hawkeye-like) ------------------------
// Tiny SHCT with 12-bit PC signatures; train only on leader-A sets
#define SHCT_ENTRIES 512
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_ENTRIES];
static uint8_t shct_prefetch[SHCT_ENTRIES];

static inline uint16_t pc_sig12(uint64_t pc){
    uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
    return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_ENTRIES - 1u); }
static inline void shct_inc(uint8_t& x){ if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t& x){ if (x > 0) x--; }

// Leader per-line buffers (64 leader sets x 16 ways)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12-bit effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamFence + DeadLFU) ---------------
static constexpr uint32_t PC_TBL_SIZE = 256;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b effective
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3 (2b logical)
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (4b TinyLFU)
static uint8_t  pc_dead3[PC_TBL_SIZE];       // 0..7  (3b logical)

// ---------------- Stats/selector state -------------------------
static uint64_t access_count = 0;
static uint32_t leader_req_A = 0, leader_hit_A = 0;
static uint32_t leader_req_B = 0, leader_hit_B = 0;

// ---------------- Helpers --------------------------------------
static inline void sat_inc_u2(uint8_t& x){ if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x){ if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t& x){ if (x < 7) x++; }
static inline void sat_dec_u3(uint8_t& x){ if (x > 0) x--; }
static inline void sat_inc_u4(uint8_t& x){ if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t& x){ if (x > 0) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val){
    rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_promote(uint32_t set, uint32_t way){
    if (rrpv[set][way] > 0) rrpv[set][way]--;
}
static inline void rrpv_demote(uint32_t set, uint32_t way){
    if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
}

static inline bool modeB_enabled(uint32_t set){
    if (LEADER_B(set)) return true;
    if (LEADER_A(set)) return false;
    return (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

static inline bool is_stream_pc(uint64_t PC, uint64_t paddr){
    uint32_t idx = pc_index(PC);
    uint16_t cur = line10(paddr);
    uint16_t prev = pc_last_line10[idx];
    int16_t diff = (int16_t)(cur - prev);
    bool forward = (diff == 1) || (diff == 2);
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else         sat_dec_u2(pc_stream_conf[idx]);
    pc_last_line10[idx] = cur;
    return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

static inline void lfu_touch(uint64_t PC, uint32_t type){
    if (is_demand(type)) {
        uint32_t idx = pc_index(PC);
        sat_inc_u4(pc_use4[idx]);
    }
}

static inline bool lfu_hot(uint64_t PC){
    return pc_use4[pc_index(PC)] >= PC_USE_HOT_THRESH;
}

static inline void lfu_decay_maybe(){
    if ((access_count & (LFU_DECAY_PERIOD - 1u)) == 0){
        for (uint32_t i = 0; i < PC_TBL_SIZE; i++){
            sat_dec_u4(pc_use4[i]);
        }
    }
}

// Deterministic 75% gate (no RNG) for stream soft-bypass emulation
static inline bool bypass75(uint64_t PC, uint64_t paddr, uint64_t tick){
    // 3 out of 4 values map to true
    uint64_t h = (PC >> 2) ^ (paddr >> 6) ^ (tick);
    return ((h & 3u) != 0u);
}

static inline bool pc_dead(uint64_t PC){
    return (pc_dead3[pc_index(PC)] >= PC_DEAD_GATE_TH);
}
static inline void pc_dead_inc(uint64_t PC){ sat_inc_u3(pc_dead3[pc_index(PC)]); }
static inline void pc_dead_dec(uint64_t PC){ sat_dec_u3(pc_dead3[pc_index(PC)]); }

// ---------------- Core API -------------------------------------
void InitReplacementState() {
    std::memset(rrpv,        0, sizeof(rrpv));
    std::memset(hitcnt,      0, sizeof(hitcnt));
    std::memset(stream_tag,  0, sizeof(stream_tag));
    std::memset(was_pref,    0, sizeof(was_pref));
    std::memset(bandit_score,0, sizeof(bandit_score));

    for (uint32_t s = 0; s < LLC_SETS; s++)
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            rrpv[s][w] = maxRRPV;

    std::memset(shct_demand,  0, sizeof(shct_demand));
    std::memset(shct_prefetch,0, sizeof(shct_prefetch));
    std::memset(hawk_sig,     0, sizeof(hawk_sig));
    std::memset(hawk_used,    0, sizeof(hawk_used));
    std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));

    std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
    std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
    std::memset(pc_use4,        0, sizeof(pc_use4));
    std::memset(pc_dead3,       0, sizeof(pc_dead3));

    access_count = 0;
    leader_req_A = leader_hit_A = 0;
    leader_req_B = leader_hit_B = 0;
}

uint32_t GetVictimInSet(
    uint32_t /*cpu*/,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t /*PC*/,
    uint64_t /*paddr*/,
    uint32_t /*type*/
) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // SRRIP victim search
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == maxRRPV) return w;
        }
        // Age everyone one step (bounded)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
        }
    }
}

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
    access_count++;
    lfu_touch(PC, type);
    lfu_decay_maybe();

    // Leader accounting (demand requests only)
    if (is_demand(type)) {
        if (LEADER_A(set)) {
            leader_req_A++;
            if (hit) leader_hit_A++;
        } else if (LEADER_B(set)) {
            leader_req_B++;
            if (hit) leader_hit_B++;
        }
    }

    // Periodic bandit decision (defaults to Mode A, conservative)
    if ((access_count % BANDIT_EPOCH) == 0) {
        // Compare leader hit-rates (avoid div)
        uint64_t a_num = (uint64_t)leader_hit_A * (uint64_t)(1 + leader_req_B);
        uint64_t b_num = (uint64_t)leader_hit_B * (uint64_t)(1 + leader_req_A);
        bool b_better = (b_num > a_num);
        for (uint32_t s = 0; s < LLC_SETS; s++) {
            if (b_better) { if (bandit_score[s] < 3) bandit_score[s]++; }
            else          { if (bandit_score[s] > 0) bandit_score[s]--; }
        }
        leader_req_A = leader_hit_A = 0;
        leader_req_B = leader_hit_B = 0;
    }

    // Stream detection updates per access
    bool stream_pred = is_stream_pc(PC, paddr);
    bool hot_pc      = lfu_hot(PC);

    if (hit) {
        // Demand hit counting
        if (is_demand(type)) {
            if (hitcnt[set][way] < 3) hitcnt[set][way]++;
            // PC dead decreases on realized reuse
            pc_dead_dec(PC);
        }

        // Multi-hit promotion gate with stream escape and prefetch quarantine
        uint8_t need_hits = (stream_tag[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS);
        if (stream_tag[set][way] && hot_pc) {
            // Early escape for hot streams
            need_hits = HITS_PROMOTE_STR_HOT;
        }
        if (was_pref[set][way]) {
            // quarantine depth=2 => require one extra hit
            if (need_hits < 3) need_hits++;
        }

        if (is_demand(type) && (hitcnt[set][way] >= need_hits)) {
            // Promote once and clear stream/quarantine flags
            rrpv_set(set, way, 0);
            stream_tag[set][way] = 0;
            was_pref[set][way]   = 0;
        } else {
            // Default hit handling
            if (stream_tag[set][way]) {
                // Demote-on-touch for guarded streams
                rrpv_demote(set, way);
            } else {
                // Mild promotion (but not full MRU)
                rrpv_promote(set, way);
            }
        }

        // Hawkeye reuse mark for leader-A
        if (LEADER_A(set)) {
            uint32_t slot = LEADER_SLOT(set);
            hawk_used[slot][way] = 1;
        }

        return;
    }

    // Miss/Fill path: prepare insertion
    bool use_modeB = modeB_enabled(set);
    uint8_t ins_depth = INSERT_COLD_DEPTH;
    uint8_t new_stream_tag = 0;
    uint8_t new_was_pref   = is_prefetch(type) ? 1 : 0;

    // Hawkeye training on leader-A: update outgoing victim info before overwrite
    if (LEADER_A(set)) {
        uint32_t slot = LEADER_SLOT(set);
        uint16_t osig = hawk_sig[slot][way];
        uint8_t  ouse = hawk_used[slot][way];
        uint8_t  opre = hawk_is_pref[slot][way];
        if (osig) {
            uint32_t idx = shct_idx(osig);
            if (ouse) {
                if (opre) shct_inc(shct_prefetch[idx]);
                else      shct_inc(shct_demand[idx]);
            } else {
                if (opre) shct_dec(shct_prefetch[idx]);
                else      shct_dec(shct_demand[idx]);
                // dead signal for Mode B approximation (no reuse)
                pc_dead_inc(PC);
            }
        }
        // Install new leader metadata
        hawk_sig[slot][way]     = pc_sig12(PC);
        hawk_used[slot][way]    = 0;
        hawk_is_pref[slot][way] = new_was_pref;
    }

    // Decide insertion depth
    if (is_writeback(type)) {
        // Never bypass on WB; keep cold
        ins_depth = STREAM_TAIL_DEPTH;
        new_stream_tag = 0;
    } else if (is_prefetch(type)) {
        // Always tail; quarantine
        ins_depth = STREAM_TAIL_DEPTH;
        new_stream_tag = stream_pred ? 1 : 0;
    } else {
        // Demand
        if (use_modeB) {
            bool dead_cold = pc_dead(PC) && !hot_pc;
            if (stream_pred) {
                // Strong stream fence: always hard-tail; emulate 75% bypass via hard-tail + no early promotion
                ins_depth = STREAM_TAIL_DEPTH;
                new_stream_tag = 1;
                (void)bypass75(PC, paddr, access_count); // deterministic gate (state-less in CRC2)
            } else if (dead_cold) {
                ins_depth = STREAM_TAIL_DEPTH;
                new_stream_tag = 1; // guard like stream
            } else {
                ins_depth = INSERT_WARM_DEPTH; // normal reuse-friendly
                new_stream_tag = 0;
            }
        } else {
            // Mode A (Hawkeye-like)
            uint16_t sig = pc_sig12(PC);
            uint32_t idx = shct_idx(sig);
            uint8_t conf = shct_demand[idx];
            if (conf > (SHCT_MAX >> 1)) ins_depth = INSERT_WARM_DEPTH;
            else                        ins_depth = INSERT_COLD_DEPTH;
            new_stream_tag = 0; // stream gating via Mode B only
        }
    }

    // Install metadata
    rrpv_set(set, way, ins_depth);
    hitcnt[set][way]     = 0;
    stream_tag[set][way] = new_stream_tag;
    was_pref[set][way]   = new_was_pref;
}

void PrintStats() {}
void PrintStats_Heartbeat() {}