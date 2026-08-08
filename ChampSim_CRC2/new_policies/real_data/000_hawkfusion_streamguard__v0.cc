#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define OPTGEN_VECTOR_SIZE 128

// Simple CRC32 helper used for PC signatures (matches Hawkeye implementation)
static inline uint64_t CRC(uint64_t addr) {
    static const uint64_t poly = 0xEDB88320ULL;
    uint64_t val = addr;
    for (unsigned i = 0; i < 32; i++)
        val = (val & 1u) ? ((val >> 1) ^ poly) : (val >> 1);
    return val;
}

#include "optgen.h"

// ChampSim CRC2 constants
#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2-aligned)
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t type) {
    return (type != ACCESS_PREFETCH) && (type != ACCESS_WRITEBACK);
}

// RRIP parameters (3-bit)
static constexpr uint8_t maxRRPV = 7;

// Per-line metadata (bit-packed conceptually): rrpv(3b), quarantine(1b), seen_once(1b)
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t quarantine[LLC_SETS][LLC_WAYS]; // 1 = quarantined (stream/prefetch)
static uint8_t seen_once[LLC_SETS][LLC_WAYS];  // 1 = already had 1 demand hit

// --------- Hawkeye-like sampling (Mode A leader and follower guidance) ---------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)

// 64 sampled sets: ((low6)==(next6))
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// Per-set timers and OPTgen (only meaningful on sampled sets)
static constexpr uint32_t TIMER_SIZE = 1024;
static uint64_t perset_mytimer[LLC_SETS];
static OPTgen perset_optgen[LLC_SETS];

// Signatures for sampled sets (12-bit; stored in 16-bit)
static uint16_t signatures[LLC_SETS][LLC_WAYS];

// Sampler (fixed array, no dynamic allocation in hot path)
static constexpr uint32_t SAMPLED_CACHE_ENTRIES = 2048; // trimmed for budget
static constexpr uint32_t SAMPLER_WAYS = 8;
static constexpr uint32_t SAMPLER_SETS = SAMPLED_CACHE_ENTRIES / SAMPLER_WAYS;

struct SamplerEntry {
    uint8_t  valid;
    uint8_t  lru;           // 0..(SAMPLER_WAYS-1)
    uint8_t  tag;           // 8-bit tag (CRC hash)
    uint16_t sig;           // 12-bit signature (stored in 16b)
    uint64_t last_quanta;   // last access time (quanta)
    uint8_t  prefetched;    // 1 if last fill was prefetch
};
static SamplerEntry sampler[SAMPLER_SETS][SAMPLER_WAYS];

// Tiny unified PC-usefulness predictor (1K entries x 5b)
static constexpr uint32_t PRED_SIZE = 1024;
static uint8_t pc_use[PRED_SIZE]; // 0..31
static inline uint32_t pred_idx(uint64_t pc_sig) { return (uint32_t)pc_sig & (PRED_SIZE - 1); }
static inline void pred_inc(uint32_t idx) { if (pc_use[idx] < 31) pc_use[idx]++; }
static inline void pred_dec(uint32_t idx) { if (pc_use[idx] > 0)  pc_use[idx]--; }

// 12-bit PC signature helper (CRC over PC bits -> 12b)
static inline uint16_t pc_sig12(uint64_t PC) {
    // Fold CRC helper above to 12 bits
    return (uint16_t)(CRC(PC) & 0xFFFu);
}

// --------- Mode B: PC-based stream detection (forward-run) ---------
static constexpr uint32_t PC_TBL_SIZE = 128;
static uint16_t pc_last_line[PC_TBL_SIZE]; // 12-bit line index of last touch
static uint8_t  pc_conf[PC_TBL_SIZE];      // 2-bit forward-run confidence (0..3)
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1); }
static inline uint16_t line12(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x0FFFu); }
static inline void sat_inc_u2(uint8_t& x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t& x) { if (x > 0) x--; }

// --------- Hashed bandit selector (followers) ----------
static constexpr uint32_t BANDIT_SIZE = 128;
static int8_t bandit[BANDIT_SIZE]; // signed, saturating [-7..7]
static inline uint32_t bandit_idx(uint32_t set) {
    // simple mix of bits
    return ((set ^ (set >> 4) ^ (set >> 7)) & (BANDIT_SIZE - 1));
}
static inline void bandit_inc(uint32_t set) { if (bandit[bandit_idx(set)] < 7) bandit[bandit_idx(set)]++; }
static inline void bandit_dec(uint32_t set) { if (bandit[bandit_idx(set)] > -7) bandit[bandit_idx(set)]--; }
static constexpr int8_t BANDIT_THRESH = 2; // enable Mode B for followers if >= 2

// Tunables
static constexpr uint8_t FRIENDLY_THRESH = 20; // Mode-A friendly if pc_use >= 20
static constexpr uint8_t INSERT_A_FRIENDLY = 2; // near-MRU
static constexpr uint8_t INSERT_A_COLD     = 7; // tail
static constexpr uint8_t INSERT_B_WARM     = 6; // warm tail for cold-but-not-stream
static constexpr uint8_t STREAM_CONF_THRESH = 2; // need >=2 for stream mark

// Initialize replacement state
void InitReplacementState() {
    // Per-line meta
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = maxRRPV;
            quarantine[s][w] = 0;
            seen_once[s][w] = 0;
            signatures[s][w] = 0;
        }
        perset_mytimer[s] = 0;
        perset_optgen[s].init(LLC_WAYS - 2);
    }

    // Sampler arrays
    for (uint32_t ss = 0; ss < SAMPLER_SETS; ss++) {
        for (uint32_t w = 0; w < SAMPLER_WAYS; w++) {
            sampler[ss][w].valid = 0;
            sampler[ss][w].lru = w;
            sampler[ss][w].tag = 0;
            sampler[ss][w].sig = 0;
            sampler[ss][w].last_quanta = 0;
            sampler[ss][w].prefetched = 0;
        }
    }

    // Predictors and bandit
    std::memset(pc_use, 0, sizeof(pc_use));
    std::memset(bandit, 0, sizeof(bandit));
    std::memset(pc_last_line, 0, sizeof(pc_last_line));
    std::memset(pc_conf, 0, sizeof(pc_conf));
}

// RRIP victim selection with bounded aging
static inline uint32_t rrip_victim_and_age(uint32_t set, const BLOCK* current_set) {
    // Return any invalid way immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Try maxRRPV
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] == maxRRPV) return w;
    }
    // Age up to 8 passes
    for (int pass = 0; pass < 8; pass++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (rrpv[set][w] == maxRRPV) return w;
        for (uint32_t w = 0; w < LLC_WAYS; w++)
            if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
    // Fallback: select the most aged
    uint32_t victim = 0; uint8_t best = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (rrpv[set][w] >= best) { best = rrpv[set][w]; victim = w; }
    }
    return victim;
}

// Hawkeye sampled training (array-based sampler; no dynamic allocations)
static inline void sampler_access(uint32_t set, uint64_t paddr, uint64_t PC, uint32_t type) {
    if (!SEL_SAMPLED(set)) return;

    uint64_t& timer = perset_mytimer[set];
    uint64_t curr_quanta = timer % OPTGEN_VECTOR_SIZE;

    // Sampler indices/tags (8-bit tag)
    uint32_t samp_set = (uint32_t)((paddr >> 6) % SAMPLER_SETS);
    uint8_t  samp_tag = (uint8_t)(CRC(paddr >> 12) & 0xFFu);
    SamplerEntry* set_entries = sampler[samp_set];

    // Look for tag
    int hit_way = -1;
    for (int w = 0; w < (int)SAMPLER_WAYS; w++) {
        if (set_entries[w].valid && set_entries[w].tag == samp_tag) { hit_way = w; break; }
    }

    if (hit_way != -1 && type != ACCESS_PREFETCH) {
        // Train predictor using OPTgen "should_cache" based on last_quanta
        uint64_t last_q = set_entries[hit_way].last_quanta;
        uint64_t curr_t = timer;
        if (curr_t < last_q) curr_t += TIMER_SIZE;
        bool wrap = ((curr_t - last_q) > OPTGEN_VECTOR_SIZE);
        if (!wrap) {
            bool cache_it = perset_optgen[set].should_cache(curr_quanta, (last_q % OPTGEN_VECTOR_SIZE));
            uint32_t idx = pred_idx(set_entries[hit_way].sig);
            if (cache_it) pred_inc(idx); else pred_dec(idx);
        }
        // Update LRU and metadata
        for (int w = 0; w < (int)SAMPLER_WAYS; w++) if (set_entries[w].lru < set_entries[hit_way].lru) set_entries[w].lru++;
        set_entries[hit_way].lru = 0;
        set_entries[hit_way].last_quanta = timer;
        set_entries[hit_way].prefetched = 0;
    } else if (hit_way == -1) {
        // Replace LRU
        int lru_way = 0; uint8_t max_lru = 0;
        for (int w = 0; w < (int)SAMPLER_WAYS; w++) {
            if (!set_entries[w].valid) { lru_way = w; max_lru = SAMPLER_WAYS - 1; break; }
            if (set_entries[w].lru >= max_lru) { max_lru = set_entries[w].lru; lru_way = w; }
        }
        for (int w = 0; w < (int)SAMPLER_WAYS; w++) if (set_entries[w].valid && set_entries[w].lru < max_lru) set_entries[w].lru++;
        set_entries[lru_way].valid = 1;
        set_entries[lru_way].lru = 0;
        set_entries[lru_way].tag = samp_tag;
        set_entries[lru_way].sig = pc_sig12(PC);
        set_entries[lru_way].last_quanta = timer;
        set_entries[lru_way].prefetched = (type == ACCESS_PREFETCH) ? 1 : 0;
    }

    // Update OPTgen occupancy vector for this access
    perset_optgen[set].add_access(curr_quanta);

    // Advance time
    timer = (timer + 1) % TIMER_SIZE;
}

// Stream detector update (forward-run +1/+2)
static inline bool update_stream_conf(uint64_t PC, uint64_t paddr) {
    uint32_t idx = pc_index(PC);
    uint16_t cur = line12(paddr);
    int16_t delta = (int16_t)cur - (int16_t)pc_last_line[idx];
    bool forward = (delta == 1) || (delta == 2);
    if (forward) sat_inc_u2(pc_conf[idx]); else sat_dec_u2(pc_conf[idx]);
    pc_last_line[idx] = cur;
    return (pc_conf[idx] >= STREAM_CONF_THRESH);
}

// Find victim in the set
uint32_t GetVictimInSet(
    uint32_t cpu,
    uint32_t set,
    const BLOCK *current_set,
    uint64_t PC,
    uint64_t paddr,
    uint32_t type
) {
    (void)cpu; (void)PC; (void)paddr; (void)type;

    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    return rrip_victim_and_age(set, current_set);
}

// Update replacement state
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

    // Always update the Hawkeye sampler first (leaders + followers on sampled sets)
    sampler_access(set, paddr, PC, type);

    // Ignore writebacks for policy actions
    if (type == ACCESS_WRITEBACK) return;

    // On hit: multi-hit gating
    if (hit) {
        if (is_demand(type)) {
            if (seen_once[set][way] == 0) {
                seen_once[set][way] = 1;
                // Keep quarantine if any, and avoid aggressive promotion
                if (!quarantine[set][way]) {
                    // small move toward MRU
                    if (rrpv[set][way] > 2) rrpv[set][way] = 2;
                }
            } else {
                // Confirmed multi-hit: promote and clear quarantine
                rrpv[set][way] = 0;
                quarantine[set][way] = 0;
            }
        }
        return;
    }

    // Miss fill: compute insertion using Mode A or Mode B
    // Eviction feedback for leader-B sets (bandit): attribute the old line outcome
    if (LEADER_B(set)) {
        // If the victim was valid and died without a re-reference, penalize; else reward
        // We don't have explicit valid flag here; current_set isn't provided in Update.
        // Use seen_once as proxy if the filled way held a live line.
        if (seen_once[set][way] == 0) bandit_dec(set);
        else                          bandit_inc(set);
    }

    // Update stream detector and read predictor
    bool is_stream = update_stream_conf(PC, paddr);
    uint16_t sig12 = pc_sig12(PC);
    uint32_t pidx = pred_idx(sig12);
    uint8_t strength = pc_use[pidx];

    // Choose mode for insertion
    bool use_modeB = false;
    if (LEADER_A(set)) use_modeB = false;
    else if (LEADER_B(set)) use_modeB = true;
    else use_modeB = (bandit[bandit_idx(set)] >= BANDIT_THRESH);

    // Insertion policy
    uint8_t ins_rrpv = maxRRPV;
    uint8_t ins_quar = 0;
    if (!use_modeB) {
        // Mode A: Hawkeye-guided insertion depth
        if (strength >= FRIENDLY_THRESH) ins_rrpv = INSERT_A_FRIENDLY;
        else                             ins_rrpv = INSERT_A_COLD;
        ins_quar = (type == ACCESS_PREFETCH) ? 1 : 0; // prefetch quarantine
    } else {
        // Mode B: StreamGuard
        if (type == ACCESS_PREFETCH) {
            ins_rrpv = maxRRPV; ins_quar = 1; // prefetch lines tail + quarantine
        } else if (is_stream) {
            ins_rrpv = maxRRPV; ins_quar = 1; // stream hard-tail quarantine (bypass-equivalent)
        } else if (strength < FRIENDLY_THRESH) {
            ins_rrpv = INSERT_B_WARM; ins_quar = 0; // cold-ish PCs: warm-tail
        } else {
            ins_rrpv = INSERT_A_FRIENDLY; ins_quar = 0; // confirmed useful
        }
    }

    // Install metadata
    rrpv[set][way] = ins_rrpv;
    quarantine[set][way] = ins_quar;
    seen_once[set][way] = 0;
    if (SEL_SAMPLED(set)) signatures[set][way] = sig12;
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}
