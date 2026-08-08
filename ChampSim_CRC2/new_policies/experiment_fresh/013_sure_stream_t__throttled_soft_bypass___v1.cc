#include <cstdint>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

#ifndef LOAD
#define LOAD 0
#endif
#ifndef RFO
#define RFO 1
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif
#ifndef WRITEBACK
#define WRITEBACK 3
#endif

// ---------------- Tunables (refined) ----------------
#define PC_SIG_ENTRIES 1024      // 10-bit index
#define PILOT_SETS 128           // expanded pilot coverage
#define PILOT_TAG_BITS 16
#define RUN_THR 2                // base +1 stride runlength for streaming
#define TTL_MAX 3
#define DECAY_PERIOD 4096
#define MAB_ENTRIES 256
#define MAB_COOL_INIT 3

// ---------------- Per-line metadata (conceptually 11 bits/line) ----------------
struct LineMeta {
    uint8_t lru;     // 0..15 (0=MRU)
    uint8_t ttl;     // 0..3
    uint8_t mhits;   // 0..3 demand-hit count
    uint8_t pf;      // 0/1 prefetch-quarantined
    uint8_t stream;  // 0/1 stream-clamped (no promotion until >=2 demand hits)
    uint8_t cold;    // 0/1 predicted cold at fill
};
static LineMeta meta[LLC_SETS][LLC_WAYS];

// ---------------- PC⊕page sentinel + utility ----------------
static uint16_t pcsig_last_line[PC_SIG_ENTRIES]; // low12 of last line number
static uint8_t  pcsig_run[PC_SIG_ENTRIES];       // 0..3 runlength of +1 strides
static uint8_t  pcsig_util[PC_SIG_ENTRIES];      // 0..3 SHiP-like utility

// ---------------- Pilot sampler (128 sampled sets × 16 ways) ----------------
static uint16_t pilot_tag[PILOT_SETS][LLC_WAYS];
static uint16_t pilot_pcidx[PILOT_SETS][LLC_WAYS]; // stores 10-bit index
static uint8_t  pilot_valid[PILOT_SETS][LLC_WAYS];
static uint8_t  pilot_used[PILOT_SETS][LLC_WAYS];

// ---------------- Miss-After-Bypass throttle (direct-mapped 256 entries) ----
static uint8_t mab_tag[MAB_ENTRIES];  // upper 2 bits of 10-bit pc-sig index
static uint8_t mab_cool[MAB_ENTRIES]; // 0..3 cooldown (active if >0)

// ---------------- Time ----------------
static uint64_t event_ctr = 0;
static uint32_t decay_ptr = 0;

// ---------------- Helpers ----------------
static inline bool is_demand(uint32_t t) { return (t == LOAD) || (t == RFO); }
static inline bool is_demandish(uint32_t t) { return (t == LOAD) || (t == RFO) || (t == PREFETCH); }
static inline uint64_t line_number(uint64_t paddr) { return (paddr >> 6); }  // 64B lines
static inline uint64_t page_number(uint64_t paddr) { return (paddr >> 12); }

static inline uint32_t pc_sig_index(uint64_t PC, uint64_t paddr) {
    uint64_t page = page_number(paddr);
    uint64_t mix = (PC ^ (page << 1) ^ (page >> 1));
    mix ^= (mix >> 13);
    mix *= 0x9e3779b97f4a7c15ULL;
    mix ^= (mix >> 17);
    return static_cast<uint32_t>(mix) & (PC_SIG_ENTRIES - 1);
}
static inline uint16_t line_sig16(uint64_t line) {
    uint64_t x = line * 0x9e3779b97f4a7c15ULL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    return static_cast<uint16_t>(x & ((1u << PILOT_TAG_BITS) - 1));
}
static inline bool is_pilot_set(uint32_t set) {
    // 2048 sets / 128 = 16: choose every 16th set
    return ((set % (LLC_SETS / PILOT_SETS)) == 0);
}
static inline uint32_t pilot_set_index(uint32_t set) {
    return (set / (LLC_SETS / PILOT_SETS)) & (PILOT_SETS - 1);
}

// LRU helpers
static inline void lru_make_mru(uint32_t set, uint32_t way) {
    uint8_t old = meta[set][way].lru;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru < old) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = 0;
}
static inline void lru_insert_pos(uint32_t set, uint32_t way, uint8_t pos) {
    if (pos >= LLC_WAYS) pos = LLC_WAYS - 1;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (w == way) continue;
        if (meta[set][w].lru <= pos) {
            uint8_t np = meta[set][w].lru + 1;
            meta[set][w].lru = (np > (LLC_WAYS - 1)) ? (LLC_WAYS - 1) : np;
        }
    }
    meta[set][way].lru = pos;
}

// Periodic light decay (smooths run/utility and MAB cooldown)
static inline void periodic_decay() {
    if ((event_ctr & (DECAY_PERIOD - 1)) != 0) return;

    // Decay a handful of pcsig entries
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t idx = (decay_ptr + (i * 31)) & (PC_SIG_ENTRIES - 1);
        if (pcsig_run[idx] > 0) pcsig_run[idx]--;
        if (pcsig_util[idx] > 0) pcsig_util[idx]--;
    }
    // Decay a few MAB cooldowns
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t midx = (decay_ptr + (i * 17)) & (MAB_ENTRIES - 1);
        if (mab_cool[midx] > 0) mab_cool[midx]--;
    }
    decay_ptr = (decay_ptr + 1) & (PC_SIG_ENTRIES - 1);
}

// Streaming sentinel update; returns streaming prediction for this access
static inline bool update_stream_sentinel(uint32_t pcsi, uint64_t line, uint32_t type) {
    if (type == WRITEBACK) return false;
    uint16_t cur = static_cast<uint16_t>(line & 0x0FFFu); // low 12 bits
    uint16_t prev = pcsig_last_line[pcsi];
    uint16_t delta = static_cast<uint16_t>((cur - prev) & 0x0FFFu);

    if (delta == 1) {
        if (pcsig_run[pcsi] < 3) pcsig_run[pcsi]++;
    } else if (cur != prev) {
        if (pcsig_run[pcsi] > 0) pcsig_run[pcsi]--;
    }
    pcsig_last_line[pcsi] = cur;

    return (pcsig_run[pcsi] >= RUN_THR);
}

// MAB throttle helpers
static inline void mab_arm(uint32_t pcsi) {
    uint32_t idx = pcsi & (MAB_ENTRIES - 1);
    uint8_t tag = static_cast<uint8_t>(pcsi >> 8); // upper 2 bits
    mab_tag[idx] = tag;
    mab_cool[idx] = MAB_COOL_INIT;
}
static inline bool mab_throttled(uint32_t pcsi) {
    uint32_t idx = pcsi & (MAB_ENTRIES - 1);
    uint8_t tag = static_cast<uint8_t>(pcsi >> 8);
    if ((mab_tag[idx] == tag) && (mab_cool[idx] > 0)) {
        // Do not decrement here; periodic decay handles cooldown
        return true;
    }
    return false;
}

// Victim scoring: higher => better eviction candidate
static inline uint32_t score_line(const LineMeta &m) {
    uint32_t s = 0;
    s += (m.lru * 4);                                // older preferred
    s += (3 - m.ttl) * 3;                            // low TTL preferred
    s += (m.mhits == 0 ? 3 : (m.mhits == 1 ? 1 : 0));// zero/single-hit preferred
    s += (m.pf ? 2 : 0);                             // prefetch quarantined
    s += (m.stream ? 4 : 0);                         // stream-clamped preferred
    s += (m.cold ? 1 : 0);                           // predicted cold preferred
    return s;
}

// Initialize replacement state
void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            meta[s][w].lru = static_cast<uint8_t>(w);
            meta[s][w].ttl = 0;
            meta[s][w].mhits = 0;
            meta[s][w].pf = 0;
            meta[s][w].stream = 0;
            meta[s][w].cold = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIG_ENTRIES; i++) {
        pcsig_last_line[i] = 0;
        pcsig_run[i] = 0;
        pcsig_util[i] = 1; // neutral start
    }
    for (uint32_t ps = 0; ps < PILOT_SETS; ps++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            pilot_tag[ps][w] = 0;
            pilot_pcidx[ps][w] = 0;
            pilot_valid[ps][w] = 0;
            pilot_used[ps][w] = 0;
        }
    }
    for (uint32_t i = 0; i < MAB_ENTRIES; i++) {
        mab_tag[i] = 0;
        mab_cool[i] = 0;
    }
    event_ctr = 0;
    decay_ptr = 0;
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
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }

    // Light TTL aging on older half of the stack
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (meta[set][w].ttl > 0 && meta[set][w].lru >= (LLC_WAYS / 2)) {
            meta[set][w].ttl--;
        }
    }

    // Choose best-scoring victim
    uint32_t best_way = 0;
    uint32_t best_score = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        uint32_t sc = score_line(meta[set][w]);
        if (w == 0 || sc > best_score) {
            best_score = sc;
            best_way = w;
        }
    }
    return best_way; // replaced block index
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
    event_ctr++;
    periodic_decay();

    uint64_t line = line_number(paddr);
    uint32_t pcsi = pc_sig_index(PC, paddr);

    // Update streaming sentinel on any demandish access; writebacks ignored
    bool stream_now = update_stream_sentinel(pcsi, line, type);

    if (hit) {
        // Pilot: record use on hit
        if (is_pilot_set(set)) {
            uint32_t ps = pilot_set_index(set);
            pilot_used[ps][way] = 1;
        }

        if (type == WRITEBACK) {
            // Do not alter policy for writeback hits
            return;
        }

        // Demand or demandish hit handling
        bool demand = is_demand(type);
        if (demand && meta[set][way].mhits < 3) meta[set][way].mhits++;

        // Demote prefetch quarantining only after first demand hit
        if (meta[set][way].pf && demand) {
            meta[set][way].pf = 0;
        }

        // Stream-clamped lines: only promote after >=2 demand hits
        if (meta[set][way].stream) {
            if (demand && meta[set][way].mhits >= 2) {
                // Clear stream clamp, arm MAB, and promote
                meta[set][way].stream = 0;
                meta[set][way].cold = 0;
                mab_arm(pcsi);
                meta[set][way].ttl = (meta[set][way].ttl < TTL_MAX) ? (uint8_t)2 : meta[set][way].ttl;
                lru_make_mru(set, way);
            } else {
                // Very light promotion (or none)
                uint8_t cur = meta[set][way].lru;
                if (cur > 0) lru_insert_pos(set, way, cur - 1);
            }
            return;
        }

        // Normal (non-stream) line promotion with multi-hit gating
        if (demand) {
            if (meta[set][way].mhits >= 2 || pcsig_util[pcsi] >= 2) {
                if (meta[set][way].ttl < TTL_MAX) meta[set][way].ttl++;
                meta[set][way].cold = 0;
                lru_make_mru(set, way);
            } else {
                // Modest promotion
                uint8_t cur = meta[set][way].lru;
                if (cur > 1) lru_insert_pos(set, way, cur - 2);
                else lru_make_mru(set, way);
            }
        } else {
            // Prefetch hit or similar: minimal movement
            uint8_t cur = meta[set][way].lru;
            if (cur > 0) lru_insert_pos(set, way, cur - 1);
        }
        return;
    }

    // Miss / Fill path
    // Pilot update on eviction of this way
    if (is_pilot_set(set)) {
        uint32_t ps = pilot_set_index(set);
        if (pilot_valid[ps][way]) {
            uint32_t idx = pilot_pcidx[ps][way] & (PC_SIG_ENTRIES - 1);
            if (pilot_used[ps][way]) {
                if (pcsig_util[idx] < 3) pcsig_util[idx]++;
            } else {
                if (pcsig_util[idx] > 0) pcsig_util[idx]--;
            }
        }
        // Install new pilot entry
        pilot_tag[ps][way] = line_sig16(line);
        pilot_pcidx[ps][way] = static_cast<uint16_t>(pcsi);
        pilot_valid[ps][way] = 1;
        pilot_used[ps][way] = 0;
    }

    // Predict streaming and coldness
    bool throttled = mab_throttled(pcsi);
    bool streaming_pred = stream_now && !throttled;
    // If throttled, raise the bar by one more run step
    if (throttled && pcsig_run[pcsi] < (RUN_THR + 1)) streaming_pred = false;

    bool coldPred = (pcsig_util[pcsi] <= 1);
    bool is_pf = (type == PREFETCH);

    // Initialize line metadata on fill
    meta[set][way].mhits = 0;
    meta[set][way].pf = is_pf ? 1 : 0;
    meta[set][way].cold = coldPred ? 1 : 0;

    if (type == WRITEBACK) {
        // Never bypass writebacks: tail-ish insertion with small TTL
        meta[set][way].stream = 0;
        meta[set][way].ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 2);
        return;
    }

    if (streaming_pred && coldPred) {
        // Aggressive soft-bypass: tail insert, ttl=0, no promotion
        meta[set][way].stream = 1;
        meta[set][way].ttl = 0;
        lru_insert_pos(set, way, LLC_WAYS - 1);
    } else {
        meta[set][way].stream = 0;
        uint8_t base_ttl = (pcsig_util[pcsi] >= 2) ? 2 : ((pcsig_util[pcsi] == 1) ? 1 : 0);
        if (is_pf) base_ttl = 0; // quarantine prefetch fills
        meta[set][way].ttl = (base_ttl > TTL_MAX) ? TTL_MAX : base_ttl;

        // Adaptive insertion position
        uint8_t pos;
        if (is_pf) {
            pos = LLC_WAYS - 2; // deep tail for prefetches
        } else if (coldPred) {
            pos = LLC_WAYS - 3; // near-LRU for cold
        } else {
            pos = 3; // close to MRU for hot
        }
        lru_insert_pos(set, way, pos);
    }
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}