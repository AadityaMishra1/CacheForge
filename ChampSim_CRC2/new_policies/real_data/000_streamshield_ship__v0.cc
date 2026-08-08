#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access types (CRC2 typical mapping)
enum AccessType : uint32_t { LOAD=0, RFO=1, PREFETCH=2, WRITEBACK=3 };

// 2-bit RRIP per line
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
// 8-bit PC-signature index per line (for SHCT update)
static uint8_t line_sig[LLC_SETS][LLC_WAYS];
// 2-bit multi-hit counter per line: 0=no hit yet, 1=one hit, 2+=multi-hit
static uint8_t line_hits[LLC_SETS][LLC_WAYS];
// Our mirror of valid state to know if an eviction occurred at fill time
static uint8_t line_valid[LLC_SETS][LLC_WAYS];

// SHiP: Signature History Counter Table (2-bit saturating counters)
static uint8_t shct[256];

// Stream tracker entry keyed by 8-bit PC-signature
struct StreamEntry {
    uint16_t last_block_low; // 12 LSBs of block number (paddr>>6)
    uint8_t  conf;           // 2-bit sequentiality confidence
};
static StreamEntry stream_tab[256];

static inline uint8_t sat_inc2(uint8_t v) { return (v < 3) ? (uint8_t)(v + 1) : v; }
static inline uint8_t sat_dec2(uint8_t v) { return (v > 0) ? (uint8_t)(v - 1) : v; }

static inline uint8_t pc_sig_idx(uint64_t PC, uint32_t set) {
    // Lightweight hash to reduce aliasing across sets
    uint64_t x = PC ^ (PC >> 2) ^ (PC >> 11) ^ set;
    return (uint8_t)(x & 0xFF);
}

static inline uint16_t block_low(uint64_t paddr) {
    // 12-bit block index low bits (64B lines)
    return (uint16_t)((paddr >> 6) & 0xFFF);
}

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            rrpv[s][w] = 3;       // distant
            line_sig[s][w] = 0;
            line_hits[s][w] = 0;
            line_valid[s][w] = 0; // mirror invalid
        }
    }
    // Initialize SHCT to weakly-used to avoid over-bypass at startup
    for (uint32_t i = 0; i < 256; i++) {
        shct[i] = 1;
        stream_tab[i].last_block_low = 0;
        stream_tab[i].conf = 0;
    }
}

// Find victim in the set (RRIP)
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

    // RRIP victim selection
    while (true) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] == 3) return w;
        }
        // Age all lines (saturating at 3)
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            if (rrpv[set][w] < 3) rrpv[set][w]++;
        }
    }
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
    const bool is_wb  = (type == WRITEBACK);
    const bool is_pf  = (type == PREFETCH);
    const uint8_t sig = pc_sig_idx(PC, set);
    const uint16_t blk = block_low(paddr);

    // Streaming detection (per-PC) using 64B sequential steps with 2+ confidence
    bool stream_now = false;
    if (!is_wb) {
        StreamEntry &e = stream_tab[sig];
        const uint16_t prev = e.last_block_low;
        const bool seq = (((uint16_t)(prev + 1) & 0x0FFF) == blk);
        // Require at least one prior sequential step, then seeing another -> stream
        stream_now = (seq && (e.conf >= 1));
        // Update tracker
        e.conf = seq ? sat_inc2(e.conf) : 0;
        e.last_block_low = blk;
    }

    if (hit) {
        // Multi-hit confirmation: never promote on first hit or for streaming PCs
        uint8_t hc = line_hits[set][way];
        if (hc < 2) line_hits[set][way] = (uint8_t)(hc + 1);

        if (!stream_now && line_hits[set][way] >= 2) {
            // Promote confirmed multi-hit to MRU (RRPV=0)
            rrpv[set][way] = 0;
        }
        // Else: keep current RRPV (no promotion on first hit; never promote streaming)
        return;
    }

    // Miss: potential eviction update for old occupant
    if (line_valid[set][way]) {
        uint8_t old_sig = line_sig[set][way];
        uint8_t old_hc  = line_hits[set][way];
        if (old_hc == 0) shct[old_sig] = sat_dec2(shct[old_sig]); // dead -> colder
        else             shct[old_sig] = sat_inc2(shct[old_sig]); // reused -> hotter
    }

    // Decide insertion RRPV
    uint8_t ins_rrpv = 2; // default middle
    if (is_wb) {
        ins_rrpv = 2; // never bypass on writeback
    } else if (is_pf) {
        ins_rrpv = 3; // prefetch low priority
    } else if (stream_now) {
        ins_rrpv = 3; // aggressive stream shielding
    } else {
        uint8_t hval = shct[sig];
        if (hval == 0)      ins_rrpv = 3; // predicted-cold -> distant insertion
        else if (hval == 1) ins_rrpv = 2; // weak
        else                ins_rrpv = 1; // likely-reuse -> closer insertion
    }

    // Install new line metadata
    rrpv[set][way]     = ins_rrpv;
    line_sig[set][way] = sig;
    line_hits[set][way]= 0;     // no hit yet
    line_valid[set][way]= 1;    // now valid
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}