#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE        1
#define LLC_SETS        (NUM_CORE * 2048)
#define LLC_WAYS        16

// RRIP parameters
static const uint8_t  RRPV_BITS         = 2;
static const uint8_t  MAX_RRPV          = (1 << RRPV_BITS) - 1; // 3

// Signature history counter table (2-bit saturating counters)
#define SHCT_SIZE        8192
#define SHCT_MAX         3
#define SHCT_INIT        1  // weakly not-predicted
#define SHCT_THRESHOLD   2  // >=2 means predict reuse

// Per-line metadata
static uint8_t  rrpv_array[LLC_SETS][LLC_WAYS];
static uint32_t sig_array[LLC_SETS][LLC_WAYS];

// Global predictor
static uint8_t  SHCT[SHCT_SIZE];

// Stats
static uint64_t hit_count, miss_count;

// Helper: RRIP victim selection with on-demand aging
static uint32_t FindVictimWay(uint32_t set) {
  while (true) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv_array[set][w] == MAX_RRPV) return w;
    }
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv_array[set][w] < MAX_RRPV)
        rrpv_array[set][w]++;
    }
  }
}

void InitReplacementState() {
  // Initialize RRPVs to LRU
  for (uint32_t s = 0; s < LLC_SETS; s++)
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      rrpv_array[s][w] = MAX_RRPV;
      sig_array[s][w]  = 0;
    }
  // Init SHCT to weakly not-predicted
  for (uint32_t i = 0; i < SHCT_SIZE; i++)
    SHCT[i] = SHCT_INIT;
  hit_count  = 0;
  miss_count = 0;
}

uint32_t GetVictimInSet(
    uint32_t        cpu,
    uint32_t        set,
    const BLOCK    *current_set,
    uint64_t        PC,
    uint64_t        paddr,
    uint32_t        type
) {
  (void)cpu; (void)current_set; (void)paddr; (void)type; (void)PC;
  return FindVictimWay(set);
}

void UpdateReplacementState(
    uint32_t  cpu,
    uint32_t  set,
    uint32_t  way,
    uint64_t  paddr,
    uint64_t  PC,
    uint64_t  victim_addr,
    uint32_t  type,
    uint8_t   hit
) {
  (void)cpu; (void)paddr; (void)victim_addr; (void)type;
  // Compute PC-based signature
  uint32_t sig = (uint32_t)((PC >> 4) & (SHCT_SIZE - 1));
  if (hit) {
    // Hit: promote to MRU and increment predictor
    hit_count++;
    rrpv_array[set][way] = 0;
    if (SHCT[sig] < SHCT_MAX) SHCT[sig]++;
  } else {
    // Miss: we'll be inserting a new line at (set,way)
    miss_count++;
    // First, decrement predictor for evicted block's signature
    uint32_t victim_sig = sig_array[set][way];
    if (SHCT[victim_sig] > 0) SHCT[victim_sig]--;
    // Now choose insertion position based on current PC prediction
    bool predict_reuse = (SHCT[sig] >= SHCT_THRESHOLD);
    rrpv_array[set][way] = predict_reuse ? 0 : MAX_RRPV;
    // Save this block's signature for future eviction updates
    sig_array[set][way] = sig;
  }
}

void PrintStats() {
  uint64_t total = hit_count + miss_count;
  double   hr    = total ? (double)hit_count * 100.0 / total : 0.0;
  std::cout << "SHiP Hits=" << hit_count
            << " Misses=" << miss_count
            << " HitRate=" << hr << "%\n";
}

void PrintStats_Heartbeat() {
  // No periodic output
}