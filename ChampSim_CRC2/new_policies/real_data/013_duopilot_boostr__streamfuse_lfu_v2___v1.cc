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

// ---------------- Tunables (retuned) ----------------
static constexpr uint8_t maxRRPV               = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH     = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH     = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH     = 7;  // hard tail quarantine

// Stream run detector: +1/+2 forward steps
static constexpr uint8_t STREAM_ARM_THRESH     = 2;  // arm after 2 forward steps
static constexpr uint8_t STREAM_BYPASS_THRESH  = 3;  // sustain 3rd step -> keep hard-tail (pseudo-bypass)

// Multi-hit promotions
static constexpr uint8_t HITS_PROMOTE_STR      = 2;  // stream/prefetch: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_NS_HOT   = 2;  // non-stream hot PC
static constexpr uint8_t HITS_PROMOTE_NS_COLD  = 3;  // non-stream cold PC

static constexpr bool    STREAM_DEMOTE_TOUCH   = true; // demote quarantined streams on touch

// TinyLFU gate (PC coldness)
static constexpr uint8_t  PC_USE_HOT_THRESH    = 10;    // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD     = 1024;  // faster decay

// Selector/leader gates
static constexpr uint32_t BANDIT_EPOCH         = 4096;
static constexpr int8_t   MODEB_ENABLE_THRESH  = 2;     // per-set bandit threshold
static constexpr int32_t  GATE_MARGIN          = 6;     // strong bias to Hawkeye

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
  // Sample 64 sets: low 6 bits == next 6 bits
  return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Per-line metadata (conceptually bit-packed) -----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];     // demand hits only (0..3)
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1=quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], defaults to 0 (favor A)
static bool    prefer_B = false;       // global gate from leaders
static int32_t leaderA_good = 0;       // hits - misses on leader A sets
static int32_t leaderB_good = 0;       // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-lite via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch; trained only in leaders
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
  uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
  return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-only per-line training buffers (64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (StreamFuse-LFU PC state) ----------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF=uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3
static uint8_t  pc_use4[PC_TBL_SIZE];        // 0..15 (TinyLFU)

// ---------------- Helpers ----------------------------------------
static inline void sat_inc_u2(uint8_t &x) { if (x < 3) x++; }
static inline void sat_dec_u2(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_u3(uint8_t &x) { if (x < 7) x++; }
static inline void sat_inc_u4(uint8_t &x) { if (x < 15) x++; }
static inline void sat_dec_u4(uint8_t &x) { if (x > 0) x--; }
static inline void sat_inc_i8(int8_t &x)  { if (x < 7) x++; }
static inline void sat_dec_i8(int8_t &x)  { if (x > -8) x--; }

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val) {
  rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set) {
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
  }
}

static inline uint8_t rrip_pick_victim(uint32_t set, const BLOCK* current_set) {
  // If any invalid way exists, return it immediately
  for (uint8_t w = 0; w < LLC_WAYS; w++) {
    if (!current_set[w].valid) return w;
  }
  // RRIP victim search with safe aging
  while (true) {
    for (uint8_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] == maxRRPV) return w;
    }
    rrpv_age_all(set); // age and retry; saturates at maxRRPV
  }
}

// Stream detector (+1/+2 forward)
static inline uint8_t update_stream_conf(uint64_t PC, uint64_t paddr) {
  uint32_t idx = pc_index(PC);
  uint16_t ln  = line10(paddr);
  uint16_t last = pc_last_line10[idx];
  bool forward = false;
  if (last != 0xFFFFu) {
    uint16_t exp1 = (uint16_t)(uint16_t)(last + 1);
    uint16_t exp2 = (uint16_t)(uint16_t)(last + 2);
    forward = (ln == exp1) || (ln == exp2);
  }
  if (forward) {
    if (pc_stream_conf[idx] < 3) pc_stream_conf[idx]++;
  } else {
    pc_stream_conf[idx] = 0;
  }
  pc_last_line10[idx] = ln;
  return pc_stream_conf[idx];
}

static inline bool modeB_enabled(uint32_t set) {
  if (LEADER_B(set)) return true;    // force Mode B on B leaders
  if (LEADER_A(set)) return false;   // force Mode A on A leaders
  return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

void InitReplacementState() {
  for (uint32_t s = 0; s < LLC_SETS; s++) {
    bandit_score[s] = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      rrpv[s][w] = maxRRPV;
      hitcnt[s][w] = 0;
      stream_tag[s][w] = 0;
    }
  }
  std::memset(shct_demand, 0, sizeof(shct_demand));
  std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
  for (uint32_t i = 0; i < 64; i++) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      hawk_sig[i][w] = 0;
      hawk_used[i][w] = 0;
      hawk_is_pref[i][w] = 0;
    }
  }
  for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
    pc_last_line10[i] = 0xFFFFu;
    pc_stream_conf[i] = 0;
    pc_use4[i] = 0;
  }
  prefer_B = false;
  leaderA_good = leaderB_good = 0;
  access_count = 0;
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
  return rrip_pick_victim(set, current_set);
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

  // Periodic LFU decay and leader gate update
  access_count++;
  if ((access_count % LFU_DECAY_PERIOD) == 0) {
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
  }
  if ((access_count % BANDIT_EPOCH) == 0) {
    prefer_B = ((leaderB_good - leaderA_good) > GATE_MARGIN);
    leaderA_good = leaderB_good = 0;
  }

  // Update PC stream state and usage
  uint32_t pc_idx = pc_index(PC);
  uint8_t  sconf  = update_stream_conf(PC, paddr);
  if (is_demand(type)) sat_inc_u4(pc_use4[pc_idx]); // only demand bumps LFU

  // Update bandit per-set hint: streamy sets lean to Mode B
  if (sconf >= STREAM_ARM_THRESH) sat_inc_i8(bandit_score[set]);
  else                            sat_dec_i8(bandit_score[set]);

  // Leader rewards (conservative: count only demand/prefetch, ignore writebacks)
  if (!is_writeback(type)) {
    if (LEADER_A(set)) { if (hit) leaderA_good++; else leaderA_good--; }
    if (LEADER_B(set)) { if (hit) leaderB_good++; else leaderB_good--; }
  }

  // HIT path: promotion decisions and training reuse
  if (hit) {
    // Mark reuse for Hawkeye leaders
    if (SEL_SAMPLED(set)) {
      hawk_used[LEADER_SLOT(set)][way] = 1;
    }

    // Promotion policy
    if (is_demand(type)) {
      // Demand hit increments the per-line hit counter (saturates)
      if (hitcnt[set][way] < 3) hitcnt[set][way]++;

      if (stream_tag[set][way]) {
        // Stream quarantine: escape to MRU only on 2nd demand hit
        if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
          rrpv_set(set, way, 0);
          stream_tag[set][way] = 0; // escape stream quarantine
        } else {
          if (STREAM_DEMOTE_TOUCH && rrpv[set][way] < maxRRPV) rrpv[set][way]++;
        }
      } else {
        // Non-stream: PC-sensitive gate (2 hits if hot, else 3)
        uint8_t need = (pc_use4[pc_idx] >= PC_USE_HOT_THRESH) ? HITS_PROMOTE_NS_HOT : HITS_PROMOTE_NS_COLD;
        if (hitcnt[set][way] >= need) {
          rrpv_set(set, way, 0); // MRU
        } else {
          // mild nudge toward MRU
          if (rrpv[set][way] > 0) rrpv[set][way]--;
        }
      }
    } else {
      // Prefetch hit: do not count toward promotion gate; gently reduce RRPV if not stream-tagged
      if (!stream_tag[set][way] && rrpv[set][way] > 0) rrpv[set][way]--;
    }
    return;
  }

  // MISS/FILL path:
  // Train Hawkeye on eviction for leader sets before overwriting metadata
  if (SEL_SAMPLED(set)) {
    uint32_t slot = LEADER_SLOT(set);
    uint16_t sig  = hawk_sig[slot][way];
    uint8_t used  = hawk_used[slot][way];
    uint8_t was_pref = hawk_is_pref[slot][way];
    if (sig) {
      uint32_t idx = shct_idx(sig);
      if (used) {
        if (was_pref) shct_inc(shct_prefetch[idx]);
        else          shct_inc(shct_demand[idx]);
      } else {
        if (was_pref) shct_dec(shct_prefetch[idx]);
        else          shct_dec(shct_demand[idx]);
      }
    }
  }

  // Decide mode for this set
  bool use_modeB = modeB_enabled(set);

  // Decide insertion depth and stream quarantine
  uint8_t insert_rrpv = INSERT_COLD_DEPTH;
  uint8_t new_stream_tag = 0;

  if (is_prefetch(type)) {
    // Prefetch quarantine: deepest tail, stream-tagged
    insert_rrpv = STREAM_TAIL_DEPTH;
    new_stream_tag = 1;
  } else if (!is_writeback(type)) {
    // Demand/RFO path
    if (sconf >= STREAM_ARM_THRESH) {
      // Armed stream: quarantine at tail; sustain on 3rd+ step
      insert_rrpv = STREAM_TAIL_DEPTH;
      new_stream_tag = 1;
      // Note: writebacks never bypass; for demand we emulate bypass via hard-tail
    } else {
      if (use_modeB) {
        // Mode B: TinyLFU hot PCs get warm insertion; cold PCs near tail
        if (pc_use4[pc_idx] >= PC_USE_HOT_THRESH) insert_rrpv = INSERT_WARM_DEPTH;
        else                                      insert_rrpv = INSERT_COLD_DEPTH;
      } else {
        // Mode A (Hawkeye-lite): SHCT-driven insertion
        uint16_t sig = pc_sig12(PC);
        uint32_t idx = shct_idx(sig);
        uint8_t hot = is_prefetch(type) ? shct_prefetch[idx] : shct_demand[idx];
        // Conservative hot threshold to avoid pollution
        if (hot >= 12) insert_rrpv = INSERT_WARM_DEPTH;
        else           insert_rrpv = INSERT_COLD_DEPTH;
      }
    }
  } else {
    // Writeback: never bypass; insert moderately warm
    insert_rrpv = INSERT_WARM_DEPTH;
  }

  // Install new line metadata
  rrpv_set(set, way, insert_rrpv);
  stream_tag[set][way] = new_stream_tag;
  hitcnt[set][way] = 0;

  // Record for Hawkeye leaders
  if (SEL_SAMPLED(set)) {
    uint32_t slot = LEADER_SLOT(set);
    hawk_sig[slot][way]     = pc_sig12(PC);
    hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    hawk_used[slot][way]    = 0;
  }
}

// Print end-of-simulation statistics
void PrintStats() {}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {}