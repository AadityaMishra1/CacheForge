#include <cstdint>
#include <cstring>
#include <algorithm>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t)    { return (t == ACCESS_LOAD) || (t == ACCESS_RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == ACCESS_PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == ACCESS_WRITEBACK); }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
  // Sample 64 sets: low 6 bits == next 6 bits
  return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }
static inline uint32_t LEADER_SLOT(uint32_t set) { return (set & 63u); } // 0..63

// ---------------- Tunables ---------------------------------------
// RRIP depths (3-bit)
static constexpr uint8_t maxRRPV             = 7;
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallow tail for hot-PC streams

// Stream detector (+1/+2 forward run)
static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // after 2 fwd steps: arm
static constexpr uint8_t STREAM_LOCK_THRESH  = 3;  // after 3 fwd steps: lock

// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS      = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR     = 2; // stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT = 1; // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH  = 1; // demote quarantined streams on touch

// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

// Selector epoch and gates
static constexpr uint32_t BANDIT_EPOCH        = 4096; // short epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Hawkeye

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt(demand hits):2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode A unless confident
static bool prefer_B = false;          // global gate from leaders
static int32_t leaderA_good = 0;       // hits - misses on leader A sets
static int32_t leaderB_good = 0;       // hits - misses on leader B sets
static uint64_t access_count = 0;

// ---------------- Mode A (Hawkeye-like via tiny SHCT) -------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch
#define SHCT_SIZE (1u << 10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc) {
  // 12b hash
  uint64_t x = pc ^ (pc >> 3) ^ (pc >> 11) ^ (pc >> 19);
  return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig) { return (uint32_t)sig & (SHCT_SIZE - 1u); }
static inline void shct_inc(uint8_t &x) { if (x < SHCT_MAX) x++; }
static inline void shct_dec(uint8_t &x) { if (x > 0) x--; }

// Leader-A/B per-line training buffers (only for 64 sampled sets)
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)

// ---------------- Mode B (DynAWARE PC state) ----------------------
// 512-entry PC tables: last line (10b), stream conf (2b), TinyLFU (4b)
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc) { return (uint32_t)pc & (PC_TBL_SIZE - 1u); }
static inline uint16_t line10(uint64_t paddr) { return (uint16_t)((paddr >> 6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 10b in 16b storage
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

static inline bool modeB_enabled(uint32_t set) {
  if (LEADER_B(set)) return true;   // force Mode B on B leaders
  if (LEADER_A(set)) return false;  // force Mode A on A leaders
  // Followers: enable B only if global gate prefers it and local bandit is confident
  return prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH);
}

// Train SHCT on eviction for leader sets
static inline void train_hawkeye_on_eviction(uint32_t set, uint32_t way, bool valid) {
  if (!valid) return;
  if (!SEL_SAMPLED(set)) return;
  uint32_t slot = LEADER_SLOT(set);
  uint16_t sig = hawk_sig[slot][way];
  uint32_t idx = shct_idx(sig);
  if (hawk_is_pref[slot][way]) {
    if (hawk_used[slot][way]) shct_inc(shct_prefetch[idx]);
    else                      shct_dec(shct_prefetch[idx]);
  } else {
    if (hawk_used[slot][way]) shct_inc(shct_demand[idx]);
    else                      shct_dec(shct_demand[idx]);
  }
  // reset reuse marker after training
  hawk_used[slot][way] = 0;
}

// Initialize replacement state
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
  std::memset(hawk_sig, 0, sizeof(hawk_sig));
  std::memset(hawk_used, 0, sizeof(hawk_used));
  std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
  std::memset(pc_last_line10, 0, sizeof(pc_last_line10));
  std::memset(pc_stream_conf, 0, sizeof(pc_stream_conf));
  std::memset(pc_use4, 0, sizeof(pc_use4));
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
  // If any invalid way exists, return it immediately
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (!current_set[w].valid) return w;
  }

  // SRRIP: search for RRPV==max; age if none
  while (true) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] == maxRRPV) {
        // Train Hawkeye on eviction for leader sets before overwriting
        train_hawkeye_on_eviction(set, w, current_set[w].valid);
        return w;
      }
    }
    // Increment all RRPVs (saturating) and retry
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      sat_inc_u3(rrpv[set][w]);
    }
  }
  // Should never reach
  // return 0;
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
  // Global accounting and LFU decay
  access_count++;
  if ((access_count % LFU_DECAY_PERIOD) == 0) {
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) sat_dec_u4(pc_use4[i]);
  }
  if ((access_count % BANDIT_EPOCH) == 0) {
    prefer_B = (leaderB_good >= (leaderA_good + GATE_MARGIN));
    leaderA_good = 0;
    leaderB_good = 0;
  }

  const bool demand = is_demand(type);
  const bool pref   = is_prefetch(type);
  const bool wb     = is_writeback(type);

  // TinyLFU update for demand uses
  uint32_t pidx = pc_index(PC);
  if (demand) sat_inc_u4(pc_use4[pidx]);
  bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

  // Stream detector (+1/+2) using pre-update state for decisions
  uint16_t ln   = line10(paddr);
  uint16_t last = pc_last_line10[pidx];
  bool forward  = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
  uint8_t conf_before = pc_stream_conf[pidx];
  bool armed  = (conf_before >= STREAM_ARM_THRESH);
  bool locked = (conf_before >= STREAM_LOCK_THRESH);
  // Update stream state
  if (forward) { if (pc_stream_conf[pidx] < 3) pc_stream_conf[pidx]++; }
  else         { if (pc_stream_conf[pidx] > 0) pc_stream_conf[pidx]--; }
  pc_last_line10[pidx] = ln;

  // Mode selection
  bool useB = modeB_enabled(set);

  // Leader reward update (demand only)
  if (SEL_SAMPLED(set) && demand) {
    if (LEADER_A(set)) { if (hit) leaderA_good++; else leaderA_good--; }
    else               { if (hit) leaderB_good++; else leaderB_good--; }
  }

  // Per-set bandit reward (followers, Mode B, demand only)
  if (!SEL_SAMPLED(set) && useB && demand) {
    if (hit) sat_inc_i8(bandit_score[set]);
    else     sat_dec_i8(bandit_score[set]);
  }

  // Reuse marking for Hawkeye leaders on hit
  if (hit && SEL_SAMPLED(set)) {
    hawk_used[LEADER_SLOT(set)][way] = 1;
  }

  if (hit) {
    // Demand hit: multi-hit gated promotions
    if (demand) {
      uint8_t hc = hitcnt[set][way];
      if (hc < 3) hc++;
      hitcnt[set][way] = hc;

      if (stream_tag[set][way]) {
        uint8_t need = pc_hot ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
        if (hc >= need) {
          // escape quarantine
          stream_tag[set][way] = 0;
          rrpv_set(set, way, 0);
        } else {
          // demote on touch to keep scans cold
          uint8_t v = rrpv[set][way];
          v = std::min<uint8_t>(maxRRPV, (uint8_t)(v + STREAM_DEMOTE_TOUCH));
          rrpv_set(set, way, v);
        }
      } else {
        if (hc >= HITS_PROMOTE_NS) {
          rrpv_set(set, way, 0);
        } else {
          uint8_t v = rrpv[set][way];
          if (v > 1) v = 1;
          rrpv_set(set, way, v);
        }
      }
    } else if (pref) {
      // Prefetch hit: do not rush to MRU; slight nudge if not quarantined
      if (!stream_tag[set][way]) {
        uint8_t v = rrpv[set][way];
        if (v > 2) v = 2;
        rrpv_set(set, way, v);
      }
    }
    return;
  }

  // Miss fill path
  // Record Hawkeye fill metadata for leaders
  if (SEL_SAMPLED(set)) {
    uint32_t slot = LEADER_SLOT(set);
    hawk_sig[slot][way] = pc_sig12(PC);
    hawk_used[slot][way] = 0;
    hawk_is_pref[slot][way] = pref ? 1 : 0;
  }

  // Decide insertion depth and flags
  uint8_t ins = INSERT_COLD_DEPTH;
  stream_tag[set][way] = 0;
  hitcnt[set][way] = 0;

  if (wb) {
    ins = INSERT_COLD_DEPTH; // never bypass writebacks
  } else if (useB) {
    if (pref) {
      ins = STREAM_TAIL_DEPTH;
      stream_tag[set][way] = 1; // quarantine all prefetch fills
    } else { // demand
      if (locked) {
        ins = STREAM_TAIL_DEPTH;
        stream_tag[set][way] = 1;
      } else if (armed) {
        ins = pc_hot ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
        stream_tag[set][way] = 1;
      } else {
        // PC coldness gate: combine SHCT + TinyLFU
        uint8_t s = shct_demand[shct_idx(pc_sig12(PC))];
        if (!pc_hot && s <= 4) ins = INSERT_COLD_DEPTH;
        else                   ins = INSERT_WARM_DEPTH;
      }
    }
  } else {
    // Mode A: Hawkeye-lite insertion
    if (pref) {
      ins = STREAM_TAIL_DEPTH;
      stream_tag[set][way] = 1;
    } else {
      uint8_t s = shct_demand[shct_idx(pc_sig12(PC))];
      ins = (s >= 12) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    }
  }

  rrpv_set(set, way, ins);
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}