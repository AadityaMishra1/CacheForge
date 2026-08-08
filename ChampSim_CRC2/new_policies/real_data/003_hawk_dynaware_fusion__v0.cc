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
// Core RRIP
static constexpr uint8_t maxRRPV             = 7;  // 3-bit SRRIP
// Insertions
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch
static constexpr uint8_t STREAM_SHALLOW_TAIL = 5;  // shallower tail for TinyLFU-hot streams
// Stream Detector
static constexpr uint8_t STREAM_CONF_THRESH  = 2;  // need 2 consecutive +1/+2 steps
// Promotions (multi-hit gating)
static constexpr uint8_t HITS_PROMOTE_NS      = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR     = 2; // stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR_HOT = 1; // TinyLFU-hot stream: MRU on 1st demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH  = 1; // demote quarantined streams on touch
// TinyLFU
static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD   = 1024;  // decay every N accesses
// Selector epoch
static constexpr uint32_t BANDIT_EPOCH       = 8192;  // compare leaders every epoch
static constexpr int8_t   MODEB_ENABLE_THRESH= 2;     // followers enable B if per-set >= 2

// ---------------- Per-line metadata (conceptually bit-packed) ----
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // 1 if quarantined stream/prefetch

// ---------------- Per-set selector --------------------------------
static int8_t bandit_score[LLC_SETS]; // [-8..7], followers bias toward Mode B
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

// Leader-A per-line training buffers (only for 64 sampled sets)
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

// +1/+2 forward-run detector with 2-step confidence
static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr, bool do_update) {
  uint32_t idx = pc_index(PC);
  uint16_t ln  = line10(paddr);
  uint16_t last = pc_last_line10[idx];
  bool forward = (ln == (uint16_t)(last + 1)) || (ln == (uint16_t)(last + 2));
  if (do_update) {
    if (forward) sat_inc_u2(pc_stream_conf[idx]);
    else pc_stream_conf[idx] = 0;
    pc_last_line10[idx] = ln;
  }
  return (pc_stream_conf[idx] >= STREAM_CONF_THRESH);
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
  for (uint32_t s = 0; s < LLC_SETS; s++) {
    bandit_score[s] = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      rrpv[s][w]       = maxRRPV;
      hitcnt[s][w]     = 0;
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

// ---------------- Victim selection (RRIP) ------------------------
static inline uint32_t rrip_find_victim(uint32_t set, const BLOCK* current_set) {
  // 1) Prefer invalid
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (!current_set[w].valid) return w;
  }
  // 2) Prefer any line at maxRRPV with stream_tag=1 (quarantine first)
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (rrpv[set][w] == maxRRPV && stream_tag[set][w]) return w;
  }
  // 3) Any line at maxRRPV
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (rrpv[set][w] == maxRRPV) return w;
  }
  // 4) Age bounded passes, ensure termination
  for (int pass = 0; pass < 8; pass++) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] == maxRRPV) return w;
    }
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
    }
  }
  // Fallback (should not happen): pick the way with largest RRPV
  uint32_t best = 0, best_rrpv = 0;
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (rrpv[set][w] >= best_rrpv) { best_rrpv = rrpv[set][w]; best = w; }
  }
  return best;
}

// Find replacement victim (must return [0..LLC_WAYS-1])
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
  uint32_t victim = rrip_find_victim(set, current_set);

  // Train Hawkeye-like predictor on A-leader eviction
  if (LEADER_A(set)) {
    uint32_t slot = LEADER_SLOT(set);
    uint16_t sig  = hawk_sig[slot][victim];
    uint8_t  used = hawk_used[slot][victim];
    uint8_t  pref = hawk_is_pref[slot][victim];

    if (sig) {
      uint32_t idx = shct_idx(sig);
      if (pref) {
        if (used) shct_inc(shct_prefetch[idx]);
        else      shct_dec(shct_prefetch[idx]);
      } else {
        if (used) shct_inc(shct_demand[idx]);
        else      shct_dec(shct_demand[idx]);
      }
    }
    // Clear training slot
    hawk_sig[slot][victim] = 0;
    hawk_used[slot][victim] = 0;
    hawk_is_pref[slot][victim] = 0;
  }

  return victim;
}

// ---------------- Update replacement state -----------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
  (void)cpu; (void)victim_addr;

  // Epoch bookkeeping and LFU decay (cheap: 512 entries)
  access_count++;
  if ((access_count % LFU_DECAY_PERIOD) == 0) {
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
  }
  // Leader reward: hits - misses (ignore writebacks)
  if (!is_writeback(type)) {
    if (LEADER_A(set)) leaderA_good += (hit ? 1 : -1);
    else if (LEADER_B(set)) leaderB_good += (hit ? 1 : -1);
  }
  // Selector epoch: prefer Mode B only if it beats Mode A on leaders
  if ((access_count % BANDIT_EPOCH) == 0) {
    prefer_B = (leaderB_good > leaderA_good);
    for (uint32_t s = 0; s < LLC_SETS; s++) {
      if (prefer_B) sat_inc_i8(bandit_score[s]);
      else          sat_dec_i8(bandit_score[s]);
    }
    leaderA_good = leaderB_good = 0;
  }

  // Update TinyLFU on demand references
  if (is_demand(type)) {
    uint32_t pidx = pc_index(PC);
    sat_inc_u4(pc_use4[pidx]);
  }

  bool use_modeB = modeB_enabled(set);

  if (hit) {
    // ----------------------- On hit -----------------------
    if (is_writeback(type)) return; // ignore WB hits

    // Stream detector updates only on demand touches
    if (is_demand(type)) {
      (void)detect_and_update_stream(PC, paddr, true);
    }

    // Multi-hit gating
    uint8_t &hc = hitcnt[set][way];
    if (is_demand(type)) { if (hc < 3) hc++; }

    uint32_t pidx = pc_index(PC);
    bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    if (stream_tag[set][way]) {
      // Quarantined stream/prefetch line behavior
      uint8_t th = hot_pc ? HITS_PROMOTE_STR_HOT : HITS_PROMOTE_STR;
      if (is_demand(type) && hc >= th) {
        rrpv_set(set, way, 0);    // escape quarantine
        stream_tag[set][way] = 0; // no longer stream
        hc = 0;
      } else if (STREAM_DEMOTE_TOUCH) {
        if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
      }
    } else {
      // Non-stream: promote only on 2nd+ demand hit
      if (is_demand(type) && hc >= HITS_PROMOTE_NS) {
        rrpv_set(set, way, 0);
        hc = 0;
      }
    }

    // Leader-A: mark reuse for SHCT training
    if (LEADER_A(set)) {
      uint32_t slot = LEADER_SLOT(set);
      hawk_used[slot][way] = 1;
    }
    return;
  }

  // ----------------------- On fill (miss) -----------------------
  // Never bypass on writeback; but keep WB insertion conservative
  uint8_t ins_depth = INSERT_COLD_DEPTH;
  uint8_t st_tag    = 0;
  hitcnt[set][way]  = 0;

  uint32_t pidx = pc_index(PC);
  bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

  if (use_modeB) {
    // DynAWARE path
    bool conf_stream = is_demand(type) ? detect_and_update_stream(PC, paddr, true)
                                       : (pc_stream_conf[pidx] >= STREAM_CONF_THRESH);

    if (is_prefetch(type)) {
      // Prefetch quarantine
      ins_depth = STREAM_TAIL_DEPTH;
      st_tag    = 1;
    } else if (conf_stream && !is_writeback(type)) {
      // Demand stream: tail or shallow-tail for hot PCs
      ins_depth = hot_pc ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
      st_tag    = 1;
    } else {
      // Non-stream or writeback: LFU-guided insertion
      if (is_writeback(type)) {
        ins_depth = INSERT_COLD_DEPTH; // be conservative
      } else {
        ins_depth = hot_pc ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
      }
      st_tag = 0;
    }
  } else {
    // Hawkeye-like path (tiny SHCT)
    uint16_t sig = pc_sig12(PC);
    uint32_t idx = shct_idx(sig);

    if (is_prefetch(type)) {
      ins_depth = STREAM_TAIL_DEPTH; // prefetch cold
    } else if (is_writeback(type)) {
      ins_depth = INSERT_COLD_DEPTH;
    } else {
      uint8_t val = shct_demand[idx];
      ins_depth = (val >= 12) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    }
    st_tag = 0;

    // Record leader-A training info on fill
    if (LEADER_A(set)) {
      uint32_t slot = LEADER_SLOT(set);
      hawk_sig[slot][way]     = sig;
      hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
      hawk_used[slot][way]    = 0;
    }
  }

  rrpv_set(set, way, ins_depth);
  stream_tag[set][way] = st_tag;
}

// ---------------- Stats stubs ------------------------------------
void PrintStats_Heartbeat() {}
void PrintStats() {}