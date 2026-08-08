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

// ---------------- Tunables ----------------
static constexpr uint8_t  maxRRPV               = 7;  // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH     = 2;  // near-MRU
static constexpr uint8_t  INSERT_COLD_DEPTH     = 6;  // near-tail
static constexpr uint8_t  STREAM_TAIL_DEPTH     = 7;  // hard tail quarantine

// Stream run detector: +1/+2 forward steps
static constexpr uint8_t  STREAM_ARM_THRESH     = 2;  // arm after 2 forward steps
static constexpr uint8_t  STREAM_HARD_THRESH    = 3;  // 3+ -> sustain pseudo-bypass

// Multi-hit promotions
static constexpr uint8_t  HITS_PROMOTE_STR      = 2;  // stream/prefetch: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_NS_HOT   = 2;  // non-stream hot PC
static constexpr uint8_t  HITS_PROMOTE_NS_COLD  = 3;  // non-stream cold PC
static constexpr bool     STREAM_DEMOTE_TOUCH   = true; // demote quarantined streams on touch

// TinyLFU gate (PC coldness)
static constexpr uint8_t  PC_USE_HOT_THRESH     = 12;    // 0..15
static constexpr uint32_t LFU_DECAY_PERIOD      = 4096;  // slower decay to reduce noise

// Selector/leader gates
static constexpr uint32_t BANDIT_EPOCH          = 4096;
static constexpr int8_t   MODEB_ENABLE_THRESH   = 2;     // per-set bandit threshold
static constexpr int32_t  GATE_MARGIN           = 6;     // global margin to prefer Hawkeye

// Stream budget per set
static constexpr uint8_t  STREAM_BUDGET_LIMIT   = 4;     // max quarantined lines per set

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

// ---------------- Per-set selector/budget -------------------------
static int8_t  bandit_score[LLC_SETS]; // [-8..7], defaults to 0 (favor A)
static uint8_t stream_budget_cnt[LLC_SETS]; // 0..16, saturated at limit

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
static constexpr uint8_t SHCT_HOT_THRESH = 16; // >=16 considered friendly

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

// Stream detector (+1/+2 forward) update; returns new confidence (0..3)
static inline uint8_t update_stream_conf(uint64_t PC, uint64_t paddr) {
  uint32_t idx = pc_index(PC);
  uint16_t ln  = line10(paddr);
  uint16_t last = pc_last_line10[idx];

  bool forward = false;
  if (last != 0xFFFFu) {
    uint16_t exp1 = (uint16_t)((last + 1) & 0x03FFu);
    uint16_t exp2 = (uint16_t)((last + 2) & 0x03FFu);
    forward = (ln == exp1) || (ln == exp2);
  }
  if (forward) {
    sat_inc_u2(pc_stream_conf[idx]); // up to 3
  } else {
    sat_dec_u2(pc_stream_conf[idx]); // decay on non-forward
  }
  pc_last_line10[idx] = ln;
  return pc_stream_conf[idx];
}

// ---------------- Initialization ---------------------------------
void InitReplacementState() {
  for (uint32_t s = 0; s < LLC_SETS; s++) {
    bandit_score[s] = 0;
    stream_budget_cnt[s] = 0;
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      rrpv[s][w] = maxRRPV;
      hitcnt[s][w] = 0;
      stream_tag[s][w] = 0;
    }
  }

  std::memset(shct_demand,   0, sizeof(shct_demand));
  std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
  for (uint32_t ls = 0; ls < 64; ls++) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      hawk_sig[ls][w] = 0;
      hawk_used[ls][w] = 0;
      hawk_is_pref[ls][w] = 0;
    }
  }

  for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
    pc_last_line10[i] = 0xFFFFu;
    pc_stream_conf[i] = 0;
    pc_use4[i] = 0;
  }

  prefer_B = false;
  leaderA_good = 0;
  leaderB_good = 0;
  access_count = 0;
}

// ---------------- Victim selection --------------------------------
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
  // RRIP victim search
  while (true) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] == maxRRPV) return w;
    }
    rrpv_age_all(set);
  }
}

// ---------------- State update ------------------------------------
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

  access_count++;

  // periodic TinyLFU decay
  if ((access_count % LFU_DECAY_PERIOD) == 0) {
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
      pc_use4[i] >>= 1;
    }
  }

  // Update PC TinyLFU on demand access
  if (is_demand(type)) {
    uint32_t pidx = pc_index(PC);
    sat_inc_u4(pc_use4[pidx]);
  }

  // Update stream conf and derive stream status
  uint8_t sc = update_stream_conf(PC, paddr);
  bool stream_armed = (sc >= STREAM_ARM_THRESH);
  bool stream_hard  = (sc >= STREAM_HARD_THRESH);

  // Update leader-goodness and bandit score
  if (LEADER_A(set)) {
    if (hit) leaderA_good++; else leaderA_good--;
  } else if (LEADER_B(set)) {
    if (hit) leaderB_good++; else leaderB_good--;
  } else {
    // Followers: gently bias bandit towards streaming sets
    if (stream_armed) sat_inc_i8(bandit_score[set]);
    else              sat_dec_i8(bandit_score[set]);
  }

  // Epochic global gate update
  if ((access_count % BANDIT_EPOCH) == 0) {
    int32_t diff = leaderB_good - leaderA_good;
    prefer_B = (diff > GATE_MARGIN);
    leaderA_good = 0;
    leaderB_good = 0;
  }

  bool use_modeB;
  if (LEADER_A(set)) use_modeB = false;
  else if (LEADER_B(set)) use_modeB = true;
  else use_modeB = (prefer_B && (bandit_score[set] >= MODEB_ENABLE_THRESH));

  // On HIT: apply promotion policy
  if (hit) {
    if (stream_tag[set][way]) {
      // Stream/prefetch quarantined line
      if (is_demand(type)) {
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        if (hitcnt[set][way] >= HITS_PROMOTE_STR) {
          // Escape quarantine on 2nd demand hit
          rrpv_set(set, way, 0);
          stream_tag[set][way] = 0;
          if (stream_budget_cnt[set] > 0) stream_budget_cnt[set]--;
        } else if (STREAM_DEMOTE_TOUCH) {
          rrpv_set(set, way, STREAM_TAIL_DEPTH);
        }
      } else {
        // Non-demand touch: keep quarantined at tail
        if (STREAM_DEMOTE_TOUCH) rrpv_set(set, way, STREAM_TAIL_DEPTH);
      }
    } else {
      // Non-stream: PC-gated multi-hit
      uint32_t pidx = pc_index(PC);
      bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
      uint8_t need = pc_hot ? HITS_PROMOTE_NS_HOT : HITS_PROMOTE_NS_COLD;
      if (is_demand(type) && hitcnt[set][way] < 3) hitcnt[set][way]++;
      if (hitcnt[set][way] >= need) {
        rrpv_set(set, way, 0); // MRU
      } else {
        // gentle promotion
        if (rrpv[set][way] > 0) rrpv[set][way]--;
      }
    }

    // Hawkeye reuse mark for leader sets
    if (SEL_SAMPLED(set)) {
      uint32_t ls = LEADER_SLOT(set);
      hawk_used[ls][way] = 1;
    }
    return;
  }

  // MISS: insertion policy and leader training
  // Leader training (train on the line being evicted)
  if (SEL_SAMPLED(set)) {
    uint32_t ls = LEADER_SLOT(set);
    uint16_t psig = hawk_sig[ls][way];
    uint8_t  used = hawk_used[ls][way];
    uint8_t  was_pref = hawk_is_pref[ls][way];
    if (was_pref) {
      if (used) shct_inc(shct_prefetch[shct_idx(psig)]);
      else      shct_dec(shct_prefetch[shct_idx(psig)]);
    } else {
      if (used) shct_inc(shct_demand[shct_idx(psig)]);
      else      shct_dec(shct_demand[shct_idx(psig)]);
    }
  }

  // Before overwrite: adjust stream budget if victim was quarantined
  if (stream_tag[set][way]) {
    stream_tag[set][way] = 0;
    if (stream_budget_cnt[set] > 0) stream_budget_cnt[set]--;
  }

  // Prepare defaults
  uint8_t new_rrpv = INSERT_COLD_DEPTH;
  uint8_t new_stream_tag = 0;
  uint8_t new_hitcnt = 0;

  if (use_modeB) {
    // Mode B: stream-aware with budget and TinyLFU fallback
    if (is_writeback(type)) {
      // Never bypass/don't quarantine writebacks
      new_rrpv = INSERT_COLD_DEPTH;
      new_stream_tag = 0;
    } else if (is_prefetch(type)) {
      // Always quarantine prefetches
      new_rrpv = STREAM_TAIL_DEPTH;
      new_stream_tag = 1;
    } else {
      // Demand
      bool can_quarantine = stream_armed && (stream_budget_cnt[set] < STREAM_BUDGET_LIMIT);
      if (can_quarantine) {
        new_rrpv = STREAM_TAIL_DEPTH; // hard tail (pseudo-bypass)
        new_stream_tag = 1;
      } else {
        // Non-stream or budget full: TinyLFU-gated insertion
        uint32_t pidx = pc_index(PC);
        bool pc_hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
        new_rrpv = pc_hot ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
        new_stream_tag = 0;
      }
    }
  } else {
    // Mode A: Hawkeye-lite adaptive insertion
    uint16_t sig = pc_sig12(PC);
    uint8_t pred = 0;
    if (is_prefetch(type)) {
      pred = shct_prefetch[shct_idx(sig)];
      new_rrpv = STREAM_TAIL_DEPTH; // prefetch always at tail
      new_stream_tag = 1;           // quarantine prefetch
    } else if (is_writeback(type)) {
      new_rrpv = INSERT_COLD_DEPTH;
      new_stream_tag = 0;
    } else {
      pred = shct_demand[shct_idx(sig)];
      new_rrpv = (pred >= SHCT_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
      new_stream_tag = 0;
    }

    // Record per-line info for leader sets (for next eviction training)
    if (SEL_SAMPLED(set)) {
      uint32_t ls = LEADER_SLOT(set);
      hawk_sig[ls][way] = sig;
      hawk_used[ls][way] = 0;
      hawk_is_pref[ls][way] = is_prefetch(type) ? 1 : 0;
    }
  }

  // Commit insertion metadata
  rrpv_set(set, way, new_rrpv);
  hitcnt[set][way] = new_hitcnt;

  if (new_stream_tag) {
    stream_tag[set][way] = 1;
    if (stream_budget_cnt[set] < STREAM_BUDGET_LIMIT) stream_budget_cnt[set]++;
  } else {
    stream_tag[set][way] = 0;
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