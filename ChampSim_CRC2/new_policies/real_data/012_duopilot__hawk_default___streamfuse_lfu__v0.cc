#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// Access helpers use ChampSim's enums
static inline bool is_demand(uint32_t t)    { return (t == LOAD) || (t == RFO); }
static inline bool is_prefetch(uint32_t t)  { return (t == PREFETCH); }
static inline bool is_writeback(uint32_t t) { return (t == WRITEBACK); }

// ---------------- Tunables ----------------
static constexpr uint8_t maxRRPV             = 7;  // 3-bit RRIP
static constexpr uint8_t INSERT_WARM_DEPTH   = 2;  // near-MRU
static constexpr uint8_t INSERT_COLD_DEPTH   = 6;  // near-tail
static constexpr uint8_t STREAM_TAIL_DEPTH   = 7;  // hard tail for streams/prefetch

static constexpr uint8_t STREAM_ARM_THRESH   = 2;  // arm after 2 forward (+1/+2) steps
static constexpr uint8_t HITS_PROMOTE_NS     = 2;  // non-stream: MRU on 2nd demand hit
static constexpr uint8_t HITS_PROMOTE_STR    = 2;  // stream/prefetch: MRU on 2nd demand hit
static constexpr uint8_t STREAM_DEMOTE_TOUCH = 1;  // demote quarantined streams on touch

static constexpr uint8_t  PC_USE_HOT_THRESH  = 8;     // TinyLFU hot threshold (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD   = 2048;  // decay every N accesses

static constexpr uint32_t BANDIT_EPOCH        = 4096; // selector epoch
static constexpr int8_t   MODEB_ENABLE_THRESH = 2;    // per-set confidence to enable B
static constexpr int32_t  GATE_MARGIN         = 3;    // global bias toward Hawkeye

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
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
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
static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective (stored in 16b)
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

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr) {
  uint32_t idx = pc_index(PC);
  uint16_t ln  = line10(paddr);
  uint16_t last = pc_last_line10[idx];
  bool forward = false;
  if (last != 0xFFFFu) {
    uint16_t exp1 = (uint16_t)(uint16_t)(last + 1);
    uint16_t exp2 = (uint16_t)(uint16_t)(last + 2);
    forward = (ln == exp1) || (ln == exp2);
  }
  if (forward) sat_inc_u2(pc_stream_conf[idx]);
  else pc_stream_conf[idx] = 0;
  pc_last_line10[idx] = ln;
  return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
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
  prefer_B = false;
  leaderA_good = leaderB_good = 0;
  access_count = 0;

  // SHCTs
  for (uint32_t i = 0; i < SHCT_SIZE; i++) {
    shct_demand[i] = 0;
    shct_prefetch[i] = 0;
  }
  // Leader buffers
  for (uint32_t i = 0; i < 64; i++) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      hawk_sig[i][w] = 0;
      hawk_used[i][w] = 0;
      hawk_is_pref[i][w] = 0;
    }
  }
  // PC tables
  for (uint32_t i = 0; i < PC_TBL_SIZE; i++) {
    pc_last_line10[i] = 0xFFFFu;
    pc_stream_conf[i] = 0;
    pc_use4[i] = 0;
  }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
  // Prefer invalid way
  for (uint32_t w = 0; w < LLC_WAYS; w++) {
    if (!current_set[w].valid) return w;
  }

  while (true) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
      if (rrpv[set][w] == maxRRPV) {
        // Train Hawkeye-lite (leaders only) on eviction of this way
        if (SEL_SAMPLED(set)) {
          uint32_t slot = LEADER_SLOT(set);
          uint16_t sig = hawk_sig[slot][w];
          uint8_t used = hawk_used[slot][w];
          uint8_t was_pref = hawk_is_pref[slot][w];
          if (sig) {
            uint32_t idx = shct_idx(sig);
            if (was_pref) {
              if (used) shct_inc(shct_prefetch[idx]);
              else      shct_dec(shct_prefetch[idx]);
            } else {
              if (used) shct_inc(shct_demand[idx]);
              else      shct_dec(shct_demand[idx]);
            }
          }
          hawk_used[slot][w] = 0;
          hawk_is_pref[slot][w] = 0;
          hawk_sig[slot][w] = 0;
        }
        return w;
      }
    }
    // Age RRPVs and retry
    rrpv_age_all(set);
  }
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
  if (is_writeback(type)) return;

  access_count++;

  // Epoch actions: update global gate and decay
  if ((access_count % BANDIT_EPOCH) == 0) {
    int32_t diff = leaderB_good - leaderA_good;
    if (diff > GATE_MARGIN)      prefer_B = true;
    else if (diff < -GATE_MARGIN) prefer_B = false;
    leaderA_good = 0;
    leaderB_good = 0;
    // mild bandit decay
    for (uint32_t s = 0; s < LLC_SETS; s++) {
      if (bandit_score[s] > 0) bandit_score[s]--;
      else if (bandit_score[s] < 0) bandit_score[s]++;
    }
    // TinyLFU decay
    for (uint32_t i = 0; i < PC_TBL_SIZE; i++) pc_use4[i] >>= 1;
  }

  // Update per-PC stream state and TinyLFU on every access (non-WB)
  bool stream_now = detect_and_update_stream(PC, paddr);
  sat_inc_u4(pc_use4[pc_index(PC)]);

  if (hit) {
    // Leader reward
    if (LEADER_A(set)) leaderA_good++;
    if (LEADER_B(set)) leaderB_good++;

    // Mark reuse in leaders for SHCT training
    if (SEL_SAMPLED(set)) {
      uint32_t slot = LEADER_SLOT(set);
      hawk_used[slot][way] = 1;
    }

    // Quarantined stream/prefetch: demote on touch to resist scans
    if (stream_tag[set][way] && STREAM_DEMOTE_TOUCH) {
      if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
    }

    // Multi-hit promotion (demand only)
    if (is_demand(type)) {
      if (hitcnt[set][way] < 3) hitcnt[set][way]++;
      uint8_t need = stream_tag[set][way] ? HITS_PROMOTE_STR : HITS_PROMOTE_NS;
      if (hitcnt[set][way] >= need) {
        rrpv_set(set, way, 0);          // MRU on 2nd+ demand hit
        stream_tag[set][way] = 0;       // escape quarantine
        sat_dec_i8(bandit_score[set]);  // reuse observed -> bias toward Mode A
      }
    }
    return;
  }

  // Miss/fill path: leader penalty
  if (LEADER_A(set)) leaderA_good--;
  if (LEADER_B(set)) leaderB_good--;

  bool use_modeB = modeB_enabled(set);
  uint8_t ins_rrpv = INSERT_COLD_DEPTH;
  uint8_t tag_stream = 0;

  if (is_prefetch(type)) {
    ins_rrpv = STREAM_TAIL_DEPTH;
    tag_stream = 1;
  } else if (use_modeB && stream_now) {
    // Stream detected (two forward steps) -> hard tail quarantine
    ins_rrpv = STREAM_TAIL_DEPTH;
    tag_stream = 1;
  } else {
    if (use_modeB) {
      // TinyLFU gate
      uint8_t use = pc_use4[pc_index(PC)];
      ins_rrpv = (use >= PC_USE_HOT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    } else {
      // Mode A: Hawkeye-lite SHCT gate (demand path)
      uint16_t sig = pc_sig12(PC);
      uint8_t val = shct_demand[shct_idx(sig)];
      ins_rrpv = (val >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
    }
  }

  rrpv_set(set, way, ins_rrpv);
  hitcnt[set][way] = 0;
  stream_tag[set][way] = tag_stream;

  // Selector local feedback
  if (tag_stream) sat_inc_i8(bandit_score[set]); else sat_dec_i8(bandit_score[set]);

  // Leader bookkeeping for SHCT training
  if (SEL_SAMPLED(set)) {
    uint32_t slot = LEADER_SLOT(set);
    hawk_sig[slot][way] = pc_sig12(PC);
    hawk_is_pref[slot][way] = is_prefetch(type) ? 1 : 0;
    hawk_used[slot][way] = 0; // not yet reused
  }
}

void PrintStats_Heartbeat() {}
void PrintStats() {}