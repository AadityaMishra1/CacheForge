#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16

// ChampSim CRC2 access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

static inline bool is_demand(uint32_t t){ return t==ACCESS_LOAD || t==ACCESS_RFO; }
static inline bool is_prefetch(uint32_t t){ return t==ACCESS_PREFETCH; }
static inline bool is_writeback(uint32_t t){ return t==ACCESS_WRITEBACK; }

// ---------------- Leader-set sampling (64 total) ------------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
  // Sample 64 sets: low 6 bits == next 6 bits
  return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==0u); }
static inline bool LEADER_B(uint32_t set){ return SEL_SAMPLED(set) && ((set & 1u)==1u); }
static inline uint32_t LEADER_SLOT(uint32_t set){ return (set & 63u); } // 0..63

// ---------------- Tunables (retuned for balance) ------------------
static constexpr uint8_t  maxRRPV              = 7; // 3-bit RRIP
static constexpr uint8_t  INSERT_WARM_DEPTH    = 2;
static constexpr uint8_t  INSERT_COLD_DEPTH    = 6;
static constexpr uint8_t  STREAM_TAIL_DEPTH    = 7; // hard tail
static constexpr uint8_t  STREAM_SHALLOW_TAIL  = 5; // tail for hot streams
static constexpr uint8_t  STREAM_ARM_THRESH    = 2; // +1/+2 steps
static constexpr uint8_t  HITS_PROMOTE_NS      = 2; // non-stream: MRU on 2nd demand hit
static constexpr uint8_t  HITS_PROMOTE_STR     = 2; // stream: MRU on 2nd demand hit
static constexpr uint8_t  STREAM_DEMOTE_TOUCH  = 1; // demote quarantined on touch
static constexpr uint8_t  PC_USE_HOT_THRESH    = 8; // TinyLFU hot (0..15)
static constexpr uint32_t LFU_DECAY_PERIOD     = 2048; // global decay period
static constexpr uint32_t BANDIT_EPOCH         = 4096; // selector epoch
static constexpr int32_t  GATE_MARGIN          = 3;    // global bias toward Hawkeye
static constexpr int8_t   MODEB_ENABLE_THRESH  = 2;    // per-slot confidence
static constexpr uint8_t  RIS_SHORT_THRESH     = 3;    // short reuse if >=3

// ---------------- Per-line metadata (conceptually packed) ---------
// rrpv:3b, hitcnt:2b, stream_tag:1b
static uint8_t rrpv[LLC_SETS][LLC_WAYS];
static uint8_t hitcnt[LLC_SETS][LLC_WAYS];
static uint8_t stream_tag[LLC_SETS][LLC_WAYS]; // quarantined stream/prefetch

static inline void rrpv_set(uint32_t set, uint32_t way, uint8_t val){
  rrpv[set][way] = (val > maxRRPV) ? maxRRPV : val;
}
static inline void rrpv_age_all(uint32_t set){
  for(uint32_t w=0; w<LLC_WAYS; w++) if (rrpv[set][w] < maxRRPV) rrpv[set][w]++;
}

// ---------------- Mode A (Hawkeye-lite) ---------------------------
// 12b PC signature -> 1K-entry SHCT (5b) for demand and prefetch (leaders only)
#define SHCT_SIZE (1u<<10)
#define SHCT_MAX 31
static uint8_t shct_demand[SHCT_SIZE];
static uint8_t shct_prefetch[SHCT_SIZE];

static inline uint16_t pc_sig12(uint64_t pc){
  uint64_t x = pc ^ (pc>>3) ^ (pc>>11) ^ (pc>>19);
  return (uint16_t)(x & 0x0FFFu);
}
static inline uint32_t shct_idx(uint16_t sig){ return (uint32_t)sig & (SHCT_SIZE-1u); }

static uint16_t hawk_sig[64][LLC_WAYS];     // 12b effective
static uint8_t  hawk_used[64][LLC_WAYS];    // reuse seen (0/1)
static uint8_t  hawk_is_pref[64][LLC_WAYS]; // filled by prefetch (0/1)
static uint8_t  hawk_valid[64][LLC_WAYS];   // slot valid (0/1)

// ---------------- Mode B (ScanRIS): PC stream + TinyLFU + RIS -----
static constexpr uint32_t PC_TBL_SIZE = 512;
static inline uint32_t pc_index(uint64_t pc){ return (uint32_t)pc & (PC_TBL_SIZE-1u); }
static inline uint16_t line10(uint64_t paddr){ return (uint16_t)((paddr>>6) & 0x03FFu); } // 64B line -> 10b

static uint16_t pc_last_line10[PC_TBL_SIZE]; // 0xFFFF = uninit
static uint8_t  pc_stream_conf[PC_TBL_SIZE]; // 0..3, 2=armed, 3=locked
static uint8_t  pc_use4[PC_TBL_SIZE];        // TinyLFU 0..15

// 64-entry RIS keyed by (PC⊕set)&63; 0..7 short-reuse confidence
static uint8_t ris_score[64];

// ---------------- Selector state (bandit with leaders) ------------
static int8_t  leaderA_slot_score[64]; // [-63..63]
static int8_t  leaderB_slot_score[64]; // [-63..63]
static int32_t leaderA_sum = 0, leaderB_sum = 0;
static bool    prefer_B = false;
static uint64_t access_count = 0;

// ---------------- Helpers -----------------------------------------
static inline void sat_inc_u2(uint8_t &x){ if (x<3) x++; }
static inline void sat_dec_u2(uint8_t &x){ if (x>0) x--; }
static inline void sat_inc_u3(uint8_t &x){ if (x<7) x++; }
static inline void sat_dec_u3(uint8_t &x){ if (x>0) x--; }
static inline void sat_inc_u4(uint8_t &x){ if (x<15) x++; }
static inline void sat_dec_u4(uint8_t &x){ if (x>0) x--; }
static inline void sat_inc_i8(int8_t &x){ if (x<63) x++; }
static inline void sat_dec_i8(int8_t &x){ if (x>-63) x--; }

static inline bool detect_and_update_stream(uint64_t PC, uint64_t paddr){
  uint32_t idx = pc_index(PC);
  uint16_t ln  = line10(paddr);
  uint16_t last = pc_last_line10[idx];
  bool forward = false;
  if (last != 0xFFFFu){
    uint16_t exp1 = (uint16_t)(last + 1);
    uint16_t exp2 = (uint16_t)(last + 2);
    forward = (ln == exp1) || (ln == exp2);
  }
  if (forward) sat_inc_u2(pc_stream_conf[idx]);
  else pc_stream_conf[idx] = 0;
  pc_last_line10[idx] = ln;
  return (pc_stream_conf[idx] >= STREAM_ARM_THRESH);
}

static inline bool modeB_enabled(uint32_t set){
  if (LEADER_B(set)) return true;   // force B on B-leaders
  if (LEADER_A(set)) return false;  // force A on A-leaders
  uint32_t slot = LEADER_SLOT(set);
  int8_t delta = (int8_t)(leaderB_slot_score[slot] - leaderA_slot_score[slot]);
  return prefer_B && (delta >= MODEB_ENABLE_THRESH);
}

// ---------------- Init --------------------------------------------
void InitReplacementState(){
  std::memset(rrpv, maxRRPV, sizeof(rrpv));
  std::memset(hitcnt, 0, sizeof(hitcnt));
  std::memset(stream_tag, 0, sizeof(stream_tag));

  std::memset(shct_demand, 0, sizeof(shct_demand));
  std::memset(shct_prefetch, 0, sizeof(shct_prefetch));
  std::memset(hawk_sig, 0, sizeof(hawk_sig));
  std::memset(hawk_used, 0, sizeof(hawk_used));
  std::memset(hawk_is_pref, 0, sizeof(hawk_is_pref));
  std::memset(hawk_valid, 0, sizeof(hawk_valid));

  for (uint32_t i=0;i<PC_TBL_SIZE;i++){
    pc_last_line10[i] = 0xFFFFu;
    pc_stream_conf[i] = 0;
    pc_use4[i] = 0;
  }
  std::memset(ris_score, 0, sizeof(ris_score));
  std::memset(leaderA_slot_score, 0, sizeof(leaderA_slot_score));
  std::memset(leaderB_slot_score, 0, sizeof(leaderB_slot_score));
  leaderA_sum = leaderB_sum = 0;
  prefer_B = false;
  access_count = 0;
}

// ---------------- Victim selection --------------------------------
uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type){
  // Prefer invalid way
  for (uint32_t w=0; w<LLC_WAYS; w++){
    if (!current_set[w].valid) return w;
  }

  // Find any maxRRPV line; age until one appears (bounded)
  for (int iter=0; iter<8; iter++){
    for (uint32_t w=0; w<LLC_WAYS; w++){
      if (rrpv[set][w] == maxRRPV) return w;
    }
    rrpv_age_all(set);
  }

  // Fallback: return the way with the highest RRPV
  uint32_t victim = 0, best = 0;
  for (uint32_t w=0; w<LLC_WAYS; w++){
    if (rrpv[set][w] >= best){ best = rrpv[set][w]; victim = w; }
  }
  return victim;
}

// ---------------- State update ------------------------------------
void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit){
  if (is_writeback(type)) return;

  // Stream detection and TinyLFU update
  bool stream_armed = detect_and_update_stream(PC, paddr);
  uint32_t pidx = pc_index(PC);
  if (hit && is_demand(type)) sat_inc_u4(pc_use4[pidx]);

  // Leader reward bookkeeping (hits +1, misses -1)
  if (SEL_SAMPLED(set)){
    uint32_t slot = LEADER_SLOT(set);
    if (hit){
      if (LEADER_A(set)){ sat_inc_i8(leaderA_slot_score[slot]); leaderA_sum++; }
      else               { sat_inc_i8(leaderB_slot_score[slot]); leaderB_sum++; }
    }else{
      if (LEADER_A(set)){ sat_dec_i8(leaderA_slot_score[slot]); leaderA_sum--; }
      else               { sat_dec_i8(leaderB_slot_score[slot]); leaderB_sum--; }
    }
  }

  if (hit){
    // On hit: multi-hit promotion; stream quarantine control
    if (stream_tag[set][way]){
      // Quarantined stream/prefetch
      if (is_demand(type) && hitcnt[set][way] < 3) hitcnt[set][way]++;
      bool hot = (pc_use4[pidx] >= PC_USE_HOT_THRESH);
      uint8_t need = hot ? 1 : HITS_PROMOTE_STR;
      if (hitcnt[set][way] >= need){
        rrpv_set(set, way, 0);
        stream_tag[set][way] = 0; // escape quarantine
      }else if (STREAM_DEMOTE_TOUCH){
        if (rrpv[set][way] < maxRRPV) rrpv[set][way]++;
      }
    }else{
      // Regular line
      if (is_demand(type)){
        if (hitcnt[set][way] < 3) hitcnt[set][way]++;
        if (hitcnt[set][way] >= HITS_PROMOTE_NS) rrpv_set(set, way, 0);
        else if (rrpv[set][way] > 0) rrpv[set][way]--;
        // RIS: quick reuse evidence
        uint32_t slot = ((uint32_t)((PC>>2) ^ set)) & 63u;
        if (rrpv[set][way] <= 2) sat_inc_u3(ris_score[slot]);
      }else{
        // prefetch hit: minor nudge only
        if (rrpv[set][way] > 0) rrpv[set][way]--;
      }
    }

    // Mark reuse in leaders for Mode A training
    if (SEL_SAMPLED(set)){
      uint32_t slot = LEADER_SLOT(set);
      hawk_used[slot][way] = 1;
    }
  }else{
    // Miss: train Mode A on eviction for leaders, then (re)insert
    if (SEL_SAMPLED(set)){
      uint32_t slot = LEADER_SLOT(set);
      if (hawk_valid[slot][way]){
        uint16_t sig = hawk_sig[slot][way];
        uint32_t idx = shct_idx(sig);
        if (hawk_is_pref[slot][way]){
          if (hawk_used[slot][way]) { if (shct_prefetch[idx] < SHCT_MAX) shct_prefetch[idx]++; }
          else                      { if (shct_prefetch[idx] > 0)        shct_prefetch[idx]--; }
        }else{
          if (hawk_used[slot][way]) { if (shct_demand[idx] < SHCT_MAX) shct_demand[idx]++; }
          else                      { if (shct_demand[idx] > 0)        shct_demand[idx]--; }
        }
      }
      // Install new signature for the incoming line
      hawk_valid[slot][way]    = 1;
      hawk_sig[slot][way]      = pc_sig12(PC);
      hawk_used[slot][way]     = 0;
      hawk_is_pref[slot][way]  = is_prefetch(type) ? 1 : 0;
    }

    // Decide insertion using selector
    bool enableB = modeB_enabled(set);
    uint8_t depth = INSERT_COLD_DEPTH;
    bool tag_stream = false;
    uint32_t slot = ((uint32_t)((PC>>2) ^ set)) & 63u;
    bool hot_pc = (pc_use4[pidx] >= PC_USE_HOT_THRESH);

    if (enableB){
      if (is_prefetch(type)){
        depth = STREAM_TAIL_DEPTH;
        tag_stream = true;
      }else if (stream_armed){
        depth = hot_pc ? STREAM_SHALLOW_TAIL : STREAM_TAIL_DEPTH;
        tag_stream = true;
        // penalize RIS for detected stream
        sat_dec_u3(ris_score[slot]);
      }else{
        depth = (ris_score[slot] >= RIS_SHORT_THRESH) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
      }
    }else{
      // Mode A: Hawkeye-lite via SHCT
      if (is_prefetch(type)) depth = STREAM_TAIL_DEPTH;
      else {
        uint8_t val = shct_demand[shct_idx(pc_sig12(PC))];
        depth = (val >= 16) ? INSERT_WARM_DEPTH : INSERT_COLD_DEPTH;
      }
    }

    rrpv_set(set, way, depth);
    hitcnt[set][way] = 0;
    stream_tag[set][way] = tag_stream ? 1 : 0;
  }

  // Periodic maintenance
  access_count++;
  if ((access_count % LFU_DECAY_PERIOD) == 0){
    for (uint32_t i=0;i<PC_TBL_SIZE;i++) pc_use4[i] >>= 1;
  }
  if ((access_count % BANDIT_EPOCH) == 0){
    int32_t sumA = 0, sumB = 0;
    for (uint32_t s=0; s<64; s++){
      sumA += leaderA_slot_score[s];
      sumB += leaderB_slot_score[s];
      // light decay toward 0 to keep responsiveness
      if (leaderA_slot_score[s] > 0) leaderA_slot_score[s]--;
      else if (leaderA_slot_score[s] < 0) leaderA_slot_score[s]++;
      if (leaderB_slot_score[s] > 0) leaderB_slot_score[s]--;
      else if (leaderB_slot_score[s] < 0) leaderB_slot_score[s]++;
      // RIS slow decay
      sat_dec_u3(ris_score[s]);
    }
    prefer_B = ((sumB - sumA) > GATE_MARGIN);
    leaderA_sum = leaderB_sum = 0; // not strictly needed but keep clean
  }
}

void PrintStats(){}
void PrintStats_Heartbeat(){}