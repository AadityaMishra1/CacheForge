// Hybrid ensemble: Hawkeye baseline (Mode A) + FUSE-RunGuard (Mode B)
// Followers default to Hawkeye and only switch to FUSE when leader sets show a win.

#include "../../inc/champsim_crc2.h"
#include <map>
#include <vector>

#define NUM_CORE 1
#define LLC_SETS NUM_CORE*2048
#define LLC_WAYS 16

// Access types
static constexpr uint32_t ACCESS_LOAD      = 0;
static constexpr uint32_t ACCESS_RFO       = 1;
static constexpr uint32_t ACCESS_PREFETCH  = 2;
static constexpr uint32_t ACCESS_WRITEBACK = 3;

// ---------------- Selector (64 sampled sets) ----------------
static constexpr uint32_t LLC_SET_BITS = 11; // log2(2048)
static inline bool SEL_SAMPLED(uint32_t set) {
    return ((set & 63u) == ((set >> (LLC_SET_BITS - 6)) & 63u));
}
static inline bool LEADER_A(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 0u); }
static inline bool LEADER_B(uint32_t set) { return SEL_SAMPLED(set) && ((set & 1u) == 1u); }

// Global selector; strong bias to Hawkeye
static constexpr uint8_t GSEL_MAX   = 31;
static constexpr uint8_t GSEL_THRES = 26; // need clear B advantage
static uint8_t GSEL = 0;

// ============================================================
// Mode A: Hawkeye (unchanged logic from hawkeye_final.cc)
// ============================================================
#define maxRRPV 7
static uint32_t rrpv[LLC_SETS][LLC_WAYS];

#define TIMER_SIZE 1024
static uint64_t perset_mytimer[LLC_SETS];

static uint64_t signatures[LLC_SETS][LLC_WAYS];
static bool prefetched[LLC_SETS][LLC_WAYS];

#define MAX_SHCT 31
#define SHCT_SIZE_BITS 11
#define SHCT_SIZE (1<<SHCT_SIZE_BITS)
#include "hawkeye_predictor.h"
static HAWKEYE_PC_PREDICTOR* demand_predictor;  //Predictor
static HAWKEYE_PC_PREDICTOR* prefetch_predictor;  //Predictor

#define OPTGEN_VECTOR_SIZE 128
#include "optgen.h"
static OPTgen perset_optgen[LLC_SETS]; // per-set occupancy vectors; we only use 64 of these

#include <math.h>
#define bitmask(l) (((l) == 64) ? (unsigned long long)(-1LL) : ((1LL << (l))-1LL))
#define bits(x, i, l) (((x) >> (i)) & bitmask(l))
#define SAMPLED_SET(set) (bits(set, 0 , 6) == bits(set, ((unsigned long long)log2(LLC_SETS) - 6), 6) )

#define SAMPLED_CACHE_SIZE 2800
#define SAMPLER_WAYS 8
#define SAMPLER_SETS SAMPLED_CACHE_SIZE/SAMPLER_WAYS
static vector<map<uint64_t, ADDR_INFO> > addr_history; // Sampler

static void replace_addr_history_element(unsigned int sampler_set)
{
    uint64_t lru_addr = 0;
    for(map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++)
    {
        if((it->second).lru == (SAMPLER_WAYS-1))
        {
            lru_addr = it->first;
            break;
        }
    }
    addr_history[sampler_set].erase(lru_addr);
}

static void update_addr_history_lru(unsigned int sampler_set, unsigned int curr_lru)
{
    for(map<uint64_t, ADDR_INFO>::iterator it=addr_history[sampler_set].begin(); it != addr_history[sampler_set].end(); it++)
    {
        if((it->second).lru < curr_lru)
        {
            (it->second).lru++;
        }
    }
}

// Hawkeye victim
static uint32_t hawk_victim(uint32_t set, const BLOCK *current_set)
{
    for (uint32_t i=0; i<LLC_WAYS; i++)
        if (!current_set[i].valid)
            return i;

    for (uint32_t i=0; i<LLC_WAYS; i++)
        if (rrpv[set][i] == maxRRPV)
            return i;

    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for (uint32_t i=0; i<LLC_WAYS; i++)
    {
        if (rrpv[set][i] >= max_rrip)
        {
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }

    if (lru_victim == -1) return 0;
    return (uint32_t)lru_victim;
}

// Hawkeye update
static void hawk_update(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    paddr = (paddr >> 6) << 6;

    if(type == ACCESS_PREFETCH)
    {
        if (!hit)
            prefetched[set][way] = true;
    }
    else
        prefetched[set][way] = false;

    if (type == ACCESS_WRITEBACK)
        return;

    if(SAMPLED_SET(set))
    {
        uint64_t curr_quanta = perset_mytimer[set] % OPTGEN_VECTOR_SIZE;

        uint32_t sampler_set = (paddr >> 6) % SAMPLER_SETS; 
        uint64_t sampler_tag = CRC(paddr >> 12) % 256;

        if((addr_history[sampler_set].find(sampler_tag) != addr_history[sampler_set].end()) && (type != ACCESS_PREFETCH))
        {
            unsigned int curr_timer = perset_mytimer[set];
            if(curr_timer < addr_history[sampler_set][sampler_tag].last_quanta)
               curr_timer = curr_timer + TIMER_SIZE;
            bool wrap =  ((curr_timer - addr_history[sampler_set][sampler_tag].last_quanta) > OPTGEN_VECTOR_SIZE);
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;
            if( !wrap && perset_optgen[set].should_cache(curr_quanta, last_quanta))
            {
                if(addr_history[sampler_set][sampler_tag].prefetched)
                    prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                else
                    demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
            }
            else
            {
                if(addr_history[sampler_set][sampler_tag].prefetched)
                    prefetch_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
                else
                    demand_predictor->decrement(addr_history[sampler_set][sampler_tag].PC);
            }
            perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);
            addr_history[sampler_set][sampler_tag].prefetched = false;
        }
        else if(addr_history[sampler_set].find(sampler_tag) == addr_history[sampler_set].end())
        {
            if(addr_history[sampler_set].size() == SAMPLER_WAYS) 
                replace_addr_history_element(sampler_set);

            addr_history[sampler_set][sampler_tag].init(curr_quanta);
            if(type == ACCESS_PREFETCH)
            {
                addr_history[sampler_set][sampler_tag].mark_prefetch();
                perset_optgen[set].add_prefetch(curr_quanta);
            }
            else
                perset_optgen[set].add_access(curr_quanta);
            update_addr_history_lru(sampler_set, SAMPLER_WAYS-1);
        }
        else
        {
            uint64_t last_quanta = addr_history[sampler_set][sampler_tag].last_quanta % OPTGEN_VECTOR_SIZE;
            if (perset_mytimer[set] - addr_history[sampler_set][sampler_tag].last_quanta < 5*NUM_CORE) 
            {
                if(perset_optgen[set].should_cache(curr_quanta, last_quanta))
                {
                    if(addr_history[sampler_set][sampler_tag].prefetched)
                        prefetch_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                    else
                       demand_predictor->increment(addr_history[sampler_set][sampler_tag].PC);
                }
            }

            addr_history[sampler_set][sampler_tag].mark_prefetch(); 
            perset_optgen[set].add_prefetch(curr_quanta);
            update_addr_history_lru(sampler_set, addr_history[sampler_set][sampler_tag].lru);
        }

        bool new_prediction = demand_predictor->get_prediction (PC);
        if (type == ACCESS_PREFETCH)
            new_prediction = prefetch_predictor->get_prediction (PC);
        addr_history[sampler_set][sampler_tag].update(perset_mytimer[set], PC, new_prediction);
        addr_history[sampler_set][sampler_tag].lru = 0;
        perset_mytimer[set] = (perset_mytimer[set]+1) % TIMER_SIZE;
    }

    bool new_prediction = demand_predictor->get_prediction (PC);
    if (type == ACCESS_PREFETCH)
        new_prediction = prefetch_predictor->get_prediction (PC);

    signatures[set][way] = PC;

    if(!new_prediction)
        rrpv[set][way] = maxRRPV;
    else
    {
        rrpv[set][way] = 0;
        if(!hit)
        {
            bool saturated = false;
            for(uint32_t i=0; i<LLC_WAYS; i++)
                if (rrpv[set][i] == maxRRPV-1)
                    saturated = true;

            for(uint32_t i=0; i<LLC_WAYS; i++)
            {
                if (!saturated && rrpv[set][i] < maxRRPV-1)
                    rrpv[set][i]++;
            }
        }
        rrpv[set][way] = 0;
    }
}

// ============================================================
// Mode B: FUSE-RunGuard (copied from 007_fuse_runguard__v0.cc)
// ============================================================
static constexpr uint8_t  PC_BITS_B        = 8;               // 256-entry tables
static constexpr uint32_t PC_SIZE_B        = (1u << PC_BITS_B);
static constexpr uint8_t  EUP_INIT_B       = 1;
static constexpr uint8_t  EUP_WARM_TH_B    = 2;
static constexpr uint8_t  LINE_LOW_BITS_B  = 12;
static constexpr uint8_t  RG_BYPASS_TH_B   = 2;
static constexpr uint8_t  AGE_MAX_B        = 3;
static constexpr uint8_t  AGE_YOUNG_B      = 1;
static constexpr uint8_t  HITS_TO_PROMOTE_B = 2;
enum : uint8_t { ST_NORMAL_B=0, ST_STREAM_B=1, ST_QUAR_B=2 };

static uint8_t AGE_B   [LLC_SETS][LLC_WAYS];
static uint8_t HITCNT_B[LLC_SETS][LLC_WAYS];
static uint8_t STATUS_B[LLC_SETS][LLC_WAYS];
static uint8_t PCSIG_B [LLC_SETS][LLC_WAYS];
static uint8_t EUP_B   [PC_SIZE_B];
static uint16_t RG_LAST_LINE_B[PC_SIZE_B];
static uint8_t  RG_CONF_B     [PC_SIZE_B];
static uint8_t  RG_LAST_ABS12_B[PC_SIZE_B];
static uint8_t  RG_LAST_FWD_B  [PC_SIZE_B];

static inline uint32_t pc_index_B(uint64_t pc) {
    uint64_t x = pc ^ (pc >> 7) ^ (pc >> 13) ^ (pc >> 19);
    return static_cast<uint32_t>(x) & (PC_SIZE_B - 1);
}
static inline uint16_t line_lowN_B(uint64_t paddr) {
    return static_cast<uint16_t>((paddr >> 6) & ((1u << LINE_LOW_BITS_B) - 1));
}
static inline bool is_demand(uint32_t type) {
    return (type == ACCESS_LOAD) || (type == ACCESS_RFO);
}
static inline void sat_inc_B(uint8_t &x, uint8_t maxv) { if (x < maxv) x++; }
static inline void sat_dec_B(uint8_t &x) { if (x > 0) x--; }

static inline void age_all_B(uint32_t set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (AGE_B[set][w] < AGE_MAX_B) AGE_B[set][w]++;
    }
}

static inline uint8_t update_runguard_B(uint64_t pc, uint64_t paddr) {
    uint32_t idx = pc_index_B(pc);
    uint16_t curr = line_lowN_B(paddr);
    int16_t delta = static_cast<int16_t>(static_cast<int32_t>(curr) - static_cast<int32_t>(RG_LAST_LINE_B[idx]));
    int16_t ad = (delta < 0) ? static_cast<int16_t>(-delta) : delta;
    bool abs12 = (ad == 1) || (ad == 2);
    bool fwd   = (delta > 0);

    if (abs12 && fwd) {
        if (RG_LAST_ABS12_B[idx] && RG_LAST_FWD_B[idx]) {
            sat_inc_B(RG_CONF_B[idx], 3);
        } else {
            if (RG_CONF_B[idx] == 0) RG_CONF_B[idx] = 1;
            RG_LAST_ABS12_B[idx] = 1;
            RG_LAST_FWD_B[idx]   = 1;
        }
    } else {
        sat_dec_B(RG_CONF_B[idx]);
        RG_LAST_ABS12_B[idx] = 0;
        RG_LAST_FWD_B[idx]   = 0;
    }
    RG_LAST_LINE_B[idx] = curr;
    return RG_CONF_B[idx];
}

static inline bool more_evictable_B(uint32_t set, uint32_t a, uint32_t b) {
    bool a_s = (STATUS_B[set][a] != ST_NORMAL_B);
    bool b_s = (STATUS_B[set][b] != ST_NORMAL_B);
    if (a_s != b_s) return a_s;

    uint8_t ea = EUP_B[ PCSIG_B[set][a] ];
    uint8_t eb = EUP_B[ PCSIG_B[set][b] ];
    if (ea != eb) return (ea < eb);

    uint8_t ha = HITCNT_B[set][a];
    uint8_t hb = HITCNT_B[set][b];
    if (ha != hb) return (ha < hb);

    return AGE_B[set][a] > AGE_B[set][b];
}

static uint32_t victim_B(uint32_t set, const BLOCK *current_set) {
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    uint32_t best = 0;
    for (uint32_t w = 1; w < LLC_WAYS; w++) {
        if (more_evictable_B(set, w, best)) best = w;
    }
    return best;
}

static void update_B(uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint32_t type, uint8_t hit) {
    const bool demand = is_demand(type);
    if (type == ACCESS_WRITEBACK) return;

    uint8_t rg_conf = 0;
    if (demand) {
        rg_conf = update_runguard_B(PC, paddr);
    }

    if (hit) {
        if (demand) {
            age_all_B(set);
            if (HITCNT_B[set][way] < HITS_TO_PROMOTE_B) HITCNT_B[set][way]++;

            if (HITCNT_B[set][way] >= HITS_TO_PROMOTE_B) {
                AGE_B[set][way] = 0;
                if (STATUS_B[set][way] != ST_NORMAL_B) STATUS_B[set][way] = ST_NORMAL_B;
                sat_inc_B(EUP_B[ PCSIG_B[set][way] ], 3);
            } else {
                if (AGE_B[set][way] > AGE_YOUNG_B) AGE_B[set][way] = AGE_YOUNG_B;
            }
        }
        return;
    }

    // Train EUP on eviction
    {
        uint8_t old_status = STATUS_B[set][way];
        uint8_t old_hits   = HITCNT_B[set][way];
        uint8_t old_sig    = PCSIG_B[set][way];

        if (old_status == ST_NORMAL_B) {
            if (old_hits >= HITS_TO_PROMOTE_B) sat_inc_B(EUP_B[old_sig], 3);
            else                               sat_dec_B(EUP_B[old_sig]);
        }
    }

    age_all_B(set);

    uint32_t sig = pc_index_B(PC);
    PCSIG_B[set][way]  = static_cast<uint8_t>(sig);
    HITCNT_B[set][way] = 0;

    if (type == ACCESS_PREFETCH) {
        STATUS_B[set][way] = ST_QUAR_B;
        AGE_B[set][way]    = AGE_MAX_B;
    } else if (demand) {
        if (rg_conf >= RG_BYPASS_TH_B) {
            STATUS_B[set][way] = ST_STREAM_B;
            AGE_B[set][way]    = AGE_MAX_B;
        } else {
            STATUS_B[set][way] = ST_NORMAL_B;
            if (EUP_B[sig] >= EUP_WARM_TH_B) AGE_B[set][way] = AGE_YOUNG_B;
            else                             AGE_B[set][way] = AGE_MAX_B;
        }
    } else {
        STATUS_B[set][way] = ST_QUAR_B;
        AGE_B[set][way]    = AGE_MAX_B;
    }
}

// ============================================================
// Top-level CRC2 API
// ============================================================
void InitReplacementState()
{
    // Hawkeye init
    for (int i=0; i<LLC_SETS; i++) {
        for (int j=0; j<LLC_WAYS; j++) {
            rrpv[i][j] = maxRRPV;
            signatures[i][j] = 0;
            prefetched[i][j] = false;
        }
        perset_mytimer[i] = 0;
        perset_optgen[i].init(LLC_WAYS-2);
    }
    addr_history.resize(SAMPLER_SETS);
    for (int i=0; i<SAMPLER_SETS; i++) 
        addr_history[i].clear();
    demand_predictor = new HAWKEYE_PC_PREDICTOR();
    prefetch_predictor = new HAWKEYE_PC_PREDICTOR();

    // FUSE init
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            AGE_B[s][w]    = AGE_MAX_B;
            HITCNT_B[s][w] = 0;
            STATUS_B[s][w] = ST_NORMAL_B;
            PCSIG_B[s][w]  = 0;
        }
    }
    for (uint32_t i = 0; i < PC_SIZE_B; i++) {
        EUP_B[i]          = EUP_INIT_B;
        RG_LAST_LINE_B[i] = 0;
        RG_CONF_B[i]      = 0;
        RG_LAST_ABS12_B[i]= 0;
        RG_LAST_FWD_B[i]  = 0;
    }
    GSEL = 0;

    std::cout << "Initialize Hawkeye + FUSE-RunGuard hybrid" << std::endl;
}

uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{
    bool useB = LEADER_B(set) || (!LEADER_A(set) && (GSEL >= GSEL_THRES));

    if (useB) return victim_B(set, current_set);
    return hawk_victim(set, current_set);
}

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    // Selector training on demand hits
    if (is_demand(type) && hit) {
        if (LEADER_B(set)) sat_inc_B(GSEL, GSEL_MAX);
        else if (LEADER_A(set)) sat_dec_B(GSEL);
    }

    // Always maintain Hawkeye state
    hawk_update(cpu, set, way, paddr, PC, victim_addr, type, hit);

    // Maintain FUSE state
    update_B(set, way, paddr, PC, type, hit);
}

void PrintStats_Heartbeat()
{

}

void PrintStats()
{
    unsigned int hits = 0;
    unsigned int accesses = 0;
    for(unsigned int i=0; i<LLC_SETS; i++)
    {
        accesses += perset_optgen[i].access;
        hits += perset_optgen[i].get_num_opt_hits();
    }

    std::cout << "OPTgen accesses: " << accesses << std::endl;
    std::cout << "OPTgen hits: " << hits << std::endl;
    std::cout << "OPTgen hit rate: " << 100*(double)hits/(double)accesses << std::endl;
    std::cout << "GSEL final: " << (unsigned)GSEL << std::endl;
    std::cout << std::endl;
}
