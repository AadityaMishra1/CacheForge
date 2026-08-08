// mockingjay.cc -- Mockingjay LLC replacement policy ported to ChampSim_CRC2

#include "champsim_crc2.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_map>

#define NUM_CPUS 1
#define LOG2_BLOCK_SIZE 6
#define LLC_SETS (NUM_CPUS * 2048)
#define LLC_WAYS 16

constexpr int HISTORY     = 8;
constexpr int GRANULARITY = 8;

constexpr int INF_RD  = LLC_WAYS * HISTORY - 1;
constexpr int INF_ETR = (LLC_WAYS * HISTORY / GRANULARITY) - 1;
constexpr int MAX_RD  = INF_RD - 22;

constexpr int SAMPLED_CACHE_WAYS      = 5;
constexpr int LOG2_SAMPLED_CACHE_SETS = 4;  // 16 sampled-cache sets

// These depend on LLC geometry; compute them at init.
static int LOG2_LLC_SET          = 0;
static int LOG2_LLC_SIZE         = 0;
static int LOG2_SAMPLED_SETS     = 0;
static int SAMPLED_CACHE_TAG_BITS = 0;
static int PC_SIGNATURE_BITS      = 0;

constexpr int TIMESTAMP_BITS = 8;

// ------------------------------------------------------------
// Global state
// ------------------------------------------------------------

// ETR (Expected Time to Reuse) state and per-set clocks
static int etr[LLC_SETS][LLC_WAYS];
static int etr_clock[LLC_SETS];

// Reuse-distance predictor: maps PC signatures to RD estimates
static std::unordered_map<uint32_t, int> rdp;

// Per-set timestamps (for sampled sets)
static int current_timestamp[LLC_SETS];

// Sampled cache lines
struct SampledCacheLine {
    bool     valid;
    uint64_t tag;
    uint64_t signature;
    int      timestamp;
};

// Sampled cache: index -> array of SampledCacheLine[SAMPLED_CACHE_WAYS]
static std::unordered_map<uint32_t, SampledCacheLine*> sampled_cache;

// Temperature / penalty constants
constexpr double TEMP_DIFFERENCE = 1.0 / 16.0;
constexpr double FLEXMIN_PENALTY = 2.0 - std::log2(static_cast<double>(NUM_CPUS)) / 4.0;

// ------------------------------------------------------------
// Helper functions (ported from original Mockingjay)
// ------------------------------------------------------------

static bool is_sampled_set(int set)
{
    int mask_length = LOG2_LLC_SET - LOG2_SAMPLED_SETS;
    int mask        = (1 << mask_length) - 1;
    return (set & mask)
        == ((set >> (LOG2_LLC_SET - mask_length)) & mask);
}

static uint64_t CRC_HASH(uint64_t _blockAddress)
{
    static const unsigned long long crcPolynomial = 3988292384ULL;
    unsigned long long _returnVal                 = _blockAddress;

    for (unsigned int i = 0; i < 3; i++) {
        if ((_returnVal & 1ULL) == 1ULL)
            _returnVal = (_returnVal >> 1) ^ crcPolynomial;
        else
            _returnVal = (_returnVal >> 1);
    }

    return _returnVal;
}

// PC signature computation (logic from the original code)
static uint64_t get_pc_signature(uint64_t pc, bool hit, bool prefetch, uint32_t core)
{
    if (NUM_CPUS == 1) {
        pc <<= 1;
        if (hit) {
            pc |= 1ULL;
        }
        pc <<= 1;
        if (prefetch) {
            pc |= 1ULL;
        }
        pc = CRC_HASH(pc);
        pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    } else {
        pc <<= 1;
        if (prefetch) {
            pc |= 1ULL;
        }
        pc <<= 2;
        pc |= static_cast<uint64_t>(core);
        pc = CRC_HASH(pc);
        pc = (pc << (64 - PC_SIGNATURE_BITS)) >> (64 - PC_SIGNATURE_BITS);
    }

    return pc;
}

static uint32_t get_sampled_cache_index(uint64_t full_addr)
{
    full_addr >>= LOG2_BLOCK_SIZE;
    full_addr = (full_addr << (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET)))
              >> (64 - (LOG2_SAMPLED_CACHE_SETS + LOG2_LLC_SET));
    return static_cast<uint32_t>(full_addr);
}

static uint64_t get_sampled_cache_tag(uint64_t x)
{
    x >>= (LOG2_LLC_SET + LOG2_BLOCK_SIZE + LOG2_SAMPLED_CACHE_SETS);
    x = (x << (64 - SAMPLED_CACHE_TAG_BITS)) >> (64 - SAMPLED_CACHE_TAG_BITS);
    return x;
}

static int search_sampled_cache(uint64_t blockAddress, uint32_t index)
{
    auto it = sampled_cache.find(index);
    if (it == sampled_cache.end() || !it->second)
        return -1;

    SampledCacheLine* sampled_set = it->second;
    for (int way = 0; way < SAMPLED_CACHE_WAYS; way++) {
        if (sampled_set[way].valid && sampled_set[way].tag == blockAddress)
            return way;
    }

    return -1;
}

static void detrain(uint32_t index, int way)
{
    auto it = sampled_cache.find(index);
    if (it == sampled_cache.end() || !it->second)
        return;

    SampledCacheLine* set = it->second;
    SampledCacheLine  temp = set[way];

    if (!temp.valid)
        return;

    uint32_t sig = static_cast<uint32_t>(temp.signature);
    auto      rd_it = rdp.find(sig);

    if (rd_it != rdp.end()) {
        rd_it->second = std::min(rd_it->second + 1, INF_RD);
    } else {
        rdp[sig] = INF_RD;
    }

    set[way].valid = false;
}

static int temporal_difference(int init, int sample)
{
    if (sample > init) {
        int diff = sample - init;
        diff     = static_cast<int>(diff * TEMP_DIFFERENCE);
        diff     = std::min(1, diff);
        return std::min(init + diff, INF_RD);
    } else if (sample < init) {
        int diff = init - sample;
        diff     = static_cast<int>(diff * TEMP_DIFFERENCE);
        diff     = std::min(1, diff);
        return std::max(init - diff, 0);
    } else {
        return init;
    }
}

static int increment_timestamp(int input)
{
    input++;
    input %= (1 << TIMESTAMP_BITS);
    return input;
}

static int time_elapsed(int global, int local)
{
    if (global >= local)
        return global - local;

    global += (1 << TIMESTAMP_BITS);
    return global - local;
}

// ------------------------------------------------------------
// CRC2 Replacement Interface
// ------------------------------------------------------------

void InitReplacementState()
{
    // Compute derived geometry-dependent fields
    LOG2_LLC_SET  = static_cast<int>(std::log2(static_cast<double>(LLC_SETS)));
    LOG2_LLC_SIZE = LOG2_LLC_SET
                  + static_cast<int>(std::log2(static_cast<double>(LLC_WAYS)))
                  + LOG2_BLOCK_SIZE;
    LOG2_SAMPLED_SETS     = LOG2_LLC_SIZE - 16;
    SAMPLED_CACHE_TAG_BITS = 31 - LOG2_LLC_SIZE;
    PC_SIGNATURE_BITS      = LOG2_LLC_SIZE - 10;

    // Initialize ETR state and timestamps
    for (int set = 0; set < LLC_SETS; set++) {
        etr_clock[set]        = GRANULARITY;
        current_timestamp[set] = 0;
        for (int way = 0; way < LLC_WAYS; way++) {
            etr[set][way] = 0;
        }
    }

    // Initialize sampled cache
    sampled_cache.clear();
    int modifier = 1 << LOG2_LLC_SET;
    int limit    = 1 << LOG2_SAMPLED_SETS;

    for (uint32_t set = 0; set < LLC_SETS; set++) {
        if (is_sampled_set(static_cast<int>(set))) {
            for (int i = 0; i < limit; i++) {
                uint32_t idx = set + modifier * i;
                sampled_cache[idx] = new SampledCacheLine[SAMPLED_CACHE_WAYS]();
            }
        }
    }

    rdp.clear();

    std::cout << "[Mockingjay] Initialized replacement state for LLC with "
              << LLC_SETS << " sets, " << LLC_WAYS << " ways.\n";
}

// Victim selection: original ETR logic + RD-based bypass
uint32_t GetVictimInSet(uint32_t      cpu,
                        uint32_t      set,
                        const BLOCK*  current_set,
                        uint64_t      ip,
                        uint64_t      full_addr,
                        uint32_t      type)
{
    // 1) If there's an invalid block, choose it
    for (uint32_t way = 0; way < LLC_WAYS; way++) {
        if (!current_set[way].valid) {
            return way;
        }
    }

    // 2) Choose victim way by ETR (largest |etr|, tie-breaker: negative)
    int max_etr    = 0;
    int victim_way = 0;

    for (int way = 0; way < LLC_WAYS; way++) {
        int val = etr[set][way];
        int mag = std::abs(val);
        if (mag > max_etr || (mag == max_etr && val < 0)) {
            max_etr    = mag;
            victim_way = way;
        }
    }

    // 3) RD-based bypass decision, same as original Mockingjay
    uint64_t pc_sig64 = get_pc_signature(ip, false, type == PREFETCH, cpu);
    uint32_t pc_sig   = static_cast<uint32_t>(pc_sig64);

    auto rd_it = rdp.find(pc_sig);
    if (type != WRITEBACK && rd_it != rdp.end()) {
        int rd = rd_it->second;
        if (rd > MAX_RD || (rd / GRANULARITY) > max_etr) {
            return LLC_WAYS;
        }
    }

    return static_cast<uint32_t>(victim_way);
}

void UpdateReplacementState(uint32_t cpu,
                            uint32_t set,
                            uint32_t way,
                            uint64_t full_addr,
                            uint64_t ip,
                            uint64_t /*victim_addr*/,
                            uint32_t type,
                            uint8_t  hit)
{
    // WRITEBACK handling (original behavior)
    if (type == WRITEBACK) {
        if (!hit) {
            etr[set][way] = -INF_ETR;
        }
        return;
    }

    // Convert PC to signature
    uint64_t pc_sig64 = get_pc_signature(ip, hit != 0, type == PREFETCH, cpu);
    uint32_t pc_sig   = static_cast<uint32_t>(pc_sig64);

    // ------------------ Sampled-cache learning ------------------
    if (is_sampled_set(static_cast<int>(set))) {
        uint32_t sampled_cache_index = get_sampled_cache_index(full_addr);
        uint64_t sampled_cache_tag   = get_sampled_cache_tag(full_addr);
        int      sampled_cache_way   = search_sampled_cache(sampled_cache_tag,
                                                            sampled_cache_index);

        // Existing entry in sampled cache: learn RD
        if (sampled_cache_way > -1) {
            SampledCacheLine& line          = sampled_cache[sampled_cache_index][sampled_cache_way];
            uint64_t          last_signature = line.signature;
            int               last_timestamp = line.timestamp;

            int sample = time_elapsed(current_timestamp[set], last_timestamp);

            if (sample <= INF_RD) {
                if (type == PREFETCH) {
                    sample = static_cast<int>(sample * FLEXMIN_PENALTY);
                }

                uint32_t last_sig_u32 = static_cast<uint32_t>(last_signature);
                auto     rd_it        = rdp.find(last_sig_u32);

                if (rd_it != rdp.end()) {
                    int init = rd_it->second;
                    rd_it->second = temporal_difference(init, sample);
                } else {
                    rdp[last_sig_u32] = sample;
                }

                line.valid = false;
            }
        }

        // Choose a line in sampled cache to evict and detrain
        int lru_way = -1;
        int lru_rd  = -1;

        SampledCacheLine* set_lines = sampled_cache[sampled_cache_index];
        for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
            if (!set_lines[w].valid) {
                lru_way = w;
                lru_rd  = INF_RD + 1;
                continue;
            }

            int last_ts = set_lines[w].timestamp;
            int sample  = time_elapsed(current_timestamp[set], last_ts);

            if (sample > INF_RD) {
                lru_way = w;
                lru_rd  = INF_RD + 1;
                detrain(sampled_cache_index, w);
            } else if (sample > lru_rd) {
                lru_way = w;
                lru_rd  = sample;
            }
        }

        if (lru_way >= 0) {
            detrain(sampled_cache_index, lru_way);
        }

        // Insert current access into sampled cache
        for (int w = 0; w < SAMPLED_CACHE_WAYS; w++) {
            if (!set_lines[w].valid) {
                set_lines[w].valid     = true;
                set_lines[w].signature = pc_sig64;
                set_lines[w].tag       = sampled_cache_tag;
                set_lines[w].timestamp = current_timestamp[set];
                break;
            }
        }

        // Advance per-set timestamp
        current_timestamp[set] = increment_timestamp(current_timestamp[set]);
    }

    // ------------------ ETR update / aging ------------------

    // Age other lines every GRANULARITY accesses
    if (etr_clock[set] == GRANULARITY) {
        for (int w = 0; w < LLC_WAYS; w++) {
            if (static_cast<uint32_t>(w) != way && std::abs(etr[set][w]) < INF_ETR) {
                etr[set][w]--;
            }
        }
        etr_clock[set] = 0;
    }
    etr_clock[set]++;

    // Set ETR for the touched way based on RD prediction
    if (way < LLC_WAYS) {
        auto rd_it = rdp.find(pc_sig);

        if (rd_it == rdp.end()) {
            if (NUM_CPUS == 1) {
                etr[set][way] = 0;
            } else {
                etr[set][way] = INF_ETR;
            }
        } else {
            int rd = rd_it->second;
            if (rd > MAX_RD) {
                etr[set][way] = INF_ETR;
            } else {
                etr[set][way] = rd / GRANULARITY;
            }
        }
    }
}

void DumpStats()
{
    // Optional: add custom policy stats here if needed.
}

void PrintStats_Heartbeat()
{
    // Optional: print running stats if desired.
}

void PrintStats()
{
    // Optional: print final stats if desired.
}
