#include <algorithm>
#include <cstdint>
#include <cstring>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

using std::max;

struct LineMetadata {
    uint8_t frequency : 3;   // 3 bits for frequency counter
    uint8_t ever_hit : 1;    // 1 bit for tracking if ever hit
    uint8_t suspect_streaming : 1; // 1 bit for streaming suspect
    uint8_t prefetch : 1;    // 1 bit for prefetch status
    uint8_t epoch : 2;       // 2 bits for epoch

    LineMetadata() : frequency(0), ever_hit(0), suspect_streaming(0), prefetch(0), epoch(0) {}
};

// Replacement state arrays
LineMetadata line_metadata[LLC_SETS][LLC_WAYS];
uint8_t epoch = 0;
uint64_t epoch_timer = 0;
const uint64_t epoch_len = 64 * 1024;

void InitReplacementState() {
    std::memset(line_metadata, 0, sizeof(line_metadata));
    epoch = 0;
    epoch_timer = 0;
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type) {
    // First, look for an invalid (empty) way
    for(uint32_t way = 0; way < LLC_WAYS; way++) {
        if(current_set[way].valid == false) {
            return way;
        }
    }
    
    // Select victim among valid lines
    int selected_way = -1;
    uint8_t min_effF = 8; // max possible effF + 1
    uint8_t min_probation = 2; // 0 = probation, 1 = protected
    
    for (uint32_t way = 0; way < LLC_WAYS; ++way) {
        uint8_t effF = max(line_metadata[set][way].frequency - ((epoch - line_metadata[set][way].epoch) & 3), static_cast<uint8_t>(0));
        uint8_t probation = (effF < 2) ? 0 : 1; // Assume N=2

        if (probation < min_probation || (probation == min_probation && effF < min_effF)) {
            min_probation = probation;
            min_effF = effF;
            selected_way = way;
        }
    }

    return static_cast<uint32_t>(selected_way);
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    if (hit) {
        // Lazy decay on hits
        LineMetadata &line = line_metadata[set][way];
        uint8_t delta = (epoch - line.epoch) & 3;
        line.frequency = max(line.frequency - delta, static_cast<uint8_t>(0));
        line.epoch = epoch;

        if (type != PREFETCH) {
            if (!line.ever_hit) line.ever_hit = 1;
            line.frequency = std::min(static_cast<uint8_t>(7), line.frequency + 1);
        }

        if (line.frequency >= 2) { // Exceeds N, assume N=2 for protection
            line.suspect_streaming = 0;
        }
    } else {
        // On miss, handle evicted line based on hit statistics
        LineMetadata &victim_line = line_metadata[set][way];
        if (victim_line.ever_hit == 0) {
            // Decrement predictor (Pseudo-implemented, not shown)
        } else /* victim_line.ever_hit == 1 */ {
            // Increment predictor (Pseudo-implemented, not shown)
        }

        // Insert new line
        LineMetadata &new_line = line_metadata[set][way];
        uint8_t dead_pred = 0; // Assume dead_pred evaluation here, simplified
        uint8_t is_prefetch = (type == PREFETCH);

        new_line.frequency = (is_prefetch || dead_pred) ? 0 : 1;
        new_line.ever_hit = 0;
        new_line.suspect_streaming = dead_pred ? 1 : 0;
        new_line.prefetch = is_prefetch;
        new_line.epoch = epoch;
    }
    
    epoch_timer++;
    if (epoch_timer == epoch_len) {
        epoch = (epoch + 1) & 3;
        epoch_timer = 0;
    }
}

void PrintStats() {}

void PrintStats_Heartbeat() {}