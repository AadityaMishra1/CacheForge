from typing import List, Dict
from RAG import ExperimentRAG
import os

class PolicyPromptGenerator:
    def __init__(self, db_path: str = 'funsearch.db'):
        self.rag = ExperimentRAG(db_path)
    
    def _get_code_template(self) -> str:
        """Returns the required C++ code template matching ChampSim CRC2 interface"""
        return '''```cpp
#include <vector>
#include <cstdint>
#include <iostream>
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE * 2048)
#define LLC_WAYS 16

// ============================================================================
// DECLARE YOUR GLOBAL STATE ARRAYS HERE (static keyword required)
// Example: static uint8_t rrpv[LLC_SETS][LLC_WAYS];
// ============================================================================


// Initialize replacement state
void InitReplacementState() {
    // Initialize your global arrays here
    // --- IMPLEMENT THE FUNCTION ---
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
    // Use your global state arrays to select victim
    // --- IMPLEMENT THE FUNCTION ---
    return 0; // replaced block index
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
    // Update your global state arrays based on access
    // --- IMPLEMENT THE FUNCTION ---
}

// Print end-of-simulation statistics
void PrintStats() {
    // --- KEEP THIS FUNCTION BLANK ---
}

// Print periodic (heartbeat) statistics
void PrintStats_Heartbeat() {
    // --- KEEP THIS FUNCTION BLANK ---
}
```
'''
    
    def _read_policy_code(self, file_path: str) -> str:
        """Reads the C++ code from a file, raising an error if not found"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise
    
    def generate_prompt(self, workload: str) -> str:
        """Generate a detailed prompt with actual policy implementations"""
        top_policies = self.rag.get_top_policies_by_cache_hit(workload, top_n=2)
        if not top_policies:
            return f"No data available for workload: {workload}"
        
        parts: List[str] = [
            "You are a cache policy design expert. Analyze the workload and top policies, then create a new improved policy.\n\n",
            "# Workload\n",
            f"Name: {workload}\n",
            f"Description: {top_policies[0]['workload_description']}\n\n",
            "# Examples\n"
        ]
        for i, policy in enumerate(top_policies, 1):
            code = self._read_policy_code(policy['cpp_file_path'])
            parts += [
                f"## Policy {i}\n",
                f"Name: {policy['policy']}\n",
                f"Description: {policy['policy_description']}\n",
                f"Cache Hit Rate: {policy['cache_hit_rate']:.2%}\n",
                "Implementation:\n",
                f"```cpp\n{code}\n```\n\n"
            ]
        parts += [
            "# Task\nCreate a new cache replacement policy in C++17 that combines strengths and fixes weaknesses.\n\n",
            "Your response MUST follow exactly this format:\n",
            "## Policy Name\n[Policy name]\n\n",
            "## Policy Description\n[One-paragraph explanation]\n\n",
            "## C++ Implementation\n",
            self._get_code_template(),
            "# Guidelines\n",
            "1. Include \"../inc/champsim_crc2.h\" at the very top.\n",
            "2. Implement all five functions: InitReplacementState, GetVictimInSet, UpdateReplacementState, PrintStats, PrintStats_Heartbeat.\n",
            "3. In GetVictimInSet, **do not bypass** (i.e., return `LLC_WAYS`) on **WRITEBACK** accesses—only allow bypass for LOAD/RFO/PREFETCH when predictor says 'cold'"
            "4. Use the BLOCK* current_set pointer and check its .valid field for empty ways.\n",
            "5. Combine the best ideas; add comments to explain design choices.\n"
        ]
        return ''.join(parts)
    
    def close(self):
        self.rag.close()

# Example usage
if __name__ == '__main__':
    generator = PolicyPromptGenerator('funsearch.db')
    try:
        prompt = generator.generate_prompt('Astar')
        print(prompt)
    finally:
        generator.close()
