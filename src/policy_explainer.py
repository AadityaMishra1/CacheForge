import os
import sys

# Try to initialize OpenAI/Groq client if API key is available
try:
    # Check if we're using Groq
    if os.getenv("LLM_PROVIDER", "").lower() == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
        EXPLAINER_MODEL = "llama-3.1-8b-instant"  # Fast Groq model for explanations
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        EXPLAINER_MODEL = "gpt-4o-mini"
except Exception:
    client = None
    EXPLAINER_MODEL = None

def load_canonical_policies():
    """Load descriptions of canonical cache replacement policies for RAG"""
    return """
    Reference Cache Replacement Policies:
    
    1. LRU (Least Recently Used):
       - Evicts the cache line that was accessed longest ago
       - Uses recency counters (log2(N) bits per line for N-way cache)
       - Works well for temporal locality, fails on scans
       - Storage: 4 bits per line for 16-way cache
    
    2. RRIP (Re-Reference Interval Prediction):
       - Predicts re-reference interval with small counters (2-3 bits)
       - Uses distant/long re-reference prediction for insertions
       - Resists thrashing from scanning patterns
       - SRRIP: Static RRIP with fixed insertion
       - DRRIP: Dynamic RRIP with set dueling
    
    3. SHiP (Signature-based Hit Predictor):
       - Augments recency with program context (PC signatures)
       - Uses a signature table to predict cache-friendliness
       - Combines RRIP-style counters with signature-based prediction
       - Tracks which PCs generate cache-friendly vs cache-averse lines
    
    4. Hawkeye:
       - Uses Belady's optimal algorithm (evict furthest future re-use)
       - Trains online classifier with sampled sets
       - Predicts cache-friendly vs cache-averse lines
       - Uses OPTgen to generate optimal labels for training
    
    5. LIME (Less is More):
       - Minimalist approach focusing on essential signals
       - Combines frequency and recency with minimal overhead
       - Designed for efficiency and simplicity
    """

def explain_policy_brief(policy_code: str) -> str:
    """
    Generate brief explanation for database storage.

    Args:
        policy_code: C++ source code as string

    Returns:
        Brief explanation (~100 words)
    """
    canonical_policies = load_canonical_policies()

    prompt = f"""Analyze this cache replacement policy briefly (under 100 words).

{canonical_policies}

Policy code:
```cpp
{policy_code[:2000]}
```

Provide in one paragraph:
- Core mechanism (1 sentence)
- Key innovation vs baselines
- Expected strong workloads (2-3 names)
- Storage estimate (X bits/line)
"""

    if client is None:
        return "[Explanation skipped: no API client configured]"

    try:
        response = client.chat.completions.create(
            model=EXPLAINER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Explanation error: {str(e)}]"


def explain_policy_structured(policy_code: str) -> dict:
    """
    Generate structured explanation for surrogate model features.

    Args:
        policy_code: C++ source code as string

    Returns:
        Dictionary with structured features for the policy
    """
    # Simple heuristic-based analysis for structured features
    code_lower = policy_code.lower()

    features = {
        "uses_lru": int("lru" in code_lower or "recency" in code_lower),
        "uses_frequency": int("freq" in code_lower or "counter" in code_lower),
        "uses_rrip": int("rrip" in code_lower or "rrpv" in code_lower),
        "uses_pc": int("pc" in code_lower or "signature" in code_lower),
        "uses_hawkeye": int("hawkeye" in code_lower or "optgen" in code_lower),
        "uses_bypass": int("bypass" in code_lower),
        "code_length": len(policy_code),
        "num_conditionals": policy_code.count("if "),
    }

    return features


def explain_policy(policy_path):
    """Analyze and explain a C++ cache replacement policy"""
    
    # Read the policy code
    with open(policy_path, 'r') as f:
        policy_code = f.read()
    
    # Get canonical policy descriptions for RAG
    canonical_policies = load_canonical_policies()
    
    # Build prompt
    prompt = f"""You are a computer architecture expert analyzing cache replacement policies.

{canonical_policies}

Now analyze this C++ cache replacement policy:
```cpp
{policy_code}
```

Provide a structured analysis:

1. **High-Level Summary**: What is the core logic of this policy? (2-3 sentences)

2. **Key Data Structures**: 
   - What metadata does it track per cache line or per set?
   - What do these data structures represent?
   - Estimate storage overhead in bits per line

3. **Replacement Logic**:
   - How does it decide which line to evict?
   - What signals does it use (recency, frequency, PC, address patterns, etc.)?
   - Walk through the find_victim() function logic

4. **Relationship to Canonical Policies**:
   - Which reference policy(ies) is this most similar to?
   - What novel modifications or combinations does it introduce?
   - Is it closer to LRU, RRIP, SHiP, Hawkeye, or a hybrid?

5. **Expected Behavior**:
   - What workload patterns should this policy handle well?
   - What patterns might it struggle with?
   - Which SPEC 2006 benchmarks would likely benefit most?

Be specific and technical. Reference line numbers, function names, or variable names from the code.
"""
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use mini for cost efficiency
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python policy_explainer.py <path_to_policy.cc>")
        print("\nExample:")
        print("  python policy_explainer.py ChampSim_CRC2/champ_repl_pol/lru.cc")
        sys.exit(1)
    
    policy_path = sys.argv[1]
    
    if not os.path.exists(policy_path):
        print(f"Error: File not found: {policy_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING POLICY: {os.path.basename(policy_path)}")
    print(f"{'='*80}\n")
    
    explanation = explain_policy(policy_path)
    print(explanation)
    print(f"\n{'='*80}\n")
