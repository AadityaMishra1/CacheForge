import sys, os
sys.path.append(os.path.abspath(".."))

import sqlite3
import subprocess
import re
from pathlib import Path
from collections import defaultdict
import json
from typing import Any, Dict, List, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


#----Do not change--------------------------------------------
WARMUP_INST         = "1000000"
SIM_INST            = "100000000"
SCORE               = "IPC"
LIB_PATH            = "../ChampSim_CRC2/lib/config2.a"
INCLUDE_DIR         = "../ChampSim_CRC2/inc"

NUM_INST_SETUP      = "100M_1M"
json_path           = f"../src/{NUM_INST_SETUP}/json"
workloads_json_path = f"{json_path}/workloads.json"
Policies_json_path  = f"{json_path}/policies.json"
Perf_json_path      = f"{json_path}/performance_data.json"
Score_json_path     = f"{json_path}/score.json"
perf_out_path       = f"../src/{NUM_INST_SETUP}/out"

empty_DB            = False
few_workloads       = False
few_workloads_list  =[]

BASELINE_STORAGE_KB = {
    "DRRIP": 8.0, #12.0,
    "SRRIP": 8.0, #8.0,
    "LRU": 16.0, #24.0,
    "Ship++": 20.0, #16.0,
    "Hawkeye": 31.8, #31.8,
    "Less is More": 31.2, #16, ###
    "Multiperspective": 29.75, #32, ###
    "Reordering-based Cache Replacement": 31.88, #32, ###
}
#-------------------------------------------------------------

#----You should set for your experiment ----------------------
ITERATIONS          = int(os.getenv("ITERATIONS", "10"))

_mem_default        = "false" if str(os.getenv("LLM_PROVIDER", "groq")).lower() == "groq" else "true"
memory_enable       = str(os.getenv("MEMORY_ENABLE", _mem_default)).lower() not in {"0", "false", "off", "no"}
max_core            = int(os.getenv("MAX_CORE", "4"))


experiment_name     = "experiment_000"
DB_PATH             = f"../DB/{experiment_name}.db"
OUT_DIR             = Path(f"../ChampSim_CRC2/new_policies/{experiment_name}")
MEMORY_PATH         = Path(f"../memory_json/{experiment_name}.json")

candidates_per_iter = int(os.getenv("CANDIDATES_PER_ITER", "1"))
surrogate_enable    = str(os.getenv("SURROGATE_ENABLE", "false")).lower() not in {"0", "false", "off", "no"}


LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "groq").lower()   # "openai", "ollama", or "groq"

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
OPENAI_BASE_URL     = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")  # Groq-compatible default
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_MEM_MODEL    = os.getenv("OPENAI_MEM_MODEL", "openai/gpt-oss-120b")
OPENAI_EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_MODEL       = "gpt-oss:20b"
OLLAMA_MEM_MODEL   = "gpt-oss:20b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

if LLM_PROVIDER == "openai":
    MODEL      = OPENAI_MODEL
    MEM_MODEL  = OPENAI_MEM_MODEL
elif LLM_PROVIDER == "groq":
    MODEL      = OPENAI_MODEL
    MEM_MODEL  = OPENAI_MEM_MODEL
else:
    MODEL      = OLLAMA_MODEL
    MEM_MODEL  = OLLAMA_MEM_MODEL

# Ollama mode: local or hpc
OLLAMA_MODE = "hpc"                 # "local" or "hpc"

HPC_OLLAMA_HOSTS = [
    "gpu16","gpu17","gpu18","gpu19","gpu20","gpu21","gpu22","gpu23",
    "gpu24","gpu25","gpu26","gpu27","gpu28","gpu29","gpu30","gpu31",
    "gpu32","gpu33","gpu34","gpu35","gpu36"
]

HPC_OLLAMA_PORTS = [11432, 11434, 11431, 11433]



LRU_FEWSHOT = """
## Policy Name
LRU

## Policy Description
A classic Least Recently Used policy for a 16-way LLC; each set maintains a total order among its 16 lines via a stack position (0=MRU, 15=LRU). On a hit, positions older than the hit line are incremented by one and the accessed line becomes MRU.

## C++ Implementation
```cpp
#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS (NUM_CORE*2048)
#define LLC_WAYS 16

static uint32_t lru[LLC_SETS][LLC_WAYS];

void InitReplacementState() {
    for (uint32_t s = 0; s < LLC_SETS; s++) {
        for (uint32_t w = 0; w < LLC_WAYS; w++) {
            lru[s][w] = w;
        }
    }
}

uint32_t GetVictimInSet(uint32_t cpu, uint32_t set, const BLOCK* current_set,
                        uint64_t PC, uint64_t paddr, uint32_t type) {
    // If any invalid way exists, return it immediately
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (!current_set[w].valid) return w;
    }
    // Return the way whose position is LRU (LLC_WAYS-1)
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (lru[set][w] == (LLC_WAYS - 1)) return w;
    }
    return 0; // fallback
}

void UpdateReplacementState(uint32_t cpu, uint32_t set, uint32_t way,
                            uint64_t paddr, uint64_t PC, uint64_t victim_addr,
                            uint32_t type, uint8_t hit) {
    uint32_t pos = lru[set][way];
    for (uint32_t w = 0; w < LLC_WAYS; w++) {
        if (lru[set][w] < pos) lru[set][w]++;
    }
    lru[set][way] = 0; // MRU
}

void PrintStats_Heartbeat() {}
void PrintStats() {}
```

## Minimum Storage Overhead
16.000

## Storage-Method + Storage-Breakdown
Per-line 4-bit LRU stack position (W=16); total_lines=LLC_SETS*LLC_WAYS; sum bits/8/1024.
name=lru_position, kind=per-line, count=32768, bits_each=4, total_bits=131072
"""

system_prompt = f"""You are an expert cache replacement policy designer for ChampSim CRC2 (LLC level only). Your job is to produce a replacement policy that maximizes average {SCORE} across the provided workloads.

Strict output format (must comply exactly):
- Output exactly five sections and nothing else, in this order:
1) ## Policy Name
2) ## Policy Description
3) ## C++ Implementation
- Under '## C++ Implementation' include exactly one fenced code block that starts with ```cpp and ends with ```.
4) ## Minimum Storage Overhead
- Put only one float number in KB with exactly 3 decimals (e.g., 16.000). No units, no extra text.
5) ## Storage-Method + Storage-Breakdown
- First line: a single-sentence method (no quotes).
- Next 1–4 lines: one per component in the form
    name=<id>, kind=per-line|per-set|global, count=<int>, bits_each=<int>, total_bits=<int>

Storage rules (information-theoretic):
- Report minimal metadata bits only (e.g., a 2-bit counter is 2 bits), never sizeof(C++ types).
- Unless you explicitly state different assumptions, use LLC_SETS=2048, LLC_WAYS=16, total_lines=32768.
- Do NOT print storage from C++; report it only in sections 4 and 5.

CRITICAL C++ Implementation Rules:
1. ALL state arrays MUST be declared as static globals BEFORE any functions.
2. NEVER declare state arrays inside functions (they won't be accessible elsewhere).
3. Example CORRECT structure:
   
   static uint8_t my_counter[LLC_SETS][LLC_WAYS];  // <-- GLOBAL DECLARATION
   
   void InitReplacementState() {{
       for (int s = 0; s < LLC_SETS; s++)
           for (int w = 0; w < LLC_WAYS; w++)
               my_counter[s][w] = 0;
   }}
   
   uint32_t GetVictimInSet(...) {{
       // Can now use my_counter here
       return 0;
   }}

4. Example WRONG structure (DO NOT DO THIS):
   
   void InitReplacementState() {{
       uint8_t my_counter[LLC_SETS][LLC_WAYS];  // <-- WRONG! Local variable
   }}
   
   uint32_t GetVictimInSet(...) {{
       // ERROR: my_counter doesn't exist here!
   }}

5. Use the template's marked section (after #define, before functions) for ALL state declarations.

Key policy requirements:
- Never bypass (return LLC_WAYS) on WRITEBACK type.
- Check current_set[w].valid for finding empty ways first.
- Use simple, proven techniques (RRIP, SHiP-style signatures, frequency counters).
- Keep code compact and avoid undefined behavior.
- Stay under 64 KiB total metadata budget.
"""


# ------------------------------------------------------------
#.                     Helper Functions
# ------------------------------------------------------------

#def default_score_func(ipc: float, total_hit_rate: float, metrics: Dict[str, float]) -> float:
#    sk = metrics.get("storage_kb")
#    return ipc / sk

def default_score_func(ipc: float, total_hit_rate: float, metrics: Dict[str, float]) -> float:
    if SCORE == "hit rate":
        return total_hit_rate
    else:
        return ipc


#def default_score_func(ipc: float, total_hit_rate: float, metrics: Dict[str, float]) -> float:
#    sk = metrics.get("storage_kb")
#    try:
#        if sk is None or sk <= 0:
#            return ipc   # fallback if no storage
#        return ipc / sk
#    except Exception:
#        return ipc


def ensure_experiments_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workload TEXT NOT NULL,
            policy TEXT NOT NULL,
            policy_description TEXT NOT NULL,
            cpp_file_path TEXT NOT NULL,
            cache_hit_rate REAL NOT NULL,
            load_hit_rate REAL NOT NULL,
            rfo_hit_rate REAL NOT NULL,
            prefetch_hit_rate REAL NOT NULL,
            writeback_hit_rate REAL NOT NULL,
            IPC REAL NOT NULL,
            score REAL NOT NULL,
            storage_kb REAL,
            storage_note TEXT
        )
    """)
    # Ensure uniqueness so we can upsert per (workload, policy) result
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_unique
        ON experiments(workload, policy)
    """)
    conn.commit()


def ensure_workloads_schema(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS workloads (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS workload_simpoints (
            workload TEXT NOT NULL,
            sp_index INTEGER NOT NULL,
            weight REAL NOT NULL,
            trace TEXT,
            PRIMARY KEY (workload, sp_index),
            FOREIGN KEY (workload) REFERENCES workloads(name) ON DELETE CASCADE
        )
    """)
    conn.commit()


def upsert_experiment_row(
        conn: sqlite3.Connection,
        *,
        workload: str,
        policy: str,
        policy_description: str,
        cpp_file_path: str,
        metrics: Dict[str, float],  # keys: 'ipc', 'total_hit_rate', 'load_hit_rate', 'rfo_hit_rate', 'prefetch_hit_rate', 'writeback_hit_rate'
        score_func: Callable[[float, float, Dict[str, float]], float] = default_score_func,
        storage_kb_override: Optional[float] = None,
        storage_note_override: Optional[str] = None,
) -> None:
    """
    Insert or replace one experiments row with computed score = score_func(ipc, total, metrics+storage).
    storage_kb is taken from override first, else BASELINE_STORAGE_KB[policy], else None.
    """
    c = conn.cursor()
    ipc   = float(metrics.get("ipc", 0.0))
    total = float(metrics.get("total_hit_rate", 0.0))
    load  = float(metrics.get("load_hit_rate", 0.0))
    rfo   = float(metrics.get("rfo_hit_rate", 0.0))
    pref  = float(metrics.get("prefetch_hit_rate", 0.0))
    wb    = float(metrics.get("writeback_hit_rate", 0.0))

    sk = storage_kb_override
    snote = storage_note_override

    score_metrics = {"load": load, "rfo": rfo, "prefetch": pref, "writeback": wb, "storage_kb": sk}
    score = float(score_func(ipc, total, score_metrics))

    sql = """
        INSERT OR REPLACE INTO experiments (
            workload, policy, policy_description, cpp_file_path,
            cache_hit_rate, load_hit_rate, rfo_hit_rate,
            prefetch_hit_rate, writeback_hit_rate, IPC, score,
            storage_kb, storage_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    c.execute(sql, (
        workload,
        policy,
        policy_description,
        cpp_file_path,
        total, load, rfo, pref, wb,
        ipc,
        score,
        sk,
        snote
    ))
    conn.commit()


def insert_performance_data(
    conn: sqlite3.Connection,
    performance_data: Dict[str, Dict[str, Dict[str, Any]]],
    policies: Dict[str, Dict[str, Any]],
    score_func: Callable[[float, float, Dict[str, float]], float] = default_score_func,
) -> None:

    for workload, per_policy in performance_data.items():
        for policy, result in per_policy.items():
            weighted = result["weighted"]
            metrics = {
                "ipc":               float(weighted.get("ipc", 0.0)),
                "total_hit_rate":    float(weighted.get("total_hit_rate", 0.0)),
                "load_hit_rate":     float(weighted.get("load_hit_rate", 0.0)),
                "rfo_hit_rate":      float(weighted.get("rfo_hit_rate", 0.0)),
                "prefetch_hit_rate": float(weighted.get("prefetch_hit_rate", 0.0)),
                "writeback_hit_rate":float(weighted.get("writeback_hit_rate", 0.0)),
            }

            pinfo = policies.get(policy, {})
            pdesc = pinfo.get("description", "")
            pfile = str(pinfo.get("file_path", ""))

            sk_override = BASELINE_STORAGE_KB.get(policy)
            s_note_override = "Baseline Policy"

            upsert_experiment_row(
                conn,
                workload=workload,
                policy=policy,
                policy_description=pdesc,
                cpp_file_path=pfile,
                metrics=metrics,
                score_func=score_func,
                storage_kb_override=sk_override,
                storage_note_override=s_note_override,
            )


def insert_workloads(conn: sqlite3.Connection, workloads: Dict[str, Dict[str, Any]]) -> None:
    # ------------------------------------------------------------
    # Insert workloads dict (name -> description, simpoints)
    # workloads[name] = {
    #   "description": str,
    #   "simpoints": [ {"index": int, "weight": float, "trace": optional str}, ... ]
    # }
    # ------------------------------------------------------------

    c = conn.cursor()
    sql_w = """
        INSERT OR REPLACE INTO workloads (name, description) VALUES (?, ?)
    """
    sql_sp = """
        INSERT OR REPLACE INTO workload_simpoints (workload, sp_index, weight, trace)
        VALUES (?, ?, ?, ?)
    """
    for name, wdata in workloads.items():
        desc = wdata.get("description", "")
        c.execute(sql_w, (name, desc))

        simpoints = wdata.get("simpoints", [])
        if simpoints:
            rows = []
            for sp in simpoints:
                sp_idx = int(sp["index"])
                weight = float(sp.get("weight", 0.0))
                trace = sp.get("trace")
                rows.append((name, sp_idx, weight, trace))
            c.executemany(sql_sp, rows)

    conn.commit()


def insert_scores(
    conn: sqlite3.Connection,
    loaded_score: Dict[str, Dict[str, float]],
    policies: Dict[str, Dict[str, Any]],
    *,
    score_func: Callable[[float, float, Dict[str, float]], float] = default_score_func,
    workload_label: str = "ALL",
) -> None:

    for policy, metrics in loaded_score.items():
        metrics = {
            "ipc":               float(metrics.get("ipc", 0.0)),
            "total_hit_rate":    float(metrics.get("total_hit_rate", 0.0)),
            "load_hit_rate":     float(metrics.get("load_hit_rate", 0.0)),
            "rfo_hit_rate":      float(metrics.get("rfo_hit_rate", 0.0)),
            "prefetch_hit_rate": float(metrics.get("prefetch_hit_rate", 0.0)),
            "writeback_hit_rate":float(metrics.get("writeback_hit_rate", 0.0)),
        }

        pinfo = policies.get(policy, {})
        pdesc = pinfo.get("description", "")
        pfile = str(pinfo.get("file_path", ""))

        sk_override = BASELINE_STORAGE_KB.get(policy)
        s_note_override = "Baseline Policy"

        upsert_experiment_row(
            conn,
            workload=workload_label,
            policy=policy,
            policy_description=pdesc,
            cpp_file_path=pfile,
            metrics=metrics,
            score_func=score_func,
            storage_kb_override=sk_override,
            storage_note_override=s_note_override,
        )


def compile_policy(src: Path, OUT_DIR: Path) -> Path:
    """g++ <src> -> <src>.out  (always recompiles)"""
    exe = OUT_DIR / (src.stem + ".out")
    subprocess.run(
        [
            "g++",
            "-Wall",
            "-std=c++17",
            f"-I{INCLUDE_DIR}",
            str(src),
            LIB_PATH,
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe


def run_policy(exe: Path, trace_path: Path,
               warmup_instructions: int = 0,
               simulation_instructions: int = 10**15) -> str:
    """
    Runs ChampSim. By default, runs the whole trace
    with warmup_instructions=0 and a huge simulation budget.
    """
    cmd = [
        str(exe),
        "-warmup_instructions", str(warmup_instructions),
        "-simulation_instructions", str(simulation_instructions),
        "-traces", str(trace_path),
    ]
    out = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return out.stdout


def load_workloads_file(json_path: str | Path) -> Dict[str, Any]:
    """
    Load workloads_simpoints.json and return the full parsed dictionary.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "workloads" not in data or not isinstance(data["workloads"], dict):
        raise ValueError("Invalid JSON: missing 'workloads' object")
    return data


def get_workloads_dict(json_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load workloads_simpoints.json and return only the workloads sub-dict,
    keyed by workload name.

    Returned structure example:
      {
        "astar": {
          "description": "...",
          "simpoints": [
            {"weight": 0.45, "index": 163, "trace": "ChampSim_CRC2/.../astar_163B.trace.gz"},
            ...
          ]
        },
        ...
      }
    """
    data = load_workloads_file(json_path)
    return data["workloads"]


def get_workload_names(json_path: str | Path) -> List[str]:
    """
    Return a list of workload names present in the JSON (sorted).
    """
    workloads = get_workloads_dict(json_path)
    return sorted(workloads.keys())


def get_simpoints_for_workload(
    workloads: Dict[str, Dict[str, Any]],
    name: str,
    sort_by_weight_desc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return the simpoints list for a single workload.

    - If sort_by_weight_desc is True, sort by 'weight' descending.
    - If a simpoint lacks 'trace', it will remain without that key.
    """
    if name not in workloads:
        raise KeyError(f"Workload '{name}' not found")
    simpoints = list(workloads[name].get("simpoints", []))
    if sort_by_weight_desc:
        simpoints.sort(key=lambda sp: sp.get("weight", 0.0), reverse=True)
    return simpoints


def build_simpoint_trace_map(
    workloads: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[int, Optional[str]]]:
    """
    Build a nested dict mapping:
      workload_name -> { simpoint_index: trace_path or None }

    This makes it easy to look up a trace for a specific simpoint index.
    """
    out: Dict[str, Dict[int, Optional[str]]] = {}
    for wname, wentry in workloads.items():
        mapping: Dict[int, Optional[str]] = {}
        for sp in wentry.get("simpoints", []):
            idx = int(sp.get("index"))
            trace = sp.get("trace")  # may be absent
            mapping[idx] = trace
        out[wname] = mapping
    return out


def normalize_simpoint_weights(simpoints: list[dict]) -> list[dict]:
    """
    Keep only simpoints that have a 'trace', and normalize their weights to sum to 1.
    Adds a '_norm_weight' field to each kept simpoint.
    """
    usable = [sp for sp in simpoints if sp.get("trace")]
    total_w = sum(sp.get("weight", 0.0) for sp in usable)
    if total_w > 0:
        for sp in usable:
            sp["_norm_weight"] = sp["weight"] / total_w
    else:
        for sp in usable:
            sp["_norm_weight"] = 0.0
    return usable


def parse_ipc_and_hit_rates(output_text: str) -> Dict[str, float]:
    """
    Extract:
      - ipc
      - total_hit_rate
      - load_hit_rate
      - rfo_hit_rate
      - prefetch_hit_rate
      - writeback_hit_rate

    If a category is missing or has zero ACCESS, its hit rate is 0.0.
    """
    # Last IPC seen (handles variants like "CPU 0 cumulative IPC: ...")
    ipc_matches = re.findall(r"\bIPC\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", output_text, flags=re.IGNORECASE)
    ipc = float(ipc_matches[-1]) if ipc_matches else 0.0

    # LLC category lines (TOTAL, LOAD, RFO, PREFETCH, WRITEBACK)
    llc_matches = re.findall(
        r"^\s*LLC\s+([A-Z]+)\s+ACCESS\s*:\s*([0-9]+)\s+HIT\s*:\s*([0-9]+)",
        output_text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    cats = {}
    for cat, access, hit in llc_matches:
        cat = cat.upper()
        a = int(access)
        h = int(hit)
        cats[cat] = (a, h)

    def hr(cat_name: str) -> float:
        a, h = cats.get(cat_name, (0, 0))
        return (h / a) if a else 0.0

    # TOTAL: prefer reported TOTAL; else sum all categories we saw
    if "TOTAL" in cats:
        total_hit_rate = hr("TOTAL")
    else:
        total_access = sum(a for a, _ in cats.values())
        total_hit = sum(h for _, h in cats.values())
        total_hit_rate = (total_hit / total_access) if total_access else 0.0

    return {
        "ipc": ipc,
        "total_hit_rate": total_hit_rate,
        "load_hit_rate": hr("LOAD"),
        "rfo_hit_rate": hr("RFO"),
        "prefetch_hit_rate": hr("PREFETCH"),
        "writeback_hit_rate": hr("WRITEBACK"),
    }

def detect_max_workers(reserve=1) -> int:
    alloc = None
    v = os.getenv("LSB_DJOB_NUMPROC")
    if v and v.isdigit():
        alloc = int(v)
    if alloc is None:
        s = os.getenv("LSB_MCPU_HOSTS")
        if s:
            toks = s.split()
            try:
                alloc = sum(int(toks[i+1]) for i in range(0, len(toks), 2))
            except Exception:
                pass
    if alloc is None:
        v = os.getenv("SLURM_CPUS_ON_NODE")
        if v and v.isdigit():
            alloc = int(v)
    if alloc is None:
        v = os.getenv("SLURM_NTASKS")
        if v and v.isdigit():
            alloc = int(v)
    if alloc is None:
        cpt = os.getenv("SLURM_CPUS_PER_TASK")
        nt = os.getenv("SLURM_NTASKS")
        if cpt and nt and cpt.isdigit() and nt.isdigit():
            alloc = int(cpt) * int(nt)
    if alloc is None:
        v = os.getenv("PBS_NP") or os.getenv("NSLOTS")
        if v and v.isdigit():
            alloc = int(v)
    if alloc is None:
        alloc = os.cpu_count() or 1
    return max(1, alloc - reserve)

# Set once at import:
max_core = detect_max_workers()
print(f"[common] Scheduler alloc={max_core+1 if max_core else 'n/a'} (after reserve), using max_core={max_core}")
