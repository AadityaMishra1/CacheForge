import sys, os
sys.path.append(os.path.abspath("../src"))

# --- Limit all threadpools (BLAS, OpenMP, NumExpr, etc.) ---
for var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[var] = "1"

import os
import re
import time
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256

from llm_provider import LLMProvider
from RAG import ExperimentRAG
from PromptGeneratorReduced import PolicyPromptGenerator
from LLMMemory import HybridMemory
from policy_explainer import explain_policy_brief
from policy_explainer import explain_policy_structured
from surrogate import SurrogateManager
import common

import multiprocessing as mp

# ------------------------------------------------------------
# CoT helper
# ------------------------------------------------------------
def generate_policy_with_cot(llm: LLMProvider,
                             system_prompt: str,
                             user_prompt: str,
                             model: str,
                             temperature: float = 0.8) -> str:
    """
    Two-step generation:
      1) Architect: brainstorm 3–4 candidates, pick the best with a concise rationale.
      2) Engineer: emit ONLY the final 5-section policy using the chosen design.
    """
    # Step 1: Architect (higher temperature for diversity)
    idea_prompt = (
        "Act as the architect. Brainstorm 3–4 distinct LLC replacement policy designs "
        "for the given context, then pick ONE best design. Respond with:\n"
        "- Chosen name\n- Core mechanism (2-3 bullets)\n- Why it should win (1-2 sentences)\n\n"
        "Keep it under 120 words. Do NOT output code or the final sections yet.\n\n"
        f"Context:\n{user_prompt}"
    )
    idea_resp = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": idea_prompt}
        ],
        model=model,
        temperature=min(1.1, temperature + 0.2),
        num_ctx=8192,
        max_tokens=600
    )

    # Step 2: Engineer (deterministic, produces final format)
    print("        ↳ CoT Step 1: Architect reasoning received")
    engineer_prompt = (
        "Act as the engineer. Using ONLY the chosen design below, produce the FINAL answer "
        "in the strict 5-section format. Do NOT include the brainstorm. "
        "Honor all formatting constraints and the code template already provided in the context.\n\n"
        "Chosen design:\n"
        f"{idea_resp}\n\n"
        "Now output the final 5 sections."
    )
    final_resp = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\n" + engineer_prompt}
        ],
        model=model,
        temperature=max(0.25, temperature - 0.25),
        num_ctx=8192,
        max_tokens=8000
    )
    print("        ↳ CoT Step 2: Engineer output received")
    return final_resp

# --- Limit number of processes to common.max_core ---
MAX_WORKERS = common.max_core

# Optional: use 'spawn' to avoid inheriting threads from parent
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# Emergency safeguard (disabled): allow LLM-generated candidates
FORCE_BASELINE_CODE = False
BASELINE_SRC = Path("../ChampSim_CRC2/new_policies/real_data/hawk_dual_hawkeye_fuse.cc")

# Fixed baselines for fallback selection (per-workload max of Hawkeye and FUSE)
HAWKEYE_BASELINE_SRC = Path("../ChampSim_CRC2/new_policies/real_data/_hawkeye_baseline.cc")
FUSE_BASELINE_SRC    = Path("../ChampSim_CRC2/new_policies/real_data/007_fuse_runguard__v0.cc")

workloads = common.get_workloads_dict(common.workloads_json_path)

if getattr(common, "few_workloads", False):
    keep = getattr(common, "few_workloads_list", [])
    if isinstance(keep, str):
        keep = [x.strip() for x in re.split(r"[,\\s]+", keep) if x.strip()]
    if keep:
        workloads = {k: v for k, v in workloads.items() if k in keep}

workload_name = "\n".join(sorted(workloads.keys()))

def _read_safe(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return "// [code unavailable]"


def _sp_task(args):
    exe_path_str, workload_name, sp_index, sp_weight, trace_path_str, warmup, sim = args
    try:
        out = common.run_policy(Path(exe_path_str), trace_path=Path(trace_path_str),
                         warmup_instructions=int(warmup), simulation_instructions=int(sim))
        metrics = common.parse_ipc_and_hit_rates(out)  # from common.py
        return {
            "ok": True,
            "workload": workload_name,
            "index": sp_index,
            "weight": sp_weight,
            "trace": trace_path_str,
            "metrics": metrics
        }
    except Exception as e:
        return {
            "ok": False,
            "workload": workload_name,
            "index": sp_index,
            "weight": sp_weight,
            "trace": trace_path_str,
            "error": str(e)
        }


def _build_sp_tasks_for_policy(exe: Path, workloads_dict: Dict[str, Dict[str, Any]],
                               warmup: int, sim: int) -> List[tuple]:
    """
    Build tasks for ALL simpoints that have a non-null 'trace'.
    This satisfies: 'run all checkpoints that have a trace address'.
    """
    tasks = []
    for wname, wdata in workloads_dict.items():
        simpoints = common.normalize_simpoint_weights(wdata.get("simpoints", []))  # filters to those with 'trace' and normalizes
        if not simpoints:
            continue
        for sp in simpoints:
            tasks.append((
                str(exe),                # exe path
                wname,                   # workload name
                int(sp["index"]),        # simpoint index
                float(sp["_norm_weight"]),
                str(sp["trace"]),        # trace path
                warmup,
                sim
            ))
    return tasks


# ------------------------------------------------------------
# Baseline compilation and measurement (Hawkeye + FUSE-RunGuard)
# ------------------------------------------------------------
def _aggregate_policy_results(exe: Path, workloads: Dict[str, Dict[str, Any]], warmup: int, sim: int) -> Dict[str, Dict[str, float]]:
    tasks = _build_sp_tasks_for_policy(exe, workloads, warmup=warmup, sim=sim)
    accum: Dict[str, Dict[str, float]] = {}
    if not tasks:
        return {}
    for t in tasks:
        res = _sp_task(t)
        if not res.get("ok"):
            continue
        wl = res["workload"]
        w  = float(res["weight"])
        m  = res["metrics"]
        if wl not in accum:
            accum[wl] = {
                "ipc_h_inv": 0.0, "total_hit_rate": 0.0, "load_hit_rate": 0.0,
                "rfo_hit_rate": 0.0, "prefetch_hit_rate": 0.0, "writeback_hit_rate": 0.0
            }
        eps = 1e-12
        accum[wl]["ipc_h_inv"]          += w / max(eps, float(m["ipc"]))
        accum[wl]["total_hit_rate"]     += w * float(m["total_hit_rate"])
        accum[wl]["load_hit_rate"]      += w * float(m["load_hit_rate"])
        accum[wl]["rfo_hit_rate"]       += w * float(m["rfo_hit_rate"])
        accum[wl]["prefetch_hit_rate"]  += w * float(m["prefetch_hit_rate"])
        accum[wl]["writeback_hit_rate"] += w * float(m["writeback_hit_rate"])

    out: Dict[str, Dict[str, float]] = {}
    for wl, vals in accum.items():
        ipc = (1.0 / vals["ipc_h_inv"]) if vals["ipc_h_inv"] > 0 else 0.0
        out[wl] = {
            "ipc": ipc,
            "total_hit_rate": vals["total_hit_rate"],
            "load_hit_rate": vals["load_hit_rate"],
            "rfo_hit_rate": vals["rfo_hit_rate"],
            "prefetch_hit_rate": vals["prefetch_hit_rate"],
            "writeback_hit_rate": vals["writeback_hit_rate"],
        }
    return out

def _load_baseline_metrics_from_db(conn: sqlite3.Connection, policy_name: str) -> Dict[str, Dict[str, float]]:
    """Load per-workload metrics for a given policy from the experiments table."""
    out: Dict[str, Dict[str, float]] = {}
    try:
        cur = conn.execute(
            "SELECT workload, IPC, cache_hit_rate, load_hit_rate, rfo_hit_rate, prefetch_hit_rate, writeback_hit_rate "
            "FROM experiments WHERE policy=?",
            (policy_name,)
        )
        for row in cur.fetchall():
            wl = row[0]
            out[wl] = {
                "ipc": float(row[1]),
                "total_hit_rate": float(row[2]),   # maps cache_hit_rate -> total_hit_rate
                "load_hit_rate": float(row[3]),
                "rfo_hit_rate": float(row[4]),
                "prefetch_hit_rate": float(row[5]),
                "writeback_hit_rate": float(row[6]),
            }
    except sqlite3.Error as e:
        print(f"      [!] Baseline load failed for {policy_name}: {e}")
    return out

HAWKEYE_BASE_METRICS: Dict[str, Dict[str, float]] = {}
FUSE_BASE_METRICS: Dict[str, Dict[str, float]] = {}


def code_hash(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def sanitize(name: str) -> str:
    print("     3. 🔧 [Sanitize] Cleaning policy name\n")
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _render_surrogate_feedback(best_score: float, threshold: float, pred_history: list[float]) -> str:
    """
    Provide lightweight surrogate guidance to the LLM so it is not designing blindly.
    """
    recent = pred_history[-5:]
    preds = ", ".join(f"{p:.3f}" for p in recent) if recent else "none yet"
    return (
        f"Surrogate gate: target IPC >= {threshold:.3f} (current best={best_score:.3f}). "
        f"Recent predicted IPCs: {preds}. "
        "Key gaps: lbm (need stronger stream/scan bypass) and zeusmp (allow short reuse). "
        "Please: (1) Use a stride+1/+2 run detector; after 2 forward hits, bypass or hard-tail insert; quarantine prefetches. "
        "(2) Multi-hit promotion gate (2–3 demand hits) so stream-tagged lines can be promoted once reuse appears—helps zeusmp. "
        "(3) PC coldness/dead predictor to deprioritize noisy PCs (gcc/omnetpp). "
        "Expose tunables: stream window, forward-run length, bypass probability, hits_to_promote, prefetch quarantine depth, demotion rate."
    )

def _append_log_entry(path: Path, entry: Dict[str, Any]) -> None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8") or "[]")
            if not isinstance(data, list):
                data = []
        else:
            data = []
    except Exception:
        data = []
    data.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_policy_content(text: str,) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _extract(pattern: str):
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    name = _extract(r"##\s*Policy\s*Name\s*\n(.*?)\n")
    desc = _extract(r"##\s*Policy\s*Description\s*\n(.*?)\n")
    code = _extract(r"```cpp\s*(.*?)\s*```")
    return name, desc, code


def parse_policy_sections(text: str):
    hdr_re = re.compile(r"^##\s*(.+?)\s*$", re.MULTILINE)
    matches = list(hdr_re.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[title.lower()] = text[start:end].strip()

    name = sections.get("policy name", "")
    desc = sections.get("policy description", "")
    impl = sections.get("c++ implementation", "")
    code_match = re.search(r"```cpp\s*(.*?)\s*```", impl, flags=re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else None

    sto = sections.get("minimum storage overhead", "")
    storage_kb = None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", sto)
    if m:
        try:
            storage_kb = float(m.group(1))
        except Exception:
            storage_kb = None

    smbd = sections.get("storage-method + storage-breakdown", "")
    smbd_lines = [ln.strip() for ln in smbd.splitlines() if ln.strip()]
    storage_method = smbd_lines[0] if smbd_lines else ""
    storage_breakdown = "\n".join(smbd_lines[1:]) if len(smbd_lines) > 1 else ""
    storage_note = (storage_method + ("\n" + storage_breakdown if storage_breakdown else "")).strip() or None

    if not name or not desc or not code:
        return None, None, None, None, None
    return name, desc, code, storage_kb, storage_note


def _summarize_diagnostics(stderr: str, max_head: int = 10, max_tail: int = 5, max_errors: int = 5) -> dict:
    lines = stderr.splitlines()
    head = "\n".join(lines[:max_head])
    tail = "\n".join(lines[-max_tail:]) if lines else ""
    errors_only = [l for l in lines if ("error:" in l) or ("undefined reference" in l)]
    errors_snippet = "\n".join(errors_only[:max_errors]) if errors_only else ""
    kind = "compile_error"
    if any("undefined reference" in l for l in lines) or "ld returned 1 exit status" in stderr:
        kind = "link_error"
    return {"kind": kind, "errors": errors_snippet, "head": head, "tail": tail}


def compile_policy(cc: Path) -> Path:
    print(f"     4. 🔨 [Compile] Compiling: {cc.name}\n")
    exe = cc.with_suffix(".out")
    res = subprocess.run(
        [
            "g++",
            "-Wall",
            "-std=c++17",
            "-fno-diagnostics-color",
            "-fmax-errors=5",
            f"-I{common.INCLUDE_DIR}",
            str(cc),
            common.LIB_PATH,
            "-o",
            str(exe),
        ],
        text=True,
        capture_output=True
    )
    if res.returncode != 0:
        diag = _summarize_diagnostics(res.stderr)
        log_dir = common.OUT_DIR / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        stamp = int(time.time())
        (log_dir / f"{cc.stem}.{stamp}.stderr.log").write_text(res.stderr, encoding="utf-8")
        (log_dir / f"{cc.stem}.{stamp}.stdout.log").write_text(res.stdout or "", encoding="utf-8")
        compact = diag["errors"] or (diag["head"] + ("\n...\n" + diag["tail"] if diag["tail"] else ""))
        raise RuntimeError(f"{diag['kind']} for {cc.name}:\n{compact[:2000]}")
    return exe


def build_explore_prompt(
    rag: ExperimentRAG,
    workload_names: str,
    code_template: str,
    top_k: int = 3,
    include_impl: bool = True,
    max_code_chars: int = 6000
) -> str:
    rows = rag.get_top_by_metric("ALL", metric=common.SCORE, top_n=top_k)
    blocks = []
    # Pull in DB baselines
    for r in rows:
        impl_block = ""
        if include_impl:
            code = _read_safe(r["cpp_file_path"])
            if max_code_chars and len(code) > max_code_chars:
                code = code[:max_code_chars] + "\n// [truncated for prompt size]\n"
            impl_block = f"Implementation:\n```cpp\n{code}\n```\n"

        blocks.append(
            f"## Baseline Policy\n"
            f"Name: {r['policy']}\n"
            f"Description:\n{r['policy_description']}\n\n"
            f"Results (average across all workloads): IPC={r['ipc']:.6f}, TOTAL={r['total_hit_rate']:.6f}\n"
            f"{impl_block}"
        )

    # Removed extra baseline injection to prevent LLM drift

    header = (
        "The following SPEC2006 workloads are under consideration:\n"
        f"{workload_names}\n\n"
        "DYNAMIC ENSEMBLE MODE: Mode A is fixed to Hawkeye baseline (IPC≈0.413081). Only evolve Mode B and selector knobs to complement Hawkeye.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Do NOT modify Mode A logic or state (Hawkeye OPTgen/SHCT).\n"
        "- Mode B may be changed/tuned; keep per-line state separate between modes.\n"
        "- Suggested selector defaults: MODE_A_BIAS=40, MODE_B_THRESHOLD=5, MAX_MODE_B_FOLLOWERS=512 (conservative gating).\n"
        "- Follow the strict 5-section output format with one ```cpp block.\n"
        "- Policy name should reflect the ensemble (Hawkeye + <Mode B>).\n\n"
        "BASELINE GUARANTEE: The ensemble must never underperform Hawkeye; pivot to Mode A when Mode B loses.\n\n"
    )
    template = (
        f"C++ implementation template to fill:\n{code_template}\n\n"
    )
    return header + "\n".join(blocks) + "\n" + template


def build_exploit_prompt(prev_name: str,
                         prev_desc: str,
                         prev_code: str,
                         lineage_block: str,
                         code_template: str,
                         feedback: str = "",
                         include_prev_code: bool = True,
                         max_code_chars: int = 6000) -> str:
    header = (
        "You are in EXPLOITATION mode.\n"
        "- Do NOT create a brand-new policy; refine the previous design ONLY.\n"
        "- Mode A is fixed to Hawkeye baseline. Do NOT change Mode A logic/state. Only tune Mode B and selector knobs.\n"
        "- Suggested selector defaults: MODE_A_BIAS=40, MODE_B_THRESHOLD=5, MAX_MODE_B_FOLLOWERS=512 (Mode B ENABLED with strict gating).\n"
        "- Keep per-line state exclusive per mode; UpdateReplacementState must choose exactly one path.\n"
        "- Follow the strict 5-section output format; include exactly one ```cpp code block in section 3.\n\n"
        "- HARD BUDGET: Fit within ~32 KB. Target ≤4 bytes per line (pack 3-bit RRIP/age/status/PC sig/flags) and keep PC tables tiny. Do NOT add 64-bit per-line fields or duplicate metadata arrays.\n\n"
    )
    fb_block = f"Feedback from last run:\n{feedback}\n\n" if feedback else ""
    prev_code_block = ""
    if include_prev_code:
        code_to_show = prev_code[:max_code_chars] + ("\n// [truncated]\n" if len(prev_code) > max_code_chars else "")
        prev_code_block = f"Previous iteration code:\n```cpp\n{code_to_show}\n```\n\n"

    prev = (
        f"Previous policy (to improve): {prev_name}\n"
        f"Description:\n{prev_desc}\n\n"
        f"{prev_code_block}"
    )
    lineage = (lineage_block + "\n\n") if lineage_block else ""
    template = f"C++ implementation template reminder:\n{code_template}\n"
    return header + fb_block + lineage + prev + template


# ──────────────────────────────────────────────────────────────────────────────
# Main Feedback Loop with Memory
# ──────────────────────────────────────────────────────────────────────────────
def main():

    rag = ExperimentRAG(common.DB_PATH)
    conn = sqlite3.connect(common.DB_PATH)
    common.ensure_experiments_schema(conn)
    common.ensure_workloads_schema(conn)

    # Seed workloads table for new experiments/DBs
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM workloads")
    if c.fetchone()[0] == 0:
        workloads_dict = common.get_workloads_dict(common.workloads_json_path)
        c.execute("DELETE FROM workload_simpoints")
        c.execute("DELETE FROM workloads")
        for wname, wdata in workloads_dict.items():
            desc = wdata.get("description", "")
            c.execute("INSERT OR REPLACE INTO workloads(name, description) VALUES (?, ?)", (wname, desc))
            for sp in wdata.get("simpoints", []):
                idx = int(sp.get("index"))
                wt = float(sp.get("weight", 0.0))
                trace = sp.get("trace")
                c.execute(
                    "INSERT OR REPLACE INTO workload_simpoints(workload, sp_index, weight, trace) VALUES (?, ?, ?, ?)",
                    (wname, idx, wt, trace)
            )
        conn.commit()

    # Load baseline metrics from DB (Hawkeye seed and FUSE-RunGuard)
    global HAWKEYE_BASE_METRICS, FUSE_BASE_METRICS
    HAWKEYE_BASE_METRICS = _load_baseline_metrics_from_db(conn, "Hawkeye-CRC2 seed")
    FUSE_BASE_METRICS    = _load_baseline_metrics_from_db(conn, "FUSE-RunGuard")
    if not HAWKEYE_BASE_METRICS or not FUSE_BASE_METRICS:
        print("      [!] Baseline metrics missing in DB; fallback dicts may be empty.")

    prompt_gen = PolicyPromptGenerator(common.DB_PATH)
    llm = LLMProvider(
        provider=common.LLM_PROVIDER,
        openai_api_key=common.OPENAI_API_KEY,
        host=(common.OLLAMA_HOST if common.OLLAMA_MODE == "local" else None),
        mode=(common.OLLAMA_MODE if common.LLM_PROVIDER == "ollama" else "local"),
        hpc_hosts=(common.HPC_OLLAMA_HOSTS if common.OLLAMA_MODE == "hpc" else None),
        hpc_ports=(common.HPC_OLLAMA_PORTS if common.OLLAMA_MODE == "hpc" else None),
        default_model=common.MODEL,
        default_mem_model=common.MEM_MODEL,
        default_embed_model=(common.OPENAI_EMBED_MODEL if common.LLM_PROVIDER=="openai" else common.OLLAMA_EMBED_MODEL),
    )

    mem_enabled = bool(getattr(common, "memory_enable", True))
    memory = None
    if mem_enabled:
        memory = HybridMemory(
            llm=llm,
            model=common.MODEL,
            memory_path=common.MEMORY_PATH,
            max_tokens=7000,
            ephemeral_max_tokens=1600,
            summary_max_tokens=1200,
            compress_threshold_tokens=900,
            retrieve_k=5,
            use_embeddings=True,
            memory_model=common.MEM_MODEL,
        )

    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if mem_enabled:
        common.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    surrogate_mgr = None
    surrogate_log_path = common.MEMORY_PATH.with_suffix(".surrogate_log.json")
    surrogate_enabled = bool(getattr(common, "surrogate_enable", False))
    if surrogate_enabled:
        surrogate_mgr = SurrogateManager(
            log_path=surrogate_log_path,
            model_path=common.MEMORY_PATH.with_suffix(".surrogate.joblib"),
            min_samples=getattr(common, "surrogate_min_samples", 6),
            retrain_interval=getattr(common, "surrogate_retrain_interval", 3),
        )
        surrogate_mgr.maybe_retrain(force=True)

    top_policies = rag.get_top_by_metric("ALL", metric=common.SCORE, top_n=5)
    # Seed best_score from known Hawkeye baseline to avoid inflated prior runs
    HAWKEYE_BASELINE_IPC = 0.413081
    # Use IPC (harmonic mean row "ALL") as baseline, but never below Hawkeye baseline
    best_score = max(HAWKEYE_BASELINE_IPC, float(top_policies[0].get("ipc", 0.0)) if top_policies else 0.0)
  
    print(f"     📈 [Init] Starting best {common.SCORE}: {best_score:.6f}")

    prev_name = prev_desc = prev_code = None
    current_score = best_score
    surrogate_pred_history: list[float] = []
    last_score = best_score

    code_template = prompt_gen._get_code_template()
    lineage_id = None
    exploitation_budget = 0
    exploration = True
    seen_hashes = set() 
    seen_structs: list[dict[str, Any]] = []
    ver = 0
    i=0
    while True:
        ver = 0 if exploration else ver + 1
        if i >= common.ITERATIONS:
            break

        if i == 0 or exploration:
            # EXPLORATION: show baselines (no code for Ollama), do not use lineage_block here
            no_baselines = not rag.get_top_by_metric("ALL", metric=common.SCORE, top_n=1)
            gate_threshold = max(
                getattr(common, "surrogate_min_ipc", 0.0) or 0.0,
                getattr(common, "surrogate_skip_threshold", 0.0) * best_score if best_score > 0 else 0.0,
                getattr(common, "surrogate_skip_threshold_abs", 0.0) or 0.0,
            )
            surrogate_hint = _render_surrogate_feedback(best_score, gate_threshold, surrogate_pred_history)
            if no_baselines:
                prompt = (
                    "No baseline policy data is available yet. You are in EXPLORATION mode.\n"
                    f"Workloads:\n{workload_name}\n\n"
                    "Tune Mode B and selector knobs while keeping Mode A fixed to Hawkeye (baseline).\n"
                    "Strict 5-section output with exactly one ```cpp block in section 3.\n\n"
                    f"C++ template:\n{code_template}\n\n"
                    f"Surrogate guidance:\n{surrogate_hint}\n\n"
                )
            else:
                prompt = build_explore_prompt(
                    rag,
                    workload_name,
                    code_template,
                    top_k=2
                ) + f"\nSurrogate guidance:\n{surrogate_hint}\n"

            messages = [
                {"role": "system", "content": common.system_prompt},
                {"role": "user", "content": prompt}
            ]
            used_memory = False

        else:
            # EXPLOITATION: refine previous design; lineage is used here
            if current_score > prev_best:
                feedback = (
                    f"Great! Policy improved average {common.SCORE} from {prev_best:.6f} to {current_score:.6f}. "
                    f"Please refine further—retain winning ideas and simplify if possible."
                )
            else:
                feedback = (
                    f"Policy average {common.SCORE} was {current_score:.6f}, not better than {prev_best:.6f}. "
                    f"Try parameter retuning, insertion/bypass tweaks, and reducing noise."
                )

            gate_threshold = max(
                getattr(common, "surrogate_min_ipc", 0.0) or 0.0,
                getattr(common, "surrogate_skip_threshold", 0.0) * best_score if best_score > 0 else 0.0,
                getattr(common, "surrogate_skip_threshold_abs", 0.0) or 0.0,
            )
            surrogate_hint = _render_surrogate_feedback(best_score, gate_threshold, surrogate_pred_history)
            lineage_block = memory.render_lineage_block(lineage_id, k=6) if (mem_enabled and memory and lineage_id) else ""
            prompt = build_exploit_prompt(
                prev_name or "N/A",
                prev_desc or "N/A",
                prev_code or "// N/A",
                lineage_block,
                code_template,
                feedback=feedback + f"\n\nSurrogate guidance:\n{surrogate_hint}"
            )

            if mem_enabled and memory is not None:
                used_memory = True
                memory.add_event("feedback", feedback, severity="info")
                messages = memory.build_messages(system_prompt=common.system_prompt, user_prompt=prompt, retrieve_k=5)
            else:
                messages = [
                    {"role": "system", "content": common.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                used_memory = False
            


        if exploration:
            iter_mode = "exploration (no-memory)"
        else:
            iter_mode = "exploitation (with-memory)" if used_memory else "exploitation (no-memory)"

        # FORCE BASELINE FOR ITERATION 0: Bypass LLM entirely
        if i == 0:
            print(f"     🔒 [FORCED BASELINE] Iteration 0: Loading verbatim template (bypassing LLM)")
            baseline_template_path = Path("../ChampSim_CRC2/new_policies/real_data/_hawkeye_baseline.cc")
            if baseline_template_path.exists():
                code = baseline_template_path.read_text()
                name = "Hawkeye Baseline (Mode A seed)"
                desc = (
                    "Pure Hawkeye replacement baseline (OPTgen + SHCT). "
                    "Used as Mode A seed for the ensemble; Mode B disabled."
                )
                storage_kb_est = 32.0
                storage_note = "Mode A (Hawkeye baseline)."

                explainer_struct = explain_policy_structured(code)
                candidate_pool = [{
                    "name": name,
                    "desc": desc,
                    "code": code,
                    "storage_kb_est": storage_kb_est,
                    "storage_note": storage_note,
                    "explainer": explainer_struct,
                    "pred_ipc": None,
                    "diversity": 0.0,
                    "score": 0.0,
                    "version": 0
                }]
                print(f"     ✓ [FORCED BASELINE] Template loaded: {len(code)} chars")
                # Skip LLM generation entirely for iteration 0
            else:
                print(f"     ✗ [ERROR] Template not found, falling back to LLM")

        if i == 0 and baseline_template_path.exists():
            # Skip LLM generation for iteration 0 - template already loaded
            pass
        else:
            # Normal LLM generation for iteration 1+
            print(f"     1. 📤 [LLM] Iteration {i}: Sending prompt to model - {iter_mode}")
            temp = 1.25 if exploration else 0.75
            candidate_pool = []
            skipped_candidates = []
            for cand_idx in range(getattr(common, "candidates_per_iter", 1)):
                if i >= common.ITERATIONS:
                    break
                print(f"        ↳ Generating candidate {cand_idx+1}/{common.candidates_per_iter}")
            text = generate_policy_with_cot(
                llm=llm,
                system_prompt=common.system_prompt,
                user_prompt=prompt,
                model=common.MODEL,
                temperature=temp
            )
            provider_name = common.LLM_PROVIDER
            print(f"        ↳ Response received from {provider_name.capitalize()} (candidate {cand_idx+1})\n")

            name, desc, code, storage_kb_est, storage_note = parse_policy_sections(text)
            from_template = False
            if not (name and desc and code):
                print(f"❌ [Parse] Missing required sections at iter {i} cand {cand_idx}. Falling back to baseline template.\n")
                baseline_template_path = Path("../ChampSim_CRC2/new_policies/real_data/_hawkeye_baseline.cc")
                if baseline_template_path.exists():
                    code = baseline_template_path.read_text()
                    name = "Hawkeye Baseline (Mode A seed)"
                    desc = (
                        "Pure Hawkeye replacement baseline (OPTgen + SHCT) with Mode B disabled. "
                        "Use this as a safe fallback."
                    )
                    storage_kb_est = None
                    storage_note = ""
                    from_template = True
                else:
                    prev_name = prev_name or "N/A"
                    prev_desc = prev_desc or "N/A"
                    prev_code = prev_code or "N/A"
                    continue

            h = code_hash(code)
            if not from_template and h in seen_hashes:
                print(f"⚠️  [Duplicate] Iteration {i} cand {cand_idx}: code hash already seen; skipping candidate.\n")
                continue
            seen_hashes.add(h)

            # Validation guard removed for Hawkeye-based template to allow flexible Mode B exploration

            explainer_struct = explain_policy_structured(code)
            pred_ipc = None
            diversity = 0.0
            score = 0.0

            candidate_pool.append({
                "name": name,
                "desc": desc,
                "code": code,
                "storage_kb_est": storage_kb_est,
                "storage_note": storage_note,
                "explainer": explainer_struct,
                "pred_ipc": pred_ipc,
                "diversity": diversity,
                "score": score,
                "version": ver
            })
            if pred_ipc is not None:
                surrogate_pred_history.append(pred_ipc)

        # Surrogate disabled: no gating or probe when empty

        if not candidate_pool:
            if not exploration:
                exploitation_budget -= 1
                if exploitation_budget <= 0:
                    exploration = True
                    lineage_id = None
            continue

        candidate_pool.sort(key=lambda c: c["score"], reverse=True)
        to_sim = candidate_pool[:min(getattr(common, "candidates_to_simulate", 1), len(candidate_pool))]

        for cand in to_sim:
            if i >= common.ITERATIONS:
                break
            name = cand["name"]
            desc = cand["desc"]
            code = cand["code"]
            storage_kb_est = cand["storage_kb_est"]
            storage_note = cand["storage_note"]
            explainer_struct = cand["explainer"]
            ver = cand.get("version", ver)

            print(f"     3. 🧾 [Parse] Policy extracted: {name}")
            base = sanitize(f"{name}  v{ver}")
            cc = common.OUT_DIR / f"{i:03}_{base}.cc"
            if FORCE_BASELINE_CODE:
                try:
                    code = BASELINE_SRC.read_text(encoding="utf-8")
                    name = f"{name} (forced-hawkeye)"
                except Exception as e:
                    print(f"      [!] Baseline override failed ({e}); compiling candidate code instead.")
            cc.write_text(code, encoding="utf-8")

            try:
                exe = compile_policy(cc)
            except Exception as e:
                err = str(e)
                print(f"❌ [Compile Error for {name}]\n{err}\n")

                if mem_enabled and memory:
                    memory.record_experiment(
                        iteration=i,
                        policy_name=name,
                        policy_desc=desc,
                        results=[],              # no results
                        compile_ok=False,
                        lineage_id=lineage_id,
                        code_path=str(cc),
                        code_hash=code_hash(code),
                    )
                    memory.add_event("compile_error", err, severity="error")
                    memory._save()

                if not exploration:
                    exploitation_budget -= 1
                    if exploitation_budget <= 0:
                        exploration = True
                        lineage_id = None
                i += 1
                continue

            # Run all simpoints with traces in parallel and aggregate per workload
            tasks = _build_sp_tasks_for_policy(exe, workloads, warmup=int(common.WARMUP_INST), sim=int(common.SIM_INST))
            accum: Dict[str, Dict[str, float]] = {}  # workload -> weighted sums

            if not tasks:
                print("      [!] No simpoint tasks found; check json/workloads.json")
            else:
                print(f"     5. 🏃 [Simulate] Running {len(tasks)} simpoint(s) across {len(workloads)} workload(s) with {MAX_WORKERS} workers")
                with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp.get_context("spawn")) as pool:
                    futures = [pool.submit(_sp_task, t) for t in tasks]
                    for idx, fut in enumerate(as_completed(futures), 1):
                        res = fut.result()
                        if not res.get("ok"):
                            wl = res.get("workload", "<unknown>")
                            msg = res.get("error", "unknown error")
                            print(f"      [!] {name} → {wl}/sp{res.get('index')} FAILED: {msg}")
                            continue

                        wl = res["workload"]
                        w  = float(res["weight"])
                        m  = res["metrics"]  # {'ipc', 'total_hit_rate', 'load_hit_rate', 'rfo_hit_rate', 'prefetch_hit_rate', 'writeback_hit_rate'}

                        if wl not in accum:
                            accum[wl] = {
                                "ipc_h_inv": 0.0, "total_hit_rate": 0.0, "load_hit_rate": 0.0,
                                "rfo_hit_rate": 0.0, "prefetch_hit_rate": 0.0, "writeback_hit_rate": 0.0
                            }

                        eps = 1e-12
                        accum[wl]["ipc_h_inv"]          += w / max(eps, float(m["ipc"]))
                        accum[wl]["total_hit_rate"]     += w * float(m["total_hit_rate"])
                        accum[wl]["load_hit_rate"]      += w * float(m["load_hit_rate"])
                        accum[wl]["rfo_hit_rate"]       += w * float(m["rfo_hit_rate"])
                        accum[wl]["prefetch_hit_rate"]  += w * float(m["prefetch_hit_rate"])
                        accum[wl]["writeback_hit_rate"] += w * float(m["writeback_hit_rate"])

                        print(f"      [+] {name} → {wl}/sp{res['index']} w={w:.3f} "
                            f"TOTAL={m['total_hit_rate']:.6f} LOAD={m['load_hit_rate']:.6f} "
                            f"RFO={m['rfo_hit_rate']:.6f} PREF={m['prefetch_hit_rate']:.6f} "
                            f"WB={m['writeback_hit_rate']:.6f} IPC={m['ipc']:.6f}")

                        if idx % 10 == 0 or idx == len(futures):
                            print(f"      progress: {idx}/{len(futures)}")


        print(f"     Generation explanation for {name}...")
        try:
            policy_explanation = explain_policy_brief(code)
            enhanced_desc = f"{desc}\n\nANALYSIS: {policy_explanation}"
            print(f"     [Analysis] {policy_explanation}")
        except Exception as e:
            print(f"     [!] Explanation failed: {e}")
            enhanced_desc = desc

        results = []
        final_metrics: Dict[str, Dict[str, float]] = {}
        for wl, metrics in accum.items():
            wl_ipc = (1.0 / accum[wl]["ipc_h_inv"]) if accum[wl]["ipc_h_inv"] > 0 else 0.0
            chosen = {
                "ipc": wl_ipc,
                "total_hit_rate": metrics["total_hit_rate"],
                "load_hit_rate": metrics["load_hit_rate"],
                "rfo_hit_rate": metrics["rfo_hit_rate"],
                "prefetch_hit_rate": metrics["prefetch_hit_rate"],
                "writeback_hit_rate": metrics["writeback_hit_rate"],
            }
            src = "candidate"

            # Strict mode: never substitute baseline results; always report candidate IPCs
            best_base = None

            final_metrics[wl] = chosen
            common.upsert_experiment_row(
                conn,
                workload=wl,
                policy=name,
                policy_description=enhanced_desc,
                cpp_file_path=str(cc),
                metrics={
                "ipc":               float(chosen["ipc"]),
                "total_hit_rate":    float(chosen["total_hit_rate"]),
                "load_hit_rate":     float(chosen["load_hit_rate"]),
                "rfo_hit_rate":      float(chosen["rfo_hit_rate"]),
                "prefetch_hit_rate": float(chosen["prefetch_hit_rate"]),
                "writeback_hit_rate":float(chosen["writeback_hit_rate"]),
                },
                score_func=common.default_score_func,
                storage_kb_override=(float(storage_kb_est) if storage_kb_est is not None else None),
                storage_note_override=storage_note
            )
            print(f"      [=] {name} → {wl} (weighted) → "
                f"TOTAL={chosen['total_hit_rate']:.6f} "
                f"LOAD={chosen['load_hit_rate']:.6f} "
                f"RFO={chosen['rfo_hit_rate']:.6f} "
                f"PREF={chosen['prefetch_hit_rate']:.6f} "
                f"WB={chosen['writeback_hit_rate']:.6f} "
                f"IPC={chosen['ipc']:.6f}")
            results.append((wl, chosen["total_hit_rate"], chosen["ipc"]))

        # Explicit per-workload IPC summary (harmonic inputs)
        if final_metrics:
            print("     Per-workload IPC (weighted by simpoints):")
            for wl in sorted(final_metrics.keys()):
                fm = final_metrics[wl]
                print(f"       - {wl}: IPC={fm['ipc']:.6f} TOTAL={fm['total_hit_rate']:.6f}")

        ipc_list = [final_metrics[w]["ipc"] for w in final_metrics]
        if ipc_list:
            eps = 1e-12
            vals = [max(eps, v) for v in ipc_list]
            hm_ipc = len(vals) / sum((1.0 / v) for v in vals)
            BASELINE_IPC = 0.413081
            if hm_ipc < BASELINE_IPC:
                print(f"⚠️  [Baseline Warning] Harmonic IPC {hm_ipc:.6f} is below Hawkeye baseline ({BASELINE_IPC:.6f}). No substitution applied.")
        else:
            hm_ipc = 0.0

        if final_metrics:
            n_wl = len(final_metrics)
            def avg(key: str) -> float:
                return sum(float(final_metrics[w][key]) for w in final_metrics) / n_wl
            all_metrics = {
                "ipc":               hm_ipc,
                "total_hit_rate":    avg("total_hit_rate"),
                "load_hit_rate":     avg("load_hit_rate"),
                "rfo_hit_rate":      avg("rfo_hit_rate"),
                "prefetch_hit_rate": avg("prefetch_hit_rate"),
                "writeback_hit_rate":avg("writeback_hit_rate"),
            }
        else:
            all_metrics = {
                "ipc": 0.0, "total_hit_rate": 0.0, "load_hit_rate": 0.0,
                "rfo_hit_rate": 0.0, "prefetch_hit_rate": 0.0, "writeback_hit_rate": 0.0,
            }

        print(f"✅ [Result] Iteration {i}: {name} → "
            f"avg TOTAL={all_metrics['total_hit_rate']:.6f} | "
            f"avg LOAD={all_metrics['load_hit_rate']:.6f} | "
            f"avg RFO={all_metrics['rfo_hit_rate']:.6f} | "
            f"avg PREF={all_metrics['prefetch_hit_rate']:.6f} | "
            f"avg WB={all_metrics['writeback_hit_rate']:.6f} | "
            f"avg IPC (harmonic)={all_metrics['ipc']:.6f}")

        common.upsert_experiment_row(
            conn,
            workload="ALL",
            policy=name,
            policy_description=enhanced_desc,
            cpp_file_path=str(cc),
            metrics={
            "ipc":               float(all_metrics["ipc"]),
            "total_hit_rate":    float(all_metrics["total_hit_rate"]),
            "load_hit_rate":     float(all_metrics["load_hit_rate"]),
            "rfo_hit_rate":      float(all_metrics["rfo_hit_rate"]),
            "prefetch_hit_rate": float(all_metrics["prefetch_hit_rate"]),
            "writeback_hit_rate":float(all_metrics["writeback_hit_rate"]),
            },
            score_func=common.default_score_func,
            storage_kb_override=(float(storage_kb_est) if storage_kb_est is not None else None),
            storage_note_override=storage_note
        )

        # Log to surrogate history and retrain if needed
        if surrogate_mgr:
            _append_log_entry(surrogate_log_path, {
                "status": "ok",
                "name": name,
                "explainer": explainer_struct,
                "storage": {"estimate_kb": storage_kb_est},
                "metrics": {"average": all_metrics},
                "code_path": str(cc)
            })
            surrogate_mgr.maybe_retrain()
        seen_structs.append(explainer_struct)

        current_score = all_metrics["ipc"]
        prev_best = best_score
        improved = current_score > best_score
        near_best_margin = getattr(common, "exploit_nearbest_margin", 0.0) or 0.0
        near_best = current_score >= (best_score - near_best_margin)
        min_ipc_trigger = getattr(common, "exploit_min_ipc", 0.0) or 0.0
        above_floor = current_score >= min_ipc_trigger

        if exploration:
            if improved or near_best or above_floor:
                exploration = False
                exploitation_budget = 2  # keep exploit bursts short to force regular exploration
                lineage_id = f"{sanitize(name)}_seed_iter_{i}"
                if improved:
                    best_score = current_score
            else:
                # Stay in exploration if the seed failed to beat the current best
                lineage_id = None
                exploitation_budget = 0
        else:
            if improved:
                best_score = current_score
            elif near_best or above_floor:
                # stay in exploitation even without new best, but consume budget
                pass
            else:
                exploration = True  # immediately return to exploration on no improvement
                lineage_id = None
            exploitation_budget -= 1
            if exploitation_budget <= 0:
                exploration = True
                lineage_id = None

        results_list = results
        if mem_enabled and memory:
            memory.record_experiment(
                iteration=i,
                policy_name=name,
                policy_desc=desc,
                results=results_list,
                compile_ok=True,
                lineage_id=lineage_id,
                code_path=str(cc),
                code_hash=code_hash(code),
            )

        prev_name, prev_desc, prev_code = name, desc, code
        last_score = current_score

        if mem_enabled and memory:
            memory._save()

        i += 1
        if i >= common.ITERATIONS:
            break

    prompt_gen.close()
    rag.close()
    conn.close()


if __name__ == "__main__":
    main()
