from __future__ import annotations

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

from common import *
from RAG import *



OUT_DIR.mkdir(exist_ok=True, parents=True)
Path(perf_out_path).mkdir(exist_ok=True, parents=True) 
Path(json_path).mkdir(exist_ok=True, parents=True)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def _normalize_workload_subset(lst) -> list[str]:
    if isinstance(lst, str):
        # allow comma or whitespace separated strings
        return [x.strip() for x in re.split(r"[,\\s]+", lst) if x.strip()]
    if isinstance(lst, (list, tuple, set)):
        return [str(x) for x in lst]
    return []

def _filter_workloads_dict(workloads: Dict[str, Dict[str, Any]], keep: list[str]) -> Dict[str, Dict[str, Any]]:
    if not keep:
        return workloads
    return {k: v for k, v in workloads.items() if k in keep}

def _filter_performance_data(perf: Dict[str, Any], keep: list[str]) -> Dict[str, Any]:
    if not keep:
        return perf
    return {w: perf[w] for w in perf.keys() if w in keep}

def _clear_table(conn: sqlite3.Connection, table: str) -> None:
    c = conn.cursor()
    c.execute(f"DELETE FROM {table}")
    conn.commit()

def _reset_workloads_tables(conn: sqlite3.Connection) -> None:
    # remove all workloads and their simpoints so re-insert reflects the new subset exactly
    _clear_table(conn, "workload_simpoints")
    _clear_table(conn, "workloads")

def _reset_experiments(conn: sqlite3.Connection) -> None:
    _clear_table(conn, "experiments")

def _run_single_task(args):
    # Keep args simple (picklable)
    policy_name, exe_path_str, workload_name, sp_index, sp_weight, trace_path_str = args

    output = run_policy(Path(exe_path_str), trace_path=Path(trace_path_str), warmup_instructions=WARMUP_INST, simulation_instructions=SIM_INST)

    # Save raw output (make filename unique)
    out_name = f"{workload_name}_{policy_name}sp{sp_index}{Path(trace_path_str).name}.txt"
    out_dir = Path(perf_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / out_name).write_text(output, encoding="utf-8")

    metrics = parse_ipc_and_hit_rates(output)
    return {
        "workload": workload_name,
        "policy": policy_name,
        "index": sp_index,
        "weight": sp_weight,
        "trace": trace_path_str,
        "metrics": metrics,
    }


def compute_performance_data_in_parallel(policies, workloads, max_workers: Optional[int] = max_core):
    performance_data = defaultdict(dict)

    # 1) Compile policies (as before)
    compiled_bins = {}
    for policy_name, pdata in policies.items():
        policy_path = Path(pdata["file_path"])
        print(f"[+] Compiling policy: {policy_name}")
        compiled_bins[policy_name] = compile_policy(policy_path, OUT_DIR)

    # 2) Build tasks = one per (workload, policy, simpoint-with-trace)
    tasks = []
    for workload_name, wdata in workloads.items():
        simpoints = normalize_simpoint_weights(wdata.get("simpoints", []))
        if not simpoints:
            print(f"[WARN] No simpoint traces for '{workload_name}', skipping.")
            continue

        for policy_name, exe in compiled_bins.items():
            for sp in simpoints:
                tasks.append((
                    policy_name,
                    str(exe),
                    workload_name,
                    int(sp["index"]),
                    float(sp["_norm_weight"]),
                    str(sp["trace"]),
                ))

    if not tasks:
        print("[!] No tasks to run.")
        return performance_data

    # 3) Choose parallelism level (tune for your machine)
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) // 2)

    print(f"[+] Running {len(tasks)} tasks with {max_workers} workers...")

    # 4) Run tasks in parallel and aggregate
    accum = {}  # key: (workload, policy) -> accumulators + simpoints list
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_single_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] Task failed: {e}")
                continue

            w = res["workload"]
            p = res["policy"]
            sp_i = res["index"]
            sp_w = res["weight"]
            trace = res["trace"]
            m = res["metrics"]

            key = (w, p)
            if key not in accum:
                accum[key] = {
                    "w_ipc_inv": 0.0, "w_total": 0.0, "w_load": 0.0,
                    "w_rfo": 0.0, "w_pref": 0.0, "w_wb": 0.0,
                    "simpoints": []
                }

            acc = accum[key]
            eps = 1e-12
            acc["w_ipc_inv"]    += sp_w / max(eps, m["ipc"])
            acc["w_total"]      += sp_w * m["total_hit_rate"]
            acc["w_load"]       += sp_w * m["load_hit_rate"]
            acc["w_rfo"]        += sp_w * m["rfo_hit_rate"]
            acc["w_pref"]       += sp_w * m["prefetch_hit_rate"]
            acc["w_wb"]         += sp_w * m["writeback_hit_rate"]

            acc["simpoints"].append({
                "index": sp_i,
                "weight": sp_w,
                "ipc": m["ipc"],
                "total_hit_rate": m["total_hit_rate"],
                "load_hit_rate": m["load_hit_rate"],
                "rfo_hit_rate": m["rfo_hit_rate"],
                "prefetch_hit_rate": m["prefetch_hit_rate"],
                "writeback_hit_rate": m["writeback_hit_rate"],
                "trace": trace
            })

            if i % 10 == 0 or i == len(futures):
                print(f"  progress: {i}/{len(futures)}")

    # 5) Build performance_data in the same shape as before
    for (workload_name, policy_name), acc in accum.items():
        hm_ipc = 0.0 if acc["w_ipc_inv"] <= 0 else 1.0 / acc["w_ipc_inv"]
        performance_data[workload_name][policy_name] = {
            "weighted": {
                "ipc": hm_ipc,
                "total_hit_rate": acc["w_total"],
                "load_hit_rate": acc["w_load"],
                "rfo_hit_rate": acc["w_rfo"],
                "prefetch_hit_rate": acc["w_pref"],
                "writeback_hit_rate": acc["w_wb"]
            },
            "simpoints": acc["simpoints"]
        }

    # 6) Compute overall averages (same as before)
    overall_avg = {}
    for policy_name in compiled_bins.keys():
        entries = [performance_data[w][policy_name]["weighted"]
                for w in performance_data if policy_name in performance_data[w]]
        if not entries:
            overall_avg[policy_name] = {
                "ipc": 0.0, "total_hit_rate": 0.0, "load_hit_rate": 0.0,
                "rfo_hit_rate": 0.0, "prefetch_hit_rate": 0.0, "writeback_hit_rate": 0.0
            }
            continue
        n = len(entries)
        eps = 1e-12
        vals = [max(eps, e["ipc"]) for e in entries if e["ipc"] > 0]
        hm_ipc = 0.0 if not vals else (len(vals) / sum(1.0/v for v in vals))
        overall_avg[policy_name] = {
            "ipc": hm_ipc,
            "total_hit_rate": sum(e["total_hit_rate"] for e in entries) / n,
            "load_hit_rate": sum(e["load_hit_rate"] for e in entries) / n,
            "rfo_hit_rate": sum(e["rfo_hit_rate"] for e in entries) / n,
            "prefetch_hit_rate": sum(e["prefetch_hit_rate"] for e in entries) / n,
            "writeback_hit_rate": sum(e["writeback_hit_rate"] for e in entries) / n,
        }

    print("\n[Summary] Overall averages across workloads per policy:")
    for policy_name, avg in overall_avg.items():
        print(f"  {policy_name}: "
            f"IPC={avg['ipc']:.4f}, TOTAL={avg['total_hit_rate']:.4f}, "
            f"LOAD={avg['load_hit_rate']:.4f}, RFO={avg['rfo_hit_rate']:.4f}, "
            f"PREF={avg['prefetch_hit_rate']:.4f}, WB={avg['writeback_hit_rate']:.4f}")

    # 7) Save JSON (same paths as before)
    Path(Perf_json_path).write_text(
        json.dumps(performance_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(Score_json_path).write_text(
        json.dumps(overall_avg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return performance_data


def compute_performance_data(policies, workloads):

    performance_data = defaultdict(dict)

    # Pre-compile all policies once
    compiled_bins = {}
    for policy_name, pdata in policies.items():
        policy_path = Path(pdata["file_path"])
        print(f"[+] Compiling policy: {policy_name}")
        compiled_bins[policy_name] = compile_policy(policy_path, OUT_DIR)

    # --- Step 2: Run all policies for each workload (all simpoints) ---
    for workload_name, wdata in workloads.items():
        print(f"\n[+] Running all policies for workload: {workload_name}")

        simpoints = normalize_simpoint_weights(wdata.get("simpoints", []))
        if not simpoints:
            print(f"  [WARN] No simpoint traces found for '{workload_name}', skipping.")
            continue

        for policy_name, exe in compiled_bins.items():
            print(f"  → Policy Name: {policy_name}")

            # Weighted accumulators
            eps = 1e-12
            w_ipc_inv = 0.0
            w_total = 0.0
            w_load = 0.0
            w_rfo = 0.0
            w_pref = 0.0
            w_wb = 0.0

            per_sp_results = []

            for sp in simpoints:
                trace_path = sp["trace"]
                sp_index = sp["index"]
                sp_w = sp["_norm_weight"]

                print(f"    - SP index {sp_index}, weight {sp_w:.6f}, trace {Path(trace_path).name}")

                # Run the policy on this checkpoint/trace
                out = run_policy(exe, trace_path=trace_path, warmup_instructions=WARMUP_INST, simulation_instructions=SIM_INST)

                with open(f"{perf_out_path}/{policy_name}_{Path(trace_path).name}.txt", "w") as f:
                    f.write(out)
                #print(out)

                # Parse IPC and hit rates
                metrics = parse_ipc_and_hit_rates(out)
                ipc = metrics["ipc"]
                total_hr = metrics["total_hit_rate"]
                load_hr = metrics["load_hit_rate"]
                rfo_hr = metrics["rfo_hit_rate"]
                pref_hr = metrics["prefetch_hit_rate"]
                wb_hr = metrics["writeback_hit_rate"]

                per_sp_results.append({
                    "index": sp_index,
                    "weight": sp_w,
                    "ipc": ipc,
                    "total_hit_rate": total_hr,
                    "load_hit_rate": load_hr,
                    "rfo_hit_rate": rfo_hr,
                    "prefetch_hit_rate": pref_hr,
                    "writeback_hit_rate": wb_hr,
                    "trace": trace_path
                })

                # Weighted accumulation
                w_ipc_inv   += sp_w / max(eps, ipc)
                w_total     += sp_w * total_hr
                w_load      += sp_w * load_hr
                w_rfo       += sp_w * rfo_hr
                w_pref      += sp_w * pref_hr
                w_wb        += sp_w * wb_hr

                print(f"      ↳ IPC: {ipc:.4f} | TOTAL: {total_hr:.4f} | LOAD: {load_hr:.4f} | RFO: {rfo_hr:.4f} | PREF: {pref_hr:.4f} | WB: {wb_hr:.4f}")

            performance_data[workload_name][policy_name] = {
                "weighted": {
                    "ipc": (0.0 if w_ipc_inv <= 0 else 1.0 / w_ipc_inv),
                    "total_hit_rate": w_total,
                    "load_hit_rate": w_load,
                    "rfo_hit_rate": w_rfo,
                    "prefetch_hit_rate": w_pref,
                    "writeback_hit_rate": w_wb
                },
                "simpoints": per_sp_results
            }

            hm_ipc = 0.0 if w_ipc_inv <= 0 else 1.0 / w_ipc_inv
            print(f"    ⇒ Weighted ({workload_name}, {policy_name}): "
                f"IPC={hm_ipc:.4f}, TOTAL={w_total:.4f}, LOAD={w_load:.4f}, "
                f"RFO={w_rfo:.4f}, PREF={w_pref:.4f}, WB={w_wb:.4f}")

    # overall averages across workloads for each policy
    overall_avg = {}
    for policy_name in compiled_bins.keys():
        entries = [performance_data[w][policy_name]["weighted"]
                for w in performance_data if policy_name in performance_data[w]]
        if not entries:
            overall_avg[policy_name] = {"ipc": 0.0, "total_hit_rate": 0.0, "load_hit_rate": 0.0,
                                        "rfo_hit_rate": 0.0, "prefetch_hit_rate": 0.0, "writeback_hit_rate": 0.0}
            continue
        n = len(entries)
        eps = 1e-12
        vals = [max(eps, e["ipc"]) for e in entries if e["ipc"] > 0]
        hm_ipc = 0.0 if not vals else (len(vals) / sum(1.0/v for v in vals))
        overall_avg[policy_name] = {
            "ipc": hm_ipc,
            "total_hit_rate": sum(e["total_hit_rate"] for e in entries) / n,
            "load_hit_rate": sum(e["load_hit_rate"] for e in entries) / n,
            "rfo_hit_rate": sum(e["rfo_hit_rate"] for e in entries) / n,
            "prefetch_hit_rate": sum(e["prefetch_hit_rate"] for e in entries) / n,
            "writeback_hit_rate": sum(e["writeback_hit_rate"] for e in entries) / n,
        }

    print("\n[Summary] Overall averages across workloads per policy:")
    for policy_name, avg in overall_avg.items():
        print(f"  {policy_name}: "
            f"IPC={avg['ipc']:.4f}, TOTAL={avg['total_hit_rate']:.4f}, "
            f"LOAD={avg['load_hit_rate']:.4f}, RFO={avg['rfo_hit_rate']:.4f}, "
            f"PREF={avg['prefetch_hit_rate']:.4f}, WB={avg['writeback_hit_rate']:.4f}")
        
    #-------------------------   
    # save in a json file
    #------------------------- 
    out_path = Path(Perf_json_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(performance_data, f, indent=2, ensure_ascii=False)

    avg_path = Path(Score_json_path)
    with avg_path.open("w", encoding="utf-8") as f:
        json.dump(overall_avg, f, indent=2, ensure_ascii=False)
    
    return performance_data


def main():
    workloads = get_workloads_dict(workloads_json_path)

    # Normalize subset list from common.py
    subset = []
    if few_workloads:
        subset = _normalize_workload_subset(few_workloads_list)
        if subset:
            workloads = _filter_workloads_dict(workloads, subset)
        else:
            print("[WARN] few_workloads=True but few_workloads_list is empty; no filtering applied.")

    # Optionally load policies + perf only if we will insert them
    if not empty_DB:

        with open(Policies_json_path, "r") as f:
            policies = json.load(f)

        
        # If you compute fresh results:
        #performance_data = compute_performance_data_in_parallel(policies, workloads)


        with Path(Perf_json_path).open("r", encoding="utf-8") as f:
            loaded_performance_data = json.load(f)
        with Path(Score_json_path).open("r", encoding="utf-8") as f:
            loaded_score = json.load(f)

        # Filter performance data to the subset of workloads (if any)
        if subset:
            loaded_performance_data = _filter_performance_data(loaded_performance_data, subset)

    conn = sqlite3.connect(DB_PATH)
    ensure_experiments_schema(conn)
    ensure_workloads_schema(conn)

    # If restricting workloads, reset the tables so only the subset remains
    if few_workloads:
        _reset_workloads_tables(conn)

    # Always (re)insert workloads (full or filtered)
    insert_workloads(conn, workloads)

    if empty_DB:
        # Ensure no policy results are present (truly empty experiments table)
        _reset_experiments(conn)
        conn.close()
        print("[INFO] Created DB with workloads only (no policy results).")
        rag = ExperimentRAG(DB_PATH)
        # Optional: show workloads present
        print(rag.get_all_workloads_with_description())
        rag.close()
        return

    # Otherwise, insert policy performance data and ALL-score rows
    insert_performance_data(conn, loaded_performance_data, policies)
    insert_scores(conn, loaded_score, policies, workload_label="ALL")

    conn.close()
    print("Inserted workloads and policy data into the database.")

    rag = ExperimentRAG(DB_PATH)
    rag.print_top_policies_by_metric("ALL", metric="score", top_n=6)

if __name__ == "__main__":
    main()
