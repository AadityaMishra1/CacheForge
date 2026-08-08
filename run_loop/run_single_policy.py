#!/usr/bin/env python3
"""
Run a single compiled ChampSim policy binary across all workloads and simpoints,
using the same warmup/sim instruction counts as the main loop.

Example:
  python3 run_single_policy.py --exe ../ChampSim_CRC2/new_policies/experiment_000/007_ship_9b_srrip_r7__confident_mru___shct_aware_victim_refine_v2___demand_only_training___v7.out
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.abspath("../src"))
import common  # noqa: E402


def _build_tasks(exe: Path, workloads: Dict[str, Dict[str, Any]], warmup: int, sim: int) -> List[tuple]:
    tasks: List[Tuple[str, int, float, str, int, int]] = []
    for wname, wdata in workloads.items():
        simpoints = common.normalize_simpoint_weights(wdata.get("simpoints", []))
        for sp in simpoints:
            tasks.append(
                (
                    str(exe),
                    wname,
                    int(sp["index"]),
                    float(sp["_norm_weight"]),
                    str(sp["trace"]),
                    warmup,
                    sim,
                )
            )
    return tasks


def _run_task(args):
    exe_path_str, workload_name, sp_index, sp_weight, trace_path_str, warmup, sim = args
    out = common.run_policy(
        Path(exe_path_str),
        trace_path=Path(trace_path_str),
        warmup_instructions=int(warmup),
        simulation_instructions=int(sim),
        timeout_seconds=None,
    )
    metrics = common.parse_ipc_and_hit_rates(out)
    return {
        "ok": True,
        "workload": workload_name,
        "index": sp_index,
        "weight": sp_weight,
        "trace": trace_path_str,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, help="Path to compiled ChampSim policy .out binary")
    parser.add_argument("--workers", type=int, default=common.max_core, help="Parallel workers (default: common.max_core)")
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}")

    print(f"[run_single_policy] Executable: {exe}")
    print(f"[run_single_policy] Workers: {args.workers}")
    print(f"[run_single_policy] Warmup: {common.WARMUP_INST}  Sim: {common.SIM_INST}")

    workloads = common.get_workloads_dict(common.workloads_json_path)
    tasks = _build_tasks(exe, workloads, warmup=int(common.WARMUP_INST), sim=int(common.SIM_INST))
    print(f"[run_single_policy] Workloads: {len(workloads)}  Simpoints: {len(tasks)}")

    accum: Dict[str, Dict[str, float]] = {}

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=None) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            wl = res["workload"]
            w = float(res["weight"])
            m = res["metrics"]

            if wl not in accum:
                accum[wl] = {
                    "ipc_h_inv": 0.0,
                    "total_hit_rate": 0.0,
                    "load_hit_rate": 0.0,
                    "rfo_hit_rate": 0.0,
                    "prefetch_hit_rate": 0.0,
                    "writeback_hit_rate": 0.0,
                }

            eps = 1e-12
            accum[wl]["ipc_h_inv"] += w / max(eps, float(m["ipc"]))
            accum[wl]["total_hit_rate"] += w * float(m["total_hit_rate"])
            accum[wl]["load_hit_rate"] += w * float(m["load_hit_rate"])
            accum[wl]["rfo_hit_rate"] += w * float(m["rfo_hit_rate"])
            accum[wl]["prefetch_hit_rate"] += w * float(m["prefetch_hit_rate"])
            accum[wl]["writeback_hit_rate"] += w * float(m["writeback_hit_rate"])

            print(
                f"[{idx}/{len(futures)}] {wl}/sp{res['index']} w={w:.3f} "
                f"TOTAL={m['total_hit_rate']:.6f} LOAD={m['load_hit_rate']:.6f} "
                f"RFO={m['rfo_hit_rate']:.6f} PREF={m['prefetch_hit_rate']:.6f} "
                f"WB={m['writeback_hit_rate']:.6f} IPC={m['ipc']:.6f}"
            )

    print("\n[run_single_policy] Per-workload weighted results:")
    wl_ipcs = []
    for wl, metrics in accum.items():
        wl_ipc = 1.0 / metrics["ipc_h_inv"] if metrics["ipc_h_inv"] > 0 else 0.0
        wl_ipcs.append(wl_ipc)
        print(
            f"  {wl}: IPC={wl_ipc:.6f} "
            f"TOTAL={metrics['total_hit_rate']:.6f} "
            f"LOAD={metrics['load_hit_rate']:.6f} "
            f"RFO={metrics['rfo_hit_rate']:.6f} "
            f"PREF={metrics['prefetch_hit_rate']:.6f} "
            f"WB={metrics['writeback_hit_rate']:.6f}"
        )

    overall_ipc = sum(wl_ipcs) / len(wl_ipcs) if wl_ipcs else 0.0
    print(f"\n[run_single_policy] Overall mean IPC across workloads: {overall_ipc:.6f}")


if __name__ == "__main__":
    main()
