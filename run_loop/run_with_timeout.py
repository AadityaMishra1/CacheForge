#!/usr/bin/env python3
"""
Fixed version of run_single_policy.py with proper timeout handling.
This prevents simulations from hanging indefinitely.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
import signal

sys.path.append(os.path.abspath("../src"))
import common

# Timeout in seconds (5 minutes per simulation should be plenty)
SIM_TIMEOUT = 300


def _build_tasks(exe: Path, workloads: Dict[str, Dict[str, Any]], warmup: int, sim: int) -> List[tuple]:
    """Build list of simulation tasks"""
    tasks: List[Tuple[str, str, int, float, str, int, int]] = []
    for wname, wdata in workloads.items():
        simpoints = common.normalize_simpoint_weights(wdata.get("simpoints", []))
        for sp in simpoints:
            if sp.get("trace"):  # Only add if trace exists
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
    """Run a single simulation task with timeout"""
    exe_path_str, workload_name, sp_index, sp_weight, trace_path_str, warmup, sim = args

    try:
        # Run with timeout
        out = common.run_policy(
            Path(exe_path_str),
            trace_path=Path(trace_path_str),
            warmup_instructions=int(warmup),
            simulation_instructions=int(sim),
            timeout_seconds=SIM_TIMEOUT,  # KEY FIX: Add timeout
        )
        metrics = common.parse_ipc_and_hit_rates(out)
        return {
            "ok": True,
            "workload": workload_name,
            "index": sp_index,
            "weight": sp_weight,
            "trace": trace_path_str,
            "metrics": metrics,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "workload": workload_name,
            "index": sp_index,
            "weight": sp_weight,
            "trace": trace_path_str,
            "metrics": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run ChampSim policy with timeout protection")
    parser.add_argument("--exe", required=True, help="Path to compiled ChampSim policy .out binary")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--timeout", type=int, default=SIM_TIMEOUT, help=f"Timeout per sim in seconds (default: {SIM_TIMEOUT})")
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        raise SystemExit(f"ERROR: Executable not found: {exe}")

    print(f"[run_with_timeout] Executable: {exe}")
    print(f"[run_with_timeout] Workers: {args.workers}")
    print(f"[run_with_timeout] Warmup: {common.WARMUP_INST}  Sim: {common.SIM_INST}")
    print(f"[run_with_timeout] Timeout per simulation: {args.timeout} seconds")

    workloads = common.get_workloads_dict(common.workloads_json_path)
    tasks = _build_tasks(exe, workloads, warmup=int(common.WARMUP_INST), sim=int(common.SIM_INST))
    print(f"[run_with_timeout] Workloads: {len(workloads)}  Simpoints: {len(tasks)}\n")

    accum: Dict[str, Dict[str, float]] = {}
    failed = []
    timeouts = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_task, t) for t in tasks]

        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result(timeout=args.timeout + 10)  # Extra buffer for process overhead

                if not res["ok"]:
                    failed.append(res)
                    print(f"[{idx}/{len(futures)}] {res['workload']}/sp{res['index']} FAILED: {res['error']}")
                    continue

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

            except Exception as e:
                timeouts.append(str(e))
                print(f"[{idx}/{len(futures)}] TIMEOUT or ERROR: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("[run_with_timeout] SUMMARY")
    print("=" * 80)
    print(f"Completed: {len(accum)} workloads")
    print(f"Failed: {len(failed)}")
    print(f"Timeouts: {len(timeouts)}")

    if accum:
        print("\n[run_with_timeout] Per-workload weighted results:")
        wl_ipcs = []
        for wl in sorted(accum.keys()):
            metrics = accum[wl]
            wl_ipc = 1.0 / metrics["ipc_h_inv"] if metrics["ipc_h_inv"] > 0 else 0.0
            wl_ipcs.append(wl_ipc)
            print(
                f"  {wl:15s}: IPC={wl_ipc:.6f} "
                f"TOTAL={metrics['total_hit_rate']:.6f} "
                f"LOAD={metrics['load_hit_rate']:.6f} "
                f"RFO={metrics['rfo_hit_rate']:.6f} "
                f"PREF={metrics['prefetch_hit_rate']:.6f} "
                f"WB={metrics['writeback_hit_rate']:.6f}"
            )

        overall_ipc = sum(wl_ipcs) / len(wl_ipcs) if wl_ipcs else 0.0
        print(f"\n[run_with_timeout] Overall mean IPC across workloads: {overall_ipc:.6f}")
    else:
        print("\nWARNING: No successful simulations!")

    if failed:
        print(f"\nFailed simulations: {len(failed)}")
        for f in failed[:5]:  # Show first 5
            print(f"  - {f['workload']}/sp{f['index']}: {f['error']}")

    if timeouts:
        print(f"\nTimeouts: {len(timeouts)}")


if __name__ == "__main__":
    main()
