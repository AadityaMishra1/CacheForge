#!/usr/bin/env python3
"""
Quick extractor for memory_json/real_data.json.

Prints:
  - Summary (principles/donts/knobs/open_questions)
  - Best overall policy
  - Best policy per iteration
  - Best IPC per workload
  - Compile failures
  - Top-N policies by avg IPC
  - Optional: detailed dump for policies matching a substring
  - Optional: suite report for paper figures (iteration summaries, baselines, selected policies, global list)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Episode = Dict[str, Any]


def load_data(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def best_overall(episodes: Iterable[Episode]) -> Optional[Episode]:
    return max(
        (e for e in episodes if e.get("avg_ipc") is not None),
        key=lambda e: e["avg_ipc"],
        default=None,
    )


def best_per_iteration(episodes: Iterable[Episode]) -> List[Episode]:
    best_by_iter: Dict[int, Episode] = {}
    for e in episodes:
        if e.get("avg_ipc") is None or e.get("iteration") is None:
            continue
        i = e["iteration"]
        if i not in best_by_iter or e["avg_ipc"] > best_by_iter[i]["avg_ipc"]:
            best_by_iter[i] = e
    return [best_by_iter[i] for i in sorted(best_by_iter)]


def best_per_workload(episodes: Iterable[Episode]) -> List[Tuple[str, Episode, float, float]]:
    best: Dict[str, Tuple[Episode, float, float]] = {}
    for e in episodes:
        per = e.get("per_workload") or {}
        for wl, stats in per.items():
            ipc = stats.get("ipc")
            hit = stats.get("hit_rate")
            if ipc is None:
                continue
            if wl not in best or ipc > best[wl][1]:
                best[wl] = (e, ipc, hit)
    return [(wl, ep, ipc, hit) for wl, (ep, ipc, hit) in sorted(best.items())]


def compile_failures(episodes: Iterable[Episode]) -> List[Episode]:
    return [e for e in episodes if not e.get("compile_ok", True)]


def group_by_iteration(episodes: Iterable[Episode]) -> Dict[int, List[Episode]]:
    groups: Dict[int, List[Episode]] = {}
    for e in episodes:
        if e.get("iteration") is None:
            continue
        groups.setdefault(e["iteration"], []).append(e)
    return groups


def iteration_report(episodes: Iterable[Episode]) -> List[Tuple[int, Episode, bool]]:
    """
    Return (iteration, episode, is_new_best) sorted by iteration then ipc desc.
    is_new_best is flagged when an episode sets a new global IPC high-water mark.
    """
    best_so_far: float = float("-inf")
    rows: List[Tuple[int, Episode, bool]] = []
    groups = group_by_iteration(episodes)
    for it in sorted(groups):
        for e in sorted(groups[it], key=lambda x: (x.get("avg_ipc") or -1), reverse=True):
            ipc = e.get("avg_ipc")
            new_best = ipc is not None and ipc > best_so_far
            if new_best:
                best_so_far = ipc
            rows.append((it, e, new_best))
    return rows


def format_policy_line(e: Episode) -> str:
    name = e.get("policy_name", "<unknown>")
    iter_ = e.get("iteration")
    ipc = e.get("avg_ipc")
    hr = e.get("avg_cache_hit_rate")
    return f"{name} (iter {iter_})  avg_ipc={ipc:.6f}  hit_rate={hr:.6f}"


def safe_float(val: Any) -> str:
    try:
        return f"{float(val):.6f}"
    except Exception:
        return "None"


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def dump_policy_details(e: Episode) -> None:
    print(f"- {e.get('policy_name')} (iter {e.get('iteration')})")
    print(f"  avg_ipc={e.get('avg_ipc')}, avg_cache_hit_rate={e.get('avg_cache_hit_rate')}")
    print(f"  lineage={e.get('lineage_id')}")
    print(f"  code_path={e.get('code_path')}")
    print(f"  code_hash={e.get('code_hash')}")
    lesson = e.get("lesson") or e.get("policy_desc")
    if lesson:
        print(f"  lesson={lesson}")
    per = e.get("per_workload") or {}
    if per:
        parts = [f"{wl}: ipc={v.get('ipc')}, hr={v.get('hit_rate')}" for wl, v in per.items()]
        print("  per_workload: " + "; ".join(parts))


def dump_summary(summary: Dict[str, Any]) -> None:
    print_section("Summary")
    for k in ("principles", "donts", "knobs", "open_questions"):
        vals = summary.get(k)
        if vals:
            print(f"{k}:")
            for item in vals:
                print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract key info from real_data.json.")
    parser.add_argument(
        "--json-path",
        default="memory_json/real_data.json",
        type=Path,
        help="Path to real_data.json (default: memory_json/real_data.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top-N policies by avg_ipc (default: 10)",
    )
    parser.add_argument(
        "--policy-substr",
        type=str,
        default=None,
        help="If set, dump detailed entries whose policy_name contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="If set, print all episodes sorted by avg_ipc.",
    )
    parser.add_argument(
        "--suite-report",
        action="store_true",
        help="Emit paper-oriented data: per-iteration IPC/hit_rate with new-best flag, baselines, selected policies, global list.",
    )
    args = parser.parse_args()

    data = load_data(args.json_path)
    episodes: List[Episode] = data.get("episodes", [])

    if "summary" in data:
        dump_summary(data["summary"])

    # Best overall
    print_section("Best Overall")
    best = best_overall(episodes)
    if best:
        print(format_policy_line(best))
        print(f"  lineage={best.get('lineage_id')}")
        print(f"  code_path={best.get('code_path')}")
        print(f"  code_hash={best.get('code_hash')}")
        if best.get("lesson"):
            print(f"  lesson={best['lesson']}")

    # Best per iteration
    print_section("Best Per Iteration")
    for e in best_per_iteration(episodes):
        print(format_policy_line(e))

    # Best per workload
    print_section("Best Per Workload (by IPC)")
    for wl, ep, ipc, hit in best_per_workload(episodes):
        print(f"{wl}: ipc={ipc:.3f}, hit_rate={hit:.3f} -> {ep['policy_name']} (iter {ep['iteration']}, avg_ipc={ep['avg_ipc']:.6f})")

    # Compile failures
    fails = compile_failures(episodes)
    print_section(f"Compile Failures ({len(fails)})")
    for e in fails:
        print(f"{e.get('policy_name')}  code={e.get('code_path')}")

    # Top-N policies
    print_section(f"Top {args.top} Policies by avg_ipc")
    sorted_eps = sorted((e for e in episodes if e.get("avg_ipc") is not None), key=lambda e: e["avg_ipc"], reverse=True)
    for e in sorted_eps[: args.top]:
        print(format_policy_line(e))

    # Optional: all episodes
    if args.all_episodes:
        print_section("All Episodes (sorted by avg_ipc)")
        for e in sorted_eps:
            print(format_policy_line(e))

    # Optional: detailed dump for matching policies
    if args.policy_substr:
        needle = args.policy_substr.lower()
        print_section(f"Policies matching '{args.policy_substr}'")
        matches = [e for e in episodes if needle in e.get("policy_name", "").lower()]
        if not matches:
            print("  (none)")
        else:
            for e in matches:
                dump_policy_details(e)

    if args.suite_report:
        print_section("Iteration Summary (ipc, hit_rate, new_best)")
        for it, e, new_best in iteration_report(episodes):
            flag = "*" if new_best else " "
            print(
                f"[{flag}] iter={it} id={e.get('id')} name={e.get('policy_name')} "
                f"ipc={safe_float(e.get('avg_ipc'))} hit={safe_float(e.get('avg_cache_hit_rate'))}"
            )

        print_section("Policy ID -> Iteration")
        for it, e, _ in iteration_report(episodes):
            print(f"{e.get('id')} -> iter {it}")

        # Baseline lookup: best-effort search by name substring
        baselines = ["LRU", "MPPPB", "LIME", "RCR", "SHIP++", "Hawkeye"]
        print_section("Baselines (if present)")
        for base in baselines:
            found = next(
                (ep for ep in episodes if base.lower() in ep.get("policy_name", "").lower()),
                None,
            )
            if found:
                print(
                    f"{base}: ipc={safe_float(found.get('avg_ipc'))} "
                    f"hit={safe_float(found.get('avg_cache_hit_rate'))} "
                    f"id={found.get('id')} iter={found.get('iteration')}"
                )
            else:
                print(f"{base}: (not found in episodes)")

        # Selected policies for comparison table/plot
        print_section("Selected Policies (Hawkeye, FUSE-RunGuard, Hawk+StreamShield ShortReuse v2)")
        targets = [
            "Hawkeye",
            "FUSE-RunGuard",
            "Hawk+StreamShield ShortReuse v2",
        ]
        for t in targets:
            match = next(
                (ep for ep in episodes if t.lower() in ep.get("policy_name", "").lower()),
                None,
            )
            if not match:
                print(f"{t}: not found")
                continue
            print(f"{t}: ipc={safe_float(match.get('avg_ipc'))} hit={safe_float(match.get('avg_cache_hit_rate'))}")
            print(f"  id={match.get('id')} iter={match.get('iteration')} code={match.get('code_path')}")
            if match.get("per_workload"):
                parts = [
                    f"{wl}: ipc={safe_float(v.get('ipc'))} hit={safe_float(v.get('hit_rate'))}"
                    for wl, v in match["per_workload"].items()
                ]
                print("  per_workload: " + "; ".join(parts))

        # Global list for sorted plots
        print_section("Valid Policies (sorted by ipc)")
        for e in sorted_eps:
            phase = "post"  # purge/phase flag not tracked in this JSON
            print(
                f"{e.get('policy_name')} | id={e.get('id')} | iter={e.get('iteration')} | "
                f"ipc={safe_float(e.get('avg_ipc'))} | hit={safe_float(e.get('avg_cache_hit_rate'))} | phase={phase}"
            )


if __name__ == "__main__":
    main()
