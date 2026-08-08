import sys
from pathlib import Path

# Add repo src to path
sys.path.append("/workspace/src")
import common

# Override workloads path to the repo copy
common.workloads_json_path = "/workspace/src/100M_1M/json/workloads.json"
common.WORKLOAD_JSON_PATH = common.workloads_json_path

# Policy to test (change if needed)
policy_path = Path("/workspace/ChampSim_CRC2/new_policies/real_data/006_hawk_stride_tinylfu_ensemble__hst_e___v0.cc")

# Build task list
workloads = common.get_workloads_dict(common.workloads_json_path)
tasks = []
for wname, wdata in workloads.items():
    for sp in common.normalize_simpoint_weights(wdata.get("simpoints", [])):
        trace = sp.get("trace")
        if trace:
            tasks.append((wname, float(sp["_norm_weight"]), Path(trace)))
if not tasks:
    raise SystemExit("No simpoints with traces found")

# Compile
exe = common.compile_policy(policy_path, Path(common.OUT_DIR))

# Run and accumulate
per_wl = {}
for wname, w, trace in tasks:
    out = common.run_policy(
        exe,
        trace_path=trace,
        warmup_instructions=int(common.WARMUP_INST),
        simulation_instructions=int(common.SIM_INST),
    )
    m = common.parse_ipc_and_hit_rates(out)
    acc = per_wl.setdefault(wname, {
        "ipc_h_inv":0.0,"total_hit_rate":0.0,"load_hit_rate":0.0,
        "rfo_hit_rate":0.0,"prefetch_hit_rate":0.0,"writeback_hit_rate":0.0
    })
    eps = 1e-12
    acc["ipc_h_inv"]          += w / max(eps, float(m["ipc"]))
    acc["total_hit_rate"]     += w * float(m["total_hit_rate"])
    acc["load_hit_rate"]      += w * float(m["load_hit_rate"])
    acc["rfo_hit_rate"]       += w * float(m["rfo_hit_rate"])
    acc["prefetch_hit_rate"]  += w * float(m["prefetch_hit_rate"])
    acc["writeback_hit_rate"] += w * float(m["writeback_hit_rate"])
    print(f"[+] {policy_path.name} → {wname} trace={trace.name} IPC={m['ipc']:.6f}")

# Per-workload + overall harmonic IPC
ipc_list=[]
for wl, acc in per_wl.items():
    ipc = (1.0/acc["ipc_h_inv"]) if acc["ipc_h_inv"]>0 else 0.0
    ipc_list.append(ipc)
    print(f"[=] {wl}: IPC={ipc:.6f} TOTAL={acc['total_hit_rate']:.6f}")
hm_ipc = len(ipc_list)/sum(1.0/max(1e-12,v) for v in ipc_list) if ipc_list else 0.0
print(f"=== Overall harmonic IPC: {hm_ipc:.6f} ===")
