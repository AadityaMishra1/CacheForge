# CacheForge

An LLM in a closed loop writes, compiles, and benchmarks C++ cache replacement
policies in the ChampSim (CRC2) simulator. Each iteration generates a candidate
policy, measures its IPC on SPEC CPU 2006 traces, and feeds the result back into
the next prompt — along with a retrieval memory of past experiments and a
surrogate model that prunes weak candidates before simulation.

The strict-ensemble variant keeps Hawkeye fixed as one mode and lets the LLM
experiment on the other. Evolved candidates live in
`ChampSim_CRC2/new_policies/`; the loop's experiment memory is in `memory_json/`.

Built as a three-person course project at NC State.
OptionsEvolver (https://github.com/AadityaMishra1/OptionsEvolver) applies the
same approach to implied-volatility surface fitting.

## Report

[CacheForge final report (PDF)](CacheForge_CSC491_Report.pdf), December 2025.
Table 7 has a transcription error: the CSF IPC for `zeusmp` should be
`1.362090`, not `0.362090`. The corrected value reproduces the aggregate
`0.413873` IPC reported in Table 6.

## Quick start

## Prereqs
- Python 3.9+ with `pip` or `conda`
- g++ (C++17)
- ChampSim traces referenced in `src/100M_1M/json/workloads.json`

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or `pip install -r environment.yml` if using conda-exported deps
```

## Build & Run (strict ensemble loop)
From repo root:
```bash
cd run_loop
python3 run_loop.py
```
- The loop compiles candidates into `ChampSim_CRC2/new_policies/ensemble_strict_v1/`.
- Results land in `DB/ensemble_strict_v1.db` and memory in `memory_json/ensemble_strict_v1.json`.

## Single-Policy Compile/Run
```bash
cd run_loop
python3 run_single_policy.py --cpp ../ChampSim_CRC2/new_policies/real_data/_hawkeye_baseline.cc
```
Outputs go to `ChampSim_CRC2/new_policies/` with matching `.out` binary.

## Useful Scripts
- `RUN_THIS.sh` / `run_strict_baseline.sh`: baseline build-and-run helpers.
- `test_strict_ensemble.sh`: sanity checks for the strict ensemble template.
- `monitor_run.sh`: simple log watcher.

## Data & Logs
- DB: `DB/ensemble_strict_v1.db` (per-workload metrics).
- Memory/logs: `memory_json/ensemble_strict_v1.json` (+ `.surrogate_log.json` if enabled).
- Build logs: `build.log`, `run_strict_ensemble.log`, `run_output.log`.

## Tips
- Mode A is Hawkeye; keep it unchanged. Mode B is where you experiment.
- If a candidate fails compile, check `ChampSim_CRC2/new_policies/ensemble_strict_v1/*.stderr.log`.
- Adjust loop parameters (iterations, workers, gating) in `src/common.py`.
