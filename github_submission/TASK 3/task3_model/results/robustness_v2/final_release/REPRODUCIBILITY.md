# Reproducibility

This release is inside the current C: worktree, not automatically copied to
the earlier D: project. Run PowerShell from this `final_release` directory.

```powershell
$env:PYTHONHASHSEED='20261017'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'D:/anaconda/python.exe' -B -X utf8 ./run_all.py --verify
```

Verification reads the frozen artifact manifest. To perform fresh simulations
without changing any frozen CSV, log or configuration, use a NEW directory:

```powershell
& 'D:/anaconda/python.exe' -B -X utf8 ./run_all.py --reproduce final_release_reproduction_001
```

This copies maintained code/configurations/tests only to a new sibling. It
runs all 83 tests, preflight, five 1000-request pilots, 120 full core cases,
affected F/H cases, numerical repeat checks, artifact validation, analyses
and freezing. Failures stop the command; incomplete cases and logs remain.
Never rerun bootstrap.py on the delivered release: it is a recorded initial
source-copy step, protected against overwrite. build_runner.py records how
the independent producer was generated; it is not necessary for reproduction.

The source tree depends read-only on `task3_model/src`, `results/data`, the
A2 frozen protocol and `task_a2/extensions/common_shocks`, and the predecessor
test and before-repair result evidence. Preserve the project structure.
Installing dependencies in a separate environment is recommended; the above
Python path is the actual tested local interpreter. No API calls, retraining
or large model downloads occur.

`configs/protocol.json` preregisters N=7335, 24 service seeds
2027101800--2027101823, gamma=1.35, Jan 2 2025 UTC alignment, persistence,
six DGX hosts, 48 installed GPUs, one admission slot per host, 12h accounting,
and unchanged shocks. F uses the original first six seeds and four gamma
values. H uses those same six seeds, five CPU multipliers and the preserved
single-branch Spatial CPU control at multipliers 1 and 5.

The main ideal-control cases may be computed by independent worker processes;
their measured walls are descriptive and are not substituted into scale=0
decisions. H runs serially after all such workers have completed and performs
fresh closed-loop computation. Explicit logs identify execution mode. Two
fresh serial repeats of preselected seed 2027101806 supplement its original
core execution: identical terminal digests and absolute metric difference
<=1e-8 are checked, excluding measured wall and artifact paths.

Native HiGHS fixed work/thread/seed controls are NOT VERIFIED in the installed
SciPy1.9.1 / HiGHS1.2.2 interface. Environment variables are fixed before Python
startup, but they do not certify the solver's native thread count. Each
subproblem keeps its original 30s budget (E four-way: 7.5s each); continuous
recovery receives only the remaining budget. Node count, gap, status and wall
are logged when exposed. A 30s budget is a cap, not a claim of fixed work.

Exact wall-time and H trajectory reproduction across machines/load conditions
is not claimed. H is an explicitly hardware-dependent control-cost scenario.
Core deterministic replay consistency, statistical variation across service
seeds, and real-world effectiveness are distinct evidence levels.

Read `FINAL_RESULTS_zh.md` first, then `FINAL_MODEL_RECOMMENDATION.md` and
`FINAL_VALIDATION_REPORT.md`. `FINAL_COMPARISON.csv` contains 120 actual
core runs, not 120 independent service seeds. `results/core/paired_summary.csv`
uses 24 paired seed trajectories. The confidence interval is on
baseline-minus-proposed; positive means carbon reduction for CO2e metrics.

`results/core/before_after_changed_requests.csv.gz` and corresponding F/H
tables retain changed request IDs, regions, starts, completions and failure
states. `numerical_affected_batches.csv` links bad rounded incumbents to full
problem `.npz` evidence in `audit/recovery_candidates/`. Every run directory
saves requests, all events, solver branches, energy ledger, quality-only
posthoc attribution, actual shocks, configuration and metrics. Failed pilot
attempts are separately inventoried, never deleted or counted as new seeds.
