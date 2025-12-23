# HPC Profiling Infrastructure — Strict Measurement Discipline

**No optimization without profiling. No speculation without evidence.**

This directory contains profiling runs with full provenance tracking.

## Quick Start

### 1. Enforce Single-Thread Baseline

**MANDATORY** for reproducible baseline profiling:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Add these to your `~/.bashrc` or run them before each profiling session.

### 2. Run Baseline Profiling

**Gravity-only baseline** (fastest, recommended for initial profiling):

```bash
PYTHONPATH=src python tools/profile_run.py --tag baseline_hht_gravity \
    --integrator hht --beam-hinge shm --state gravity --gravity-steps 10
```

**Full IDA baseline** (longer, complete workload):

```bash
PYTHONPATH=src python tools/profile_run.py --tag baseline_hht_ida \
    --integrator hht --beam-hinge shm --state ida --ag-min 0.1 --ag-max 0.3 --ag-step 0.1
```

**Compare integrators**:

```bash
# HHT
PYTHONPATH=src python tools/profile_run.py --tag baseline_hht --integrator hht --state gravity

# Newmark
PYTHONPATH=src python tools/profile_run.py --tag baseline_newmark --integrator newmark --state gravity

# Explicit
PYTHONPATH=src python tools/profile_run.py --tag baseline_explicit --integrator explicit --state gravity
```

**Compare beam hinge models**:

```bash
# SHM hinge
PYTHONPATH=src python tools/profile_run.py --tag baseline_shm --beam-hinge shm --state gravity

# Fiber hinge (much slower)
PYTHONPATH=src python tools/profile_run.py --tag baseline_fiber --beam-hinge fiber --state gravity
```

### 3. View Results

Each profiling run creates:

```
outputs/profiling/<tag>/
  ├── prof.out                  # Raw cProfile data
  ├── pstats_cumtime.txt        # Top 60 functions by cumulative time
  ├── pstats_tottime.txt        # Top 60 functions by self time
  ├── manifest.json             # Full provenance (commit, env, problem sizes)
  └── REPORT.md                 # Human-readable evidence report
```

**Read the report**:

```bash
cat outputs/profiling/baseline_hht_gravity/REPORT.md
```

**Open in pstats interactive**:

```bash
python -m pstats outputs/profiling/baseline_hht_gravity/prof.out
```

## Output Structure

### manifest.json

Full provenance tracking:
- Git commit hash + dirty status
- Full command line
- Python version, NumPy version, platform info
- CPU model + core count
- Thread control environment variables (MUST be 1 for baseline)
- Problem parameters: ndof, nsteps, dt, n_elem, n_hinge, etc.
- Wall time and CPU time

### REPORT.md

Evidence-based analysis:
- Baseline validation (single-thread check)
- Problem parameters table
- Timing summary (wall time, CPU time, steps/sec)
- Top 20 hotspots by cumulative time
- Top 20 hotspots by self time
- Detailed analysis of top 3 hotspots with categorization:
  - Python loop/dispatch overhead
  - allocation pressure (repeated temporary arrays)
  - NumPy kernel dominated
  - linear algebra dominated
  - I/O dominated
  - postprocessing inside time loop

## Profiling Best Practices

### DO:
- ✅ Always set thread control env vars to 1
- ✅ Use `--state gravity` for quick profiling iterations
- ✅ Tag each run uniquely (baseline, opt1, opt2, etc.)
- ✅ Commit code before profiling (avoid dirty state)
- ✅ Read REPORT.md before proposing optimizations

### DON'T:
- ❌ Profile without single-thread baseline
- ❌ Profile with uncommitted code changes (dirty state OK, but document it)
- ❌ Run full IDA for quick profiling (use gravity-only)
- ❌ Optimize without comparing before/after profiles
- ❌ Trust "feelings" about what's slow (evidence only)

## Comparing Runs

To compare two profiling runs:

```bash
# Run baseline
PYTHONPATH=src python tools/profile_run.py --tag before --state gravity

# Apply optimization
# ...edit code...

# Run optimized version
PYTHONPATH=src python tools/profile_run.py --tag after --state gravity

# Compare
diff -u outputs/profiling/before/REPORT.md outputs/profiling/after/REPORT.md
```

Key metrics to compare:
1. Wall time (must decrease)
2. Top hotspot cumulative time (should shift)
3. Call counts (should decrease for expensive functions)

## Advanced: Scalene (Optional)

If you have `scalene` installed:

```bash
pip install scalene

scalene --json --outfile outputs/profiling/<tag>/scalene.json \
    --profile-all --cpu-only \
    src/problems/problema4_portico.py -- --state gravity --integrator hht --beam-hinge shm
```

Scalene provides line-level profiling but is not required for baseline profiling.

## Correctness Verification

After optimization, always verify correctness:

```bash
# Run baseline
PYTHONPATH=src python -m problems.problema4_portico --state gravity --integrator hht --beam-hinge shm

# Extract key invariants from outputs/problem4_shm_gravity/
# - Final displacement norm
# - Peak base shear
# - Energy residuals

# Run optimized version
# DC_FAST=1 PYTHONPATH=src python -m problems.problema4_portico --state gravity --integrator hht --beam-hinge shm

# Compare outputs (must match within stated tolerances)
```

Tolerances:
- Displacements: 1e-10 relative
- Forces: 1e-8 relative
- Energy: 1e-6 relative

---

**Remember**: No profiling, no optimization. No data, no decisions.
