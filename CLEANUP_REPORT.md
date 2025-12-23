# PHASE 0: Repository Hygiene Report

**Date**: 2025-12-23
**Branch**: claude/hpc-measurement-discipline-arJIm
**Commit**: 34b1318 (add energy balance)

## Summary

Repository cleanup completed to enforce reproducibility and minimal versioned artifacts.

## Actions Taken

### 1. Updated `.gitignore`

**Changes:**
- Added OS junk patterns (`.DS_Store`, `Thumbs.db`)
- Added test/linter cache patterns (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`)
- Changed `outputs/` to be ignored by default, with negation rules to preserve documentation:
  - `!outputs/profiling/*.md`
  - `!outputs/profiling/README.md`
- Added profiling artifact patterns: `prof.out`, `scalene*.json`, `pstats_*.txt`, `*.prof`
- Added binary artifact patterns: `*.npz`, `*.npy`
- Commented out aggressive `*.png`, `*.pdf`, `*.txt`, `*.csv`, `*.log` patterns (too broad for scientific computing)

**Rationale**: Previous `.gitignore` was too aggressive, blocking important result files (`.txt`, `.csv`, etc.). New version is targeted to generated artifacts only.

### 2. Deleted Generated Artifacts

**Removed:**
- `src/plastic_hinge_nm.egg-info/` (8.5K) - setuptools build artifact

**Not Found (already clean):**
- No `__pycache__/` directories
- No `.pyc`, `.pyo` files
- No profiling outputs yet (expected)
- No large binary artifacts in repo root

### 3. Created Cleanup Helper Scripts

**New files:**

#### `tools/clean_repo.py`
- **Purpose**: Safe removal of generated artifacts (pycache, build dirs, profiling outputs, OS junk)
- **Safety**: NEVER deletes anything under `src/` or `tests/` (except `__pycache__`)
- **Usage**:
  ```bash
  python tools/clean_repo.py           # dry-run (show what would be deleted)
  python tools/clean_repo.py --apply   # actually delete
  ```

#### `tools/clean_outputs.py`
- **Purpose**: Wipe `outputs/` directory while preserving documentation
- **Preserves**: `outputs/profiling/*.md`, `outputs/profiling/README.md`
- **Usage**:
  ```bash
  python tools/clean_outputs.py        # dry-run
  python tools/clean_outputs.py --apply # actually delete
  ```

## Output Directory Structure (Enforced)

All generated outputs **MUST** be written under `outputs/`:

```
outputs/
├── profiling/
│   ├── <tag>/
│   │   ├── prof.out              # cProfile binary
│   │   ├── pstats_cumtime.txt    # top 60 by cumulative time
│   │   ├── pstats_tottime.txt    # top 60 by total/self time
│   │   ├── manifest.json         # run metadata (commit, env, sizes, timing)
│   │   ├── REPORT.md             # human-readable analysis
│   │   └── scalene.json          # optional (if scalene installed)
│   ├── README.md                 # profiling workflow documentation (versioned)
│   └── RESULTS.md                # optimization history (versioned)
└── <problem_tag>/                # per-problem outputs (NOT versioned)
    └── ...
```

**Enforcement**: `outputs/` is gitignored except for profiling documentation.

## Current State

**Clean**: ✓
- No versioned artifacts
- No generated files in repo root
- No stale build directories

**Ready for profiling**: ✓

## Next Steps

**PHASE 1**: Build profiling infrastructure (`tools/profile_run.py`, `tools/profiling_utils.py`)

---

**Strict HPC Discipline**: No profiling infrastructure yet. Must build it before any baseline run.
