# HPC Optimization Plan — Evidence-Based, Zero Speculation

**Date**: 2025-12-23
**Baseline**: `baseline_hht_ida` (commit `612b971f`, HHT integrator, SHM hinges, 2 IDA runs)
**Wall time**: 77.856 s
**Problem size**: 18 elements, 4 hinges, 24 DOF, 10,002 timesteps (2×5001)

---

## Executive Summary

**Postprocessing accounts for 32% of runtime** (25s matplotlib, 11s plotting, 6s exports). However, **computational kernels dominate the remaining 68%** (53s).

**Top 3 computational hotspots** (by self time, excluding postprocessing):

1. **Model.assemble()** — 25.05s (32% total, 47% of computation)
   - Called 34,891 times (~3.5 iterations/step average)
   - Aggregates element and hinge contributions into global K and R

2. **Element stiffness computation** — 19.4s (25% total, 37% of computation)
   - `stiffness_and_force_global()`: 10.07s, 989,082 calls
   - `k_local()`: 5.01s, 989,280 calls
   - `rot2d()`: 2.24s, 989,568 calls
   - `_geom()`: 2.13s, 1,978,884 calls

3. **Allocation pressure** — 3.68s (5% total, 7% of computation)
   - `numpy.array()`: 2.61s, 2,553,437 calls
   - `numpy.zeros()`: 1.07s, 2,109,248 calls

**Linear solve is NOT the bottleneck**: `numpy.linalg.solve()` accounts for only 0.90s (1.2%).

---

## Hotspot Categorization (Top 10 by Self Time, Computational Only)

| Rank | Function | tottime | % | ncalls | Category |
|------|----------|---------|---|--------|----------|
| 1 | `Model.assemble()` | 25.05s | 32.2% | 34,891 | Python loop/dispatch overhead |
| 2 | `frame2d.stiffness_and_force_global()` | 10.07s | 12.9% | 989,082 | NumPy kernel dominated |
| 3 | `frame2d.k_local()` | 5.01s | 6.4% | 989,280 | NumPy kernel dominated |
| 4 | `numpy.array()` | 2.61s | 3.4% | 2,553,437 | allocation pressure |
| 5 | `SHMBeamHinge1D.eval_trial()` | 2.48s | 3.2% | 179,584 | Python loop (Bouc-Wen update) |
| 6 | `frame2d.rot2d()` | 2.24s | 2.9% | 989,568 | allocation pressure (2×2 matrices) |
| 7 | `moment_capacity_from_polygon()` | 2.20s | 2.8% | 20,109 | Python loop (N-M interaction) |
| 8 | `frame2d._geom()` | 2.13s | 2.7% | 1,978,884 | NumPy kernel dominated |
| 9 | `SHMBeamHinge1D.eval_increment()` | 1.38s | 1.8% | 89,792 | Python loop (Bouc-Wen commit) |
| 10 | `numpy.zeros()` | 1.07s | 1.4% | 2,109,248 | allocation pressure |

**Total: 56.24s (72% of wall time)**

---

## Root Cause Analysis

### Hotspot #1: `Model.assemble()` (25.05s, 32%)

**Evidence** (src/dc_solver/fem/model.py:53):
- Called 34,891 times (average ~3.5 Newton iterations per timestep)
- Loops over 18 beams + 4 hinges per call
- Aggregates local K and R into global arrays
- **High self-time indicates Python loop overhead, not NumPy kernel cost**

**Why it's hot**:
- Python `for` loop over elements/hinges (22 iterations per assemble)
- Repeated dictionary lookups (`dof_map`)
- Repeated array slicing for global K/R updates

**Callers**: Called by `hht_alpha_newton()` inside the Newton corrector loop (lines 265, 276 per iteration).

**Categorization**: Python loop/dispatch overhead.

---

### Hotspot #2: Element stiffness (19.4s total, 25%)

**Evidence** (src/dc_solver/fem/frame2d.py):
- `stiffness_and_force_global()`: 989,082 calls (18 elements × 34,891 assemble calls ÷ 2? Likely includes trial states)
- Each call:
  - Computes local 6×6 stiffness via `k_local()`
  - Applies 2D rotation via `rot2d()` (cosine/sine matrix)
  - Computes geometry via `_geom()` (length, unit vectors)
  - Returns global K and R slices

**Why it's hot**:
- Called ~989k times (very high frequency)
- Each call allocates multiple temporary arrays:
  - `rot2d()` allocates 2×2 rotation matrix per call (~989k allocations)
  - `k_local()` allocates 6×6 local stiffness
  - `stiffness_and_force_global()` allocates 6×6 global transformed stiffness
- NumPy operations on small matrices (6×6, 2×2) have Python overhead

**Categorization**: NumPy kernel dominated + allocation pressure.

---

### Hotspot #3: Allocation pressure (3.68s, 5%)

**Evidence**:
- `numpy.array()`: 2,553,437 calls
- `numpy.zeros()`: 2,109,248 calls
- Both are **pure allocation overhead**, not computation

**Why it's hot**:
- Repeated temporary array creation inside tight loops
- No preallocated buffers for local stiffness, rotation matrices, geometry
- Each element stiffness call allocates ~3-5 temporaries

**Categorization**: allocation pressure (repeated temporary arrays).

---

### Hotspot #4: SHM hinge evaluation (3.86s, 5%)

**Evidence** (src/dc_solver/hinges/models.py):
- `eval_trial()`: 2.48s, 179,584 calls
- `eval_increment()`: 1.38s, 89,792 calls
- Implements Bouc-Wen smooth hysteresis model with degradation

**Why it's hot**:
- Python loop for Bouc-Wen ODE integration (small substeps if needed)
- N-M interaction via `moment_capacity_from_polygon()` (2.20s, 20,109 calls)
- Not vectorized

**Categorization**: Python loop/dispatch overhead.

---

### Hotspot #5: Linear solve (0.90s, 1.2%)

**Evidence**:
- `numpy.linalg.solve()`: 14,881 calls, 0.90s
- Called ~1.5 times per Newton iteration (likely due to some retries/corrections)
- Problem size: 24 DOF → 24×24 dense system

**Why it's NOT a bottleneck**:
- NumPy delegates to LAPACK DGESV (highly optimized)
- Small problem size (24×24) solves in ~60µs per call
- **Less than 1.2% of runtime**

**Conclusion**: Linear solve optimization is premature. Focus on assembly and allocation.

---

## Ranked Optimization Options (Evidence-Based)

### Option 1: Preallocate buffers for element stiffness computation

**Target**: Reduce allocation pressure (3.68s → <0.5s, ~3s savings, 4% speedup)

**Mechanism**:
- Preallocate reusable buffers for:
  - 6×6 local stiffness matrix
  - 2×2 rotation matrix
  - 6×6 global stiffness matrix
  - Geometry scratch space (length, unit vectors)
- Pass buffers to `k_local()`, `rot2d()`, `stiffness_and_force_global()`
- Reduces 2.5M allocations to O(n_elem) = 18

**Implementation**:
- Add `_k_local_buf`, `_rot_buf`, `_k_global_buf` as class members in `FrameElementLinear2D`
- Modify function signatures to accept optional `out=` parameter (NumPy-style)
- **Minimal diff**: ~20 lines in `frame2d.py`

**Expected speedup**: 3-4s (4-5%)

**Risk**: Low (pure memory optimization, no algorithm change)

**Correctness check**:
- Compare final displacement vector norm: must match within 1e-10 relative
- Compare energy residuals: must match within 1e-6 relative

---

### Option 2: Numba JIT for `Model.assemble()` aggregation loop

**Target**: Reduce Python loop overhead in assembly (25.05s → ~10s, ~15s savings, 19% speedup)

**Mechanism**:
- Extract the hot inner loop (beam + hinge aggregation into global K/R) into a standalone function
- Apply `@numba.jit(nopython=True)` to the aggregation loop
- Requires:
  - Contiguous float64 arrays for K, R, dofs
  - No Python objects (dicts, lists) inside the JIT kernel
  - Explicit indexing (no fancy slicing)

**Implementation**:
- Create `dc_solver/kernels/assemble_jit.py` with:
  ```python
  @numba.jit(nopython=True)
  def aggregate_element_contributions(K_global, R_global, K_local, R_local, dofs):
      for i in range(len(dofs)):
          for j in range(len(dofs)):
              K_global[dofs[i], dofs[j]] += K_local[i, j]
          R_global[dofs[i]] += R_local[i]
  ```
- Call from `Model.assemble()` inside the element loop

**Expected speedup**: 10-15s (13-19%)

**Risk**: Medium
- Numba compilation overhead on first call (~1-2s)
- Requires careful handling of array layouts (must be contiguous)
- May not work with all NumPy dtypes/shapes without tuning

**Correctness check**:
- Same as Option 1

---

### Option 3: Compiled kernel (Fortran f2py) for element stiffness

**Target**: Replace `k_local()` + `rot2d()` + `stiffness_and_force_global()` with compiled kernel (19.4s → ~5s, ~14s savings, 18% speedup)

**Mechanism**:
- Write Fortran subroutine:
  ```fortran
  subroutine frame_stiffness_global(E, A, I, L, cos_theta, sin_theta, K_global, f_global)
    ! Compute 6x6 local stiffness, apply rotation, return global contributions
  ```
- Compile via f2py: `python -m numpy.f2py -c -m frame_kernels frame_stiffness.f90`
- Call from Python: `frame_kernels.frame_stiffness_global(...)`

**Implementation**:
- Create `plastic_hinge/kernels/frame_stiffness.f90`
- Modify `FrameElementLinear2D` to use compiled kernel if available (guarded by `DC_FAST=1`)
- Fallback to pure Python if not compiled

**Expected speedup**: 10-15s (13-19%)

**Risk**: High
- Requires Fortran compiler (gfortran)
- Build system complexity (f2py integration)
- Harder to debug/maintain
- **Only justified if Numba fails or is unavailable**

**Correctness check**:
- Same as Option 1
- Must verify contiguous memory layout (Fortran column-major vs NumPy row-major)

---

### Option 4: Reduce postprocessing overhead (OPTIONAL, not computational)

**Target**: Eliminate plotting inside solve loop (25s → 0s, 32% speedup)

**Mechanism**:
- Add `--no-plot` or `DC_NO_PLOT=1` flag to skip all matplotlib calls
- Defer plotting to a separate post-run script that reads saved state

**Implementation**:
- Wrap all `plot_*()` calls with `if not args.no_plot:`
- Save raw state (u, v, a, hinge history) to HDF5 or NPZ files
- Create `tools/post_plot.py` to regenerate plots offline

**Expected speedup**: 25s (32%) — but this is NOT computational optimization, it's workflow optimization

**Risk**: None (pure I/O elimination)

**Correctness check**: Not applicable (no numerical changes)

---

## Recommended Sequence

**Phase 4 (immediate)**: Implement **Option 1** (preallocate buffers)
- Minimal diff (~20 lines)
- Low risk
- Proves the concept of guarded optimization
- Expected: 3-4s savings (4-5% speedup)

**Phase 5 (if approved)**: Implement **Option 2** (Numba JIT for assemble)
- Medium risk
- Requires Numba dependency (already in `pyproject.toml` as optional)
- Expected: 10-15s savings (13-19% speedup)
- **Guard with `DC_FAST=1` environment variable**

**Phase 6 (defer)**: Consider **Option 3** (Fortran f2py) ONLY if:
- Numba is unavailable or fails
- Profiling shows Option 2 didn't achieve expected speedup
- Customer explicitly requests compiled kernels

**Phase 7 (optional)**: Implement **Option 4** (skip plotting) for production runs
- Not an optimization, but a workflow improvement
- Should be user-facing flag, not guarded

---

## Correctness Verification Protocol

For ANY optimization, before/after comparison must show:

1. **Displacement invariants**:
   - Final displacement vector norm: `||u_final|| == ||u_baseline||` within 1e-10 relative
   - Peak drift: `max(drift) == baseline_drift` within 1e-10 relative

2. **Force invariants**:
   - Base shear history: `Vb(t) == baseline_Vb(t)` within 1e-8 relative
   - Hinge moment-rotation curves: overlay must be identical (visual + numerical)

3. **Energy invariants**:
   - Energy residual: `|E_final - E_baseline| / E_baseline < 1e-6`

4. **Iteration counts**:
   - Newton iterations per step: must match exactly (same convergence path)
   - If iteration counts differ, optimization changed the algorithm → FAIL

5. **Profiling comparison**:
   - Run with same tag suffix: `baseline_hht_ida_v1` vs `opt1_hht_ida_v1`
   - Compare wall time, cumulative time for top 10 functions
   - Hotspot must shift (e.g., `Model.assemble()` tottime must decrease)

---

## Deliverables for Phase 4

1. **Code changes**:
   - `src/dc_solver/fem/frame2d.py`: Add preallocated buffer support
   - Guard with `DC_FAST=1` or `--fast-kernels` flag

2. **Profiling comparison**:
   - Run: `DC_FAST=0 python tools/profile_run.py --tag baseline_opt1_before --state ida --ag-min 0.1 --ag-max 0.2 --ag-step 0.1`
   - Run: `DC_FAST=1 python tools/profile_run.py --tag baseline_opt1_after --state ida --ag-min 0.1 --ag-max 0.2 --ag-step 0.1`
   - Diff: `diff -u outputs/profiling/baseline_opt1_before/REPORT.md outputs/profiling/baseline_opt1_after/REPORT.md`

3. **Correctness check**:
   - Extract final state norms from both runs
   - Compare within stated tolerances
   - Document in `OPTIMIZATION_PLAN.md` (append results section)

---

**Next action**: Await approval for Phase 4 (Option 1: preallocate buffers).
