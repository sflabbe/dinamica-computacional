# PHASE 2: Evidence-Based Hotspot Analysis

**Baseline**: `baseline_ida_minimal`
**Date**: 2025-12-23
**Workload**: IDA with 2 runs (0.1g, 0.2g), HHT integrator, SHM hinges
**Total runtime**: 82.92 s
**Environment**: Single-threaded (OMP/MKL/OPENBLAS/NUMEXPR all set to 1)

---

## Top 20 Hotspots by Total Time (Self Time)

| Rank | Function | Location | tottime (s) | % | cumtime (s) | ncalls | Category |
|------|----------|----------|-------------|---|-------------|--------|----------|
| 1 | `assemble` | model.py:53 | 25.44 | 30.7% | 49.36 | 34,895 | NumPy aggregation |
| 2 | `stiffness_and_force_global` | frame2d.py:131 | 10.28 | 12.4% | 24.12 | 989,190 | NumPy kernel |
| 3 | `k_local` | frame2d.py:59 | 5.02 | 6.1% | 6.87 | 989,388 | NumPy kernel |
| 4 | `numpy.array` | (builtin) | 2.68 | 3.2% | 2.68 | 2,581,696 | Allocation |
| 5 | `eval_trial` | models.py:620 | 2.50 | 3.0% | 7.87 | 179,600 | Fiber hinge (Python) |
| 6 | `rot2d` | frame2d.py:14 | 2.27 | 2.7% | 4.08 | 989,676 | NumPy kernel |
| 7 | `moment_capacity_from_polygon` | models.py:15 | 2.21 | 2.7% | 2.77 | 20,113 | Python loop |
| 8 | `_geom` | frame2d.py:43 | 2.13 | 2.6% | 2.49 | 1,979,100 | NumPy kernel |
| 9 | `hht_alpha_newton` | hht_alpha.py:125 | 1.66 | 2.0% | 67.47 | 2 | Integrator (orchestration) |
| 10 | `eval_increment` (SHM) | models.py:179 | 1.36 | 1.6% | 4.07 | 89,800 | **Python loop (Bouc-Wen)** |
| 11 | `base_shear` | model.py:121 | 1.11 | 1.3% | 7.12 | 10,003 | Postprocessing |
| 12 | `numpy.zeros` | (builtin) | 1.07 | 1.3% | 1.07 | 2,109,568 | Allocation |
| 13 | `dofs` | frame2d.py:51 | 0.92 | 1.1% | 1.61 | 989,676 | Array indexing |
| 14 | `solve` | linalg | 0.91 | 1.1% | 1.18 | 14,881 | Linear algebra |
| 15 | `_My0_base` | models.py:140 | 0.72 | 0.9% | 1.07 | 449,519 | SHM helper (Python) |
| 16 | `_bw_rhs` | models.py:165 | 0.49 | 0.6% | 0.56 | 451,076 | **SHM Bouc-Wen RHS** |
| 17 | `update_column_yields` | model.py:41 | 0.25 | 0.3% | 7.13 | 10,055 | N-M polygon lookup |
| 18 | `eval_increment` (column) | models.py:61 | 0.20 | 0.2% | 0.24 | 89,800 | Column hinge |
| 19 | `_degraded_My` | models.py:173 | 0.30 | 0.4% | 0.80 | 180,119 | SHM degradation |
| 20 | `_degraded_K0` | models.py:168 | 0.26 | 0.3% | 0.32 | 180,119 | SHM degradation |

---

## Detailed Analysis of Top 3 Hotspots

### #1: `Model.assemble()` — 25.44s (30.7%)

**Location**: model.py:53
**Call count**: 34,895
**Who calls it**: `hht_alpha_newton` (every Newton iteration, every timestep)

**Why it's hot**:
- Aggregates element stiffness matrices into global system matrix
- Pure Python loop over elements with NumPy array indexing
- High call count (34,895 calls = ~5000 timesteps × ~7 Newton iterations/step × 2 runs)
- Python dispatch overhead dominates (tottime 25s vs cumtime 49s → ~24s in callees)

**Evidence**:
```
ncalls: 34,895
tottime: 25.44s (self time)
cumtime: 49.36s (includes stiffness_and_force_global: 24.12s)
Per-call overhead: 0.73 ms/call (mostly Python dispatch)
```

**Categorization**: NumPy aggregation with Python dispatch overhead

---

### #2: `FrameElementLinear2D.stiffness_and_force_global()` — 10.28s (12.4%)

**Location**: frame2d.py:131
**Call count**: 989,190
**Who calls it**: `Model.assemble()` (once per element per assembly)

**Why it's hot**:
- Transforms local element stiffness to global coordinates
- NumPy matrix operations but called ~1M times
- Calls `k_local()` (5.02s) and `rot2d()` (2.27s)
- Pure NumPy but Python function call overhead significant

**Evidence**:
```
ncalls: 989,190
tottime: 10.28s (self time)
cumtime: 24.12s (includes k_local: 6.87s, rot2d: 4.08s)
Per-call overhead: 0.010 ms/call
```

**Categorization**: NumPy kernel with Python overhead

---

### #3: `FrameElementLinear2D.k_local()` — 5.02s (6.1%)

**Location**: frame2d.py:59
**Call count**: 989,388
**Who calls it**: `stiffness_and_force_global()`

**Why it's hot**:
- Computes local 6×6 stiffness matrix (Euler-Bernoulli beam)
- Called ~1M times (once per element per assembly)
- Pure NumPy but small array operations (6×6 matrix)
- Python function call overhead is ~50% of total time

**Evidence**:
```
ncalls: 989,388
tottime: 5.02s (self time)
cumtime: 6.87s
Per-call overhead: 0.005 ms/call
```

**Categorization**: NumPy kernel (small matrix operations)

---

## SHM Hinge Hotspots (Option 3 Target)

**Total SHM-related time**: ~3.2s (3.9% of total runtime)

### Components:

| Function | tottime (s) | cumtime (s) | ncalls | Category |
|----------|-------------|-------------|--------|----------|
| `eval_increment` (SHM) | 1.36 | 4.07 | 89,800 | Bouc-Wen integration loop |
| `_My0_base` | 0.72 | 1.07 | 449,519 | Axial-moment interaction |
| `_bw_rhs` | 0.49 | 0.56 | 451,076 | Bouc-Wen RHS evaluation |
| `_degraded_My` | 0.30 | 0.80 | 180,119 | Degradation (exponential) |
| `_degraded_K0` | 0.26 | 0.32 | 180,119 | Degradation (exponential) |
| `_eref` | 0.15 | 0.45 | 89,800 | Energy reference |

**Evidence for JIT optimization**:
- `eval_increment` contains substep loop (RK4 integration) with 451k calls to `_bw_rhs`
- All operations are float arithmetic (no objects, no strings, no I/O)
- Contiguous NumPy arrays or scalar floats only
- **Good JIT target**: tight numeric loop, no Python objects

**Expected speedup**: 2-4× on SHM component (Numba typically achieves 3-10× on tight Python loops)
**Wall-time impact**: 3.2s → 0.8-1.6s (save 1.6-2.4s, or 2-3% of total runtime)

---

## N-M Polygon Lookup Hotspots

| Function | tottime (s) | ncalls | Category |
|----------|-------------|--------|----------|
| `moment_capacity_from_polygon` | 2.21 | 20,113 | Python loop (polygon intersection) |
| `update_column_yields` | 0.25 | 10,055 | Column hinge N-M update |

**Evidence**:
- `moment_capacity_from_polygon`: 2.21s over 20k calls = 0.11 ms/call
- Pure Python loop over polygon vertices
- **Optimization potential**: Cache results (N values repeat), or JIT the intersection logic

---

## Postprocessing Hotspots (Inside Timestep Loop!)

| Function | tottime (s) | cumtime (s) | ncalls | Why problematic |
|----------|-------------|-------------|--------|-----------------|
| `base_shear` | 1.11 | 7.12 | 10,003 | Called every timestep; should be post-only |
| `update_column_yields` | 0.25 | 7.13 | 10,055 | N-M lookup every timestep |

**Evidence**: `base_shear()` is called 10,003 times (approximately every timestep for monitoring). This is postprocessing inside the hot loop.

**Recommendation**: Move `base_shear` logging out of the critical path if not needed for convergence.

---

## Hotspot Categorization Summary

| Category | Functions | Total time (s) | % of runtime |
|----------|-----------|----------------|--------------|
| NumPy aggregation (Python dispatch) | assemble | 25.44 | 30.7% |
| NumPy kernels (element operations) | stiffness_and_force_global, k_local, rot2d, _geom | 19.70 | 23.8% |
| Python loops (SHM Bouc-Wen) | eval_increment, _bw_rhs, _My0_base, degraded_* | 3.13 | 3.8% |
| Python loops (N-M polygon) | moment_capacity_from_polygon | 2.21 | 2.7% |
| Postprocessing (inside loop) | base_shear, update_column_yields | 1.36 | 1.6% |
| Allocation pressure | numpy.array, numpy.zeros | 3.75 | 4.5% |
| Linear algebra | solve | 0.91 | 1.1% |
| Matplotlib (plotting) | (various) | ~12 | 14.5% |

**Total measured hotspots**: 68.5s (82.6% of 82.92s runtime)

---

## Conclusions

1. **Dominant hotspot**: `Model.assemble()` at 30.7% is the #1 target for optimization (Option 2: Numba JIT for aggregation loop)

2. **Second tier**: Frame element kernels (`stiffness_and_force_global`, `k_local`) at 23.8% combined — optimization requires Fortran/C or very tight Numba

3. **Option 3 target (SHM hinges)**: 3.8% of runtime, but:
   - Good JIT candidate (pure numeric loop)
   - No architectural changes needed
   - Expected 2-4× speedup on this component (save 1.6-2.4s wall time)

4. **Low-hanging fruit**:
   - Postprocessing inside timestep loop (`base_shear`): 1.3% (1.1s)
   - N-M polygon caching: 2.7% (2.2s)

**Recommendation**: Proceed with Option 3 (SHM hinge JIT) as a proof-of-concept for Numba integration pattern, then apply same pattern to Option 2 (assemble JIT) if successful.
