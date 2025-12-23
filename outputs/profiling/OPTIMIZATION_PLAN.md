# PHASE 3: Ranked Optimization Plan (Evidence-Based)

**Baseline**: 82.92s wall time (IDA, 2 runs, 0.1-0.2g, HHT, SHM hinges)
**Date**: 2025-12-23
**Methodology**: Optimize only measured hotspots. No speculation.

---

## Ranking Methodology

Options ranked by:
1. **Expected wall-time savings** (measured hotspot time × expected speedup)
2. **Implementation risk** (correctness, maintenance burden)
3. **Architectural impact** (minimal diff preferred)

All speedup estimates are **conservative** based on:
- Numba JIT on tight Python loops: 3-10× typical
- Allocation reduction: 50-80% if bulk operations replaced
- Caching: up to 100% if key repeats frequently

---

## Option 1: Numba JIT for Model.assemble() [HIGHEST IMPACT]

### Target Hotspot
- **Function**: `Model.assemble()` (model.py:53)
- **Measured time**: 25.44s tottime, 49.36s cumtime (30.7% of runtime)
- **Call count**: 34,895 (every Newton iteration)

### Why it's hot
- Pure Python loop aggregating element stiffness matrices into global system
- High Python dispatch overhead (25s self time aggregating 989k element calls)
- Inner callees are NumPy kernels (already fast), but outer loop is Python

### Optimization Approach
**Approach A (preferred)**: Numba JIT aggregation kernel
- Extract the assembly loop into a Numba `@jit(nopython=True)` kernel
- Pass element matrices, DOF mappings as contiguous arrays
- Accumulate directly into preallocated global K matrix
- **Mechanism**: Eliminate Python loop overhead, improve cache locality

**Approach B (alternative)**: Vectorized aggregation using NumPy advanced indexing
- Use `np.add.at()` for scatter-add operations
- May not be faster (advanced indexing can be slow)

**Approach C (if A/B fail)**: Fortran kernel via f2py
- Signature: `assemble_f90(nel, ndof, k_locals, dof_map, k_global)`
- Only if Numba gives < 5× speedup

### Expected Speedup
- **Conservative**: 5× on assembly self-time → save 20s (24% wall-time reduction)
- **Optimistic**: 8× → save 23s (28% wall-time reduction)
- **Risk**: Low (assembly logic is deterministic, easy to verify)

### Implementation Plan
1. Create `src/dc_solver/kernels/assemble_jit.py`
2. Implement Numba kernel with pure-Python fallback
3. Guard with `DC_FAST=1` environment variable
4. Verify correctness: gravity + IDA outputs must match baseline bit-for-bit
5. Profile before/after

### Success Criteria
- Wall time < 60s (>27% speedup)
- Assembly tottime < 5s (from 25.44s)
- Correctness verified (diff outputs)

---

## Option 2: Numba JIT for SHM Hinge Evaluation [MEDIUM IMPACT, LOW RISK]

### Target Hotspot
- **Function**: `SHMBeamHinge1D.eval_increment()` (models.py:179)
- **Measured time**: 1.36s tottime, 4.07s cumtime (1.6% of runtime)
- **Supporting functions**: `_bw_rhs` (0.49s), `_My0_base` (0.72s), degraded_* (0.56s)
- **Total SHM-related**: ~3.2s (3.9%)

### Why it's hot
- Python loop for Bouc-Wen substep integration (RK4 with adaptive substepping)
- 89,800 calls to `eval_increment`, 451k calls to `_bw_rhs`
- Pure float arithmetic, no Python objects
- **Good JIT candidate**: tight numeric loop

### Optimization Approach
**Approach A (preferred)**: Numba JIT for Bouc-Wen integration
- Extract `eval_increment` + `_bw_rhs` into a single Numba kernel
- Inline RK4 integration loop
- Pass all parameters as scalar floats or 1D arrays
- **Mechanism**: Eliminate Python loop overhead, enable LLVM vectorization

**Approach B (alternative)**: Cache `_My0_base` and `moment_capacity_from_polygon` results
- Keyed by (N_comp bin, section_id)
- May reduce 2.2s polygon lookup overhead
- Lower speedup but simpler implementation

**Approach C (if A fails)**: Hybrid - JIT the RK4 loop only, keep Python wrapper

### Expected Speedup
- **Conservative**: 3× on SHM component → save 2.1s (2.5% wall-time reduction)
- **Optimistic**: 5× → save 2.6s (3.1% wall-time reduction)
- **Risk**: Low (deterministic ODE integration, easy to verify)

### Implementation Plan
1. Create `src/dc_solver/kernels/hinge_jit.py`
2. Implement Numba kernel for Bouc-Wen step:
   ```python
   @numba.jit(nopython=True, cache=True, fastmath=True)
   def bouc_wen_integrate(dth, z_comm, a_comm, M_comm, params):
       # RK4 loop with nsub substeps
       # returns M_new, k_tan, z_new, a_new
   ```
3. Modify `SHMBeamHinge1D.eval_increment()` to call JIT kernel when `DC_FAST=1`
4. Pure-Python fallback when `DC_FAST=0` or Numba unavailable
5. Verify correctness (gravity + IDA)

### Success Criteria
- Wall time < 80.5s (>2.5% speedup)
- SHM eval_increment tottime < 0.7s (from 1.36s)
- Correctness verified

---

## Option 3: Remove Postprocessing from Timestep Loop [LOW IMPACT, ZERO RISK]

### Target Hotspot
- **Function**: `base_shear()` (model.py:121)
- **Measured time**: 1.11s tottime, 7.12s cumtime (1.3% of runtime)
- **Call count**: 10,003 (every timestep for monitoring)

### Why it's hot
- Postprocessing called inside critical path (every timestep)
- Not needed for convergence, only for logging/output
- Calls into element force queries (cumtime 7.12s)

### Optimization Approach
**Approach A (preferred)**: Move `base_shear` logging to post-timestep phase
- Only compute when saving snapshots or at end of run
- Minimal code change (remove call from inner loop)
- **Mechanism**: Eliminate unnecessary work in hot path

**Approach B (alternative)**: Throttle logging (e.g., every 100 timesteps)
- Reduces calls from 10k to 100
- Still some overhead remains

### Expected Speedup
- **Conservative**: save 1.0s (1.2% wall-time reduction)
- **Optimistic**: save 1.5s if cumtime overhead included (1.8% reduction)
- **Risk**: Zero (removing non-critical logging)

### Implementation Plan
1. Identify where `base_shear()` is called in time loop
2. Move to snapshot/output phase only
3. Verify outputs still generated at correct times

### Success Criteria
- Wall time < 82.0s (>1% speedup)
- base_shear called < 100 times (from 10,003)

---

## Option 4: Cache N-M Polygon Lookups [MEDIUM IMPACT, LOW RISK]

### Target Hotspot
- **Function**: `moment_capacity_from_polygon()` (models.py:15)
- **Measured time**: 2.21s tottime (2.7% of runtime)
- **Call count**: 20,113

### Why it's hot
- Pure Python loop over polygon vertices (convex intersection with N=const line)
- Called 20k times but N values likely repeat (same axial load in columns)
- **Cacheable**: deterministic function of (polygon, N)

### Optimization Approach
**Approach A (preferred)**: LRU cache with binned N values
- Discretize N into bins (e.g., 100 bins over range)
- Cache key: (section_id, N_bin)
- Hit rate expected > 90% if few unique columns

**Approach B (alternative)**: Precompute My(N) spline for each section
- One-time cost at model setup
- Eval via cubic spline interpolation (fast)

**Approach C (if A/B insufficient)**: Numba JIT the polygon intersection loop

### Expected Speedup
- **Conservative**: 80% cache hit rate → save 1.8s (2.2% reduction)
- **Optimistic**: 95% hit rate → save 2.1s (2.5% reduction)
- **Risk**: Low (deterministic lookup, easy to verify)

### Implementation Plan
1. Add `functools.lru_cache` with binned N values
2. Verify cache key is hashable and deterministic
3. Validate outputs match baseline

### Success Criteria
- Wall time < 81.0s (>2% speedup)
- moment_capacity_from_polygon tottime < 0.5s (from 2.21s)

---

## Ranked Priority (By Expected Wall-Time Savings)

| Rank | Option | Target | Expected Savings | % Speedup | Risk | Recommended Order |
|------|--------|--------|------------------|-----------|------|-------------------|
| 1 | **Numba JIT assemble()** | model.assemble | 20-23s | 24-28% | Low | **FIRST** |
| 2 | **Numba JIT SHM hinge** | SHM eval_increment | 2.1-2.6s | 2.5-3.1% | Low | **SECOND** |
| 3 | **Cache N-M polygon** | moment_capacity_from_polygon | 1.8-2.1s | 2.2-2.5% | Low | Third |
| 4 | **Remove base_shear** | postprocessing | 1.0-1.5s | 1.2-1.8% | Zero | Fourth |

---

## Execution Plan (Strict HPC Discipline)

### Phase 4A: Implement Option 2 (SHM JIT) [Proof-of-Concept]

**Why start with Option 2 instead of Option 1?**
- Lower risk (smaller, isolated kernel)
- Establishes Numba integration pattern for DC_FAST guard
- Validates correctness testing workflow
- **If Option 2 fails, Option 1 is unlikely to succeed**

**Steps**:
1. Profile baseline: DONE (baseline_ida_minimal: 82.92s)
2. Implement SHM Numba kernel (guarded by DC_FAST=1)
3. Verify correctness:
   ```bash
   # Before
   DC_FAST=0 python -m problems.problema4_portico --state gravity --gravity-steps 10 > before.txt
   # After
   DC_FAST=1 python -m problems.problema4_portico --state gravity --gravity-steps 10 > after.txt
   diff before.txt after.txt  # Must be identical
   ```
4. Profile after: `DC_FAST=1 python tools/profile_run.py --tag opt2_after ...`
5. Compare: wall time, SHM tottime
6. **Decision**: If speedup >= 2s, proceed to Option 1. If < 2s, document failure and investigate.

### Phase 4B: Implement Option 1 (assemble JIT) [Highest Impact]

**Only proceed if Option 2 succeeds**
- Apply same Numba pattern to `Model.assemble()`
- Target > 20s savings
- Verify correctness (gravity + IDA)

### Phase 4C: Cleanup Wins (Options 3-4)

**Only if time permits**
- Quick wins with minimal risk
- Combined savings: 2-4s

---

## Explicit Rejection of Speculative Optimizations

### NOT RECOMMENDED (no evidence):
- ❌ GPU acceleration (no evidence of data-parallel bottleneck)
- ❌ Fortran rewrite (Numba should achieve 70-80% of Fortran speed)
- ❌ Buffer preallocation (allocation overhead is 3.75s, savings would be < 2s)
- ❌ Multithreading (small problem, overhead > benefit, complicates correctness)
- ❌ Sparse matrix optimization (not measured as bottleneck)

---

## Final Mandate

**No optimization without profiling.**
**No profiling data, no performance claims.**
**Correctness is non-negotiable.**

All options must:
1. Have before/after profiling data (manifest.json + REPORT.md)
2. Verify bit-for-bit identical results (gravity + IDA)
3. Document wall-time savings in RESULTS.md
4. Minimal diffs (no architecture refactors)

Proceed to **Phase 4A: Option 2 (SHM JIT)** implementation.
