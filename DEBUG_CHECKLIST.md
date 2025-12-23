# Debug checklist (Dinámica Computacional)

This checklist is meant to make failures *actionable*: when something diverges or collapses (e.g., `dt < dtmin`), you can quickly identify whether the problem is **modeling**, **numerics**, or **implementation**.

> Tip: when you report an issue, always attach (or paste) these files from the output directory:
> - `journal.log`
> - `<job>.msg`, `<job>.sta`, `<job>.dat`
> - `run_info.json` / `run_info.txt`
> - the CLI command used

---

## 0) One-minute triage

1) **Reproduce with the simplest setting**
   - Switch to `--beam-hinge shm` (cheaper + usually more robust)
   - Use `--integrator explicit` (no Newton; isolates modeling issues)

2) **Check gravity only first (Step 1)**
   - `--state gravity` or `--gravity`
   - If gravity is already non-convergent, Step 2 will not behave.

3) **If implicit fails** (HHT/Newmark), enable line search
   - `--line-search` applies to *gravity + dynamics + fiber local solves*

4) **If IDA says** `COLLAPSE (dt<dtmin)`
   - That is a *numerical* collapse (repeated cutbacks). It does **not necessarily** mean structural collapse.

---

## 1) Collect the minimum debug package

From the run output directory (printed at the end), collect:

- `journal.log` (full timeline)
- `<job>.msg` (exceptions + warnings)
- `<job>.sta` (step/time/iteration summary)
- `<job>.dat` (tables, sanity numbers)
- `run_info.json` + `run_info.txt` (metadata)
- Any produced `.csv` or `.png` that looks suspicious

Then record:

- OS + Python version
- Whether you ran with JIT (`DC_USE_NUMBA=1`) or without (`DC_USE_NUMBA=0`)
- Exact CLI command

---

## 2) Problem 4 specifics (gravity preload + IDA)

### Step 1: Gravity preload

Run:

```bash
PYTHONPATH=src python -m problems.problema4_portico --state gravity --beam-hinge fiber --gravity-verbose
```

Checks:

- **Verify the applied gravity load** is what you think it is:
  - In the output directory, open `gravity_compare.txt`.
  - Look for `Gravity load check` and confirm `sum(Fy)+P` is ~0.
  - This should match between Problem 4 and the Problem 6 elastic reference.

- If you are using `--beam-hinge fiber`, inspect `fiber_hinge_axial_debug.txt`:
  - `N_beam_tension` is the frame element axial force (tension-positive convention)
  - `N_target_used` is what the fiber hinge sees (compression-positive convention)
  - Large tension in the beam (negative `N_target_used`) can strongly change the section response.

- Gravity should converge in **all load steps**.
- The gravity solver uses **adaptive load substepping** by default: if a step does not converge, the load increment is halved and retried.
  Use `--gravity-verbose` to see the substepping decisions and residual norms.
- If the roof drift is wildly different between `--beam-hinge shm` and `--beam-hinge fiber`, suspect:
  - fiber hinge axial constraint solve (local `eps0` solve),
  - sign convention in the hinge moment/rotation,
  - too-stiff initial tangent / singular tangent.

- If the gravity case with `--beam-hinge shm` is **much more flexible** than the linear reference (Problem 6):
  - In SHM, the elastic-range stiffness must be **K0**, not **My**.
  - This repo auto-scales the Bouc–Wen parameter `A` so that the initial tangent satisfies `dM/dθ|_0 ≈ K0`.
  - If you changed SHM parameters manually, ensure `bw_A <= 0` (auto) or set `bw_A ≈ K0/My`.
  - Pinching should not reduce the first monotonic branch from zero; pinching is applied only on reloading toward the origin.

Suggested actions:

- Enable line search:

```bash
... --line-search
```

- Increase gravity iterations:

```bash
... --gravity-max-iter 120
```

- Ramp gravity more smoothly:

```bash
... --gravity-steps 20
```

### Step 2: Dynamic / IDA

Default run (Step 1 + Step 2):

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --integrator hht
```

If you see `COLLAPSE (dt<dtmin)`:

- Start with line search:

```bash
... --line-search
```

- Make the dynamic Newton easier:

```bash
... --max-iter 80 --tol 1e-6
```

- Reduce initial dt and/or allow smaller dt-min:

```bash
... --base-dt 0.001 --dt-min 0.000125
```

- Print cutback diagnostics:

```bash
... --debug-cutback
```

---

## 3) Modeling sanity checks

### Units

- Confirm that geometry, mass, and stiffness are consistent (SI is assumed in most scripts).
- If you change any of: section dimensions, E, density, g, record amplitude, time scaling — redo the checks below.

### Boundary conditions and constraints

- If the solver reports singular matrices, check supports, released DOFs, and hinge element assembly.
- For multi-point constraints (if used), verify you are not constraining the same DOF twice.

### Beam formulation (Timoshenko + P-Δ)

- Frame elements are **Timoshenko beams** by default (shear deformation included). If results look too stiff/soft, verify
  the cross-section properties (A, I) and Poisson's ratio (affects G) are reasonable.
- With `--nlgeom`, the solver includes a **P-Δ (geometric stiffness)** contribution. For Newton-type solvers, it is
  important that the **internal force** and the **tangent** are consistent. If you see oscillatory Newton behavior or
  repeated cutbacks only when `--nlgeom` is enabled, try reproducing with `--nlgeom` off and check the axial force levels
  (near-buckling can trigger strong P-Δ sensitivity).

### Mass sanity

In Problem 4, the script prints a mass sanity block. Watch for:

- negative masses
- missing masses in free DOFs

---

## 4) Numerical sanity checks

### Use explicit as a modeling baseline

Explicit does not require Newton iterations. If explicit behaves but HHT/Newmark fails, the issue is likely:

- Newton robustness (line search helps)
- inconsistent tangent stiffness (fiber hinge tangent / plasticity tangent)
- too aggressive time step

Run:

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --integrator explicit
```

Note: the explicit integrator automatically applies **stability substepping** when the chosen `dt` is larger than the
estimated critical explicit step (reported as `dt_crit_est`). In the returned results and `run_info.json`, look for:
- `dt` (output step)
- `dt_sub` (actual integration substep)
- `n_substeps` (substeps per output step)

### HHT-α parameter

- Default `--alpha -0.05` is mildly dissipative.
- If you see high-frequency chatter, try slightly more dissipation:

```bash
... --alpha -0.10
```

### Time step strategy

- If you repeatedly cut back, start with a smaller `--base-dt`.
- If you always hit dt-min, either allow smaller `--dt-min` or fix the underlying convergence issue (line search, more iterations, better tangents).

### Tolerances

- Too tight tolerances can force cutbacks.
- Too loose tolerances can allow drift and blow up later.

A reasonable starting point for Problem 4 fiber + HHT:

- `--tol 1e-6` to `1e-5`
- `--max-iter 50` to `80`

---

## 5) Fiber hinge troubleshooting

### Symptoms

- Gravity converges with SHM, but not with fiber.
- Large gravity displacement offset compared to elastic reference.
- Implicit dynamics cuts back aggressively right after yielding.

### Actions (in order)

1) **Enable line search**

```bash
... --line-search
```

2) **Coarsen fiber mesh** (to reduce local nonlinearities while debugging)

```bash
... --fiber-ny 12 --fiber-nz 10
```

3) **Switch to explicit** to check if the model is fundamentally unstable

```bash
... --integrator explicit
```

4) **Reduce dt**

```bash
... --base-dt 0.001 --dt-min 0.000125
```

5) **Compare SHM vs fiber** (same record, same settings)

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge compare --integrator hht --line-search
```

---

## 6) Performance / JIT

- Some plasticity routines can use Numba.
- During debugging, it can be useful to disable JIT to get cleaner tracebacks:

```bash
DC_USE_NUMBA=0 PYTHONPATH=src python -m problems.problema4_portico --integrator explicit
```

---

## 7) When you think it is a bug

Before changing code, try to isolate whether the failure is deterministic:

1) Run the same command twice.
2) Run with `DC_USE_NUMBA=0` and `DC_USE_NUMBA=1`.
3) Run with `--integrator explicit`.
4) Run with `--beam-hinge shm`.

If the failure only happens in one mode, you’ve narrowed the bug surface dramatically.
