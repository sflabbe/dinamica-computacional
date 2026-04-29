# Dinámica Computacional — 2D frame solver (plastic hinges + fiber sections)

This repository contains a compact 2D **frame** (beam/column) solver aimed at reproducible structural dynamics experiments.

## What this repo is
2D frame solver for structural dynamics experiments.

## What this repo is not
Not a certifiable structural design package. Not a replacement for EC3/EC8/EC9 checks.

## Installation and usage (Python 3.10+)

The authoritative Python requirement is defined in `pyproject.toml`: **Python 3.10+**.

### Recommended workflow: `uv` (Linux/macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras --dev
```

Run fast tests (CI-like path, JIT disabled):

```bash
DC_USE_NUMBA=0 PYTHONPATH=src:. uv run pytest -q -m "not slow"
```

Run a CLI example:

```bash
PYTHONPATH=src:. uv run python -m problems.problema4_portico --beam-hinge shm --gravity --line-search
```

Run the Streamlit app:

```bash
uv run --extra app streamlit run app/main.py
```

### Recommended workflow: `uv` (Windows PowerShell)

```powershell
winget install --id=astral-sh.uv -e
uv sync --all-extras --dev
```

Run fast tests (CI-like path, JIT disabled):

```powershell
$env:DC_USE_NUMBA="0"; $env:PYTHONPATH="src;."; uv run pytest -q -m "not slow"
```

Run a CLI example:

```powershell
$env:PYTHONPATH="src;."; uv run python -m problems.problema4_portico --beam-hinge shm --gravity --line-search
```

Run the Streamlit app:

```powershell
uv run --extra app streamlit run app/main.py
```

### `cmd.exe` equivalents (where useful)

```cmd
set DC_USE_NUMBA=0
set PYTHONPATH=src;.
uv run pytest -q -m "not slow"
```

### Conservative fallback: `pip` from `pyproject.toml`

Use this only if `uv` is not available in your environment. This fallback installs from project metadata (not from `requirements.txt`).

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[numba,app]"
python -m pip install pytest
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[numba,app]"
python -m pip install pytest
```

Windows `cmd.exe`:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[numba,app]"
python -m pip install pytest
```

Test commands with `pip` environments:

- Linux/macOS:
  ```bash
  DC_USE_NUMBA=0 PYTHONPATH=src:. pytest -q -m "not slow"
  ```
- Windows PowerShell:
  ```powershell
  $env:DC_USE_NUMBA="0"; $env:PYTHONPATH="src;."; pytest -q -m "not slow"
  ```
- Windows `cmd.exe`:
  ```cmd
  set DC_USE_NUMBA=0
  set PYTHONPATH=src;.
  pytest -q -m "not slow"
  ```

### Optional dependencies

- `numba` extra: enables optional JIT acceleration when available.
- `app` extra: installs Streamlit/pandas dependencies for `app/main.py`.

JIT is optional by design. For deterministic fast test paths (local or CI), set `DC_USE_NUMBA=0`.

## Architecture
core: src/dc_solver  
legacy/research: plastic_hinge  
examples/problems: src/problems  
ui: app

## Evidence level
Implemented and tested / partial / placeholder.

## Features

- **Coupled N–M plastic hinge** (polygonal yield surface + return mapping)
- **SHM beam hinge** (degrading hysteresis model for beam ends)
  - The SHM implementation auto-scales the Bouc–Wen parameter `A` (when `bw_A <= 0`) so the **elastic-range tangent** is
    approximately **K0** (otherwise the hinge can become unrealistically soft in gravity).
- **RC fiber-section hinge** (2D fiber discretization) used to generate N–M interaction and realistic hysteresis
- Time integration (CLI-selectable):
  - `hht` (HHT-α, default)
  - `newmark` (Newmark-β)
  - `explicit` (Velocity Verlet, with automatic stability substepping)

Outputs are written under `./outputs/` by default.

> For convergence issues (Newton / dt cutbacks / fiber hinge), see **`DEBUG_CHECKLIST.md`**.

## Typical balcony frame plot

Downstream automation can create a deterministic PNG of a typical balcony
frame without building a finite-element model:

```python
from dc_solver.post import TypicalBalconyFrameSpec, plot_typical_balcony_frame

spec = TypicalBalconyFrameSpec(
    floors=4,
    bay_width_m=4.5,
    story_height_m=3.0,
    balcony_depth_m=1.6,
    facade_x_m=0.0,
    show_node_labels=True,
)

plot_typical_balcony_frame(spec, "outputs/typical_balcony_frame.png")
```

The plotting helper uses Matplotlib's `Agg` backend and writes PNG files
without requiring a display server.  Segment geometry is also available through
`typical_balcony_frame_segments(spec)` for adapters that need coordinates.
`facade_x_m` defines the facade grid line; the interior bay extends to the left
and balconies extend to positive x.

---

## Development setup

`uv` is the source of truth for local environments and lockfile management.
`requirements.txt` is not used.

### Standard setup

```bash
uv sync --all-extras --dev
DC_USE_NUMBA=0 PYTHONPATH=src:. uv run pytest -q -m "not slow"
```

Alternative via `make`:

```bash
make sync
make test-fast
```

### Lockfile and dependencies

```bash
uv lock
uv lock --check
uv add numpy
uv add --dev pytest
uv add --optional numba "numba>=0.61"
```

The optional `numba` extra remains optional. The CI fast path sets
`DC_USE_NUMBA=0` and runs tests without requiring JIT.

## Run the problems

### Problem 2 — N–M hinge verification

```bash
python src/problems/problema2_interaccion.py
python src/problems/problema2_hinge_nm_verification.py
```

Typical exports:
- N–M paths
- M–θ and N–ε hysteresis
- `_gradient.png` versions with a step/time color gradient

### Problem 4 — portal frame (gravity preload + dynamic IDA)

**Default mode is `--state ida`**: the script runs **Step 1** gravity preload, then **Step 2** incremental dynamic analysis (IDA).

```bash
PYTHONPATH=src python -m problems.problema4_portico --integrator hht --beam-hinge fiber
```

Modes:
- `--state gravity` (or `--gravity`): Step 1 only (gravity preload)
- `--state ida` (default): Step 1 + Step 2 (IDA)

Gravity only:

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --gravity
```

When you run gravity-only, the script writes `gravity_compare.txt` which now includes a **gravity load check**:
- confirms the gravity load is a **distributed self-weight on all frame elements (columns + roof beam)**, computed from the **physical member areas** (A_col/A_beam) and member lengths (same as the Problem 6 elastic reference)
- prints `sum(Fy)+P` as a quick verification (should be ~0)

For fiber beam hinges, a small debug file is also written:
- `fiber_hinge_axial_debug.txt` (axial-force coupling used by the fiber hinges: `N_beam_tension` and `N_target_used`)

Compare SHM vs fiber hinge (runs both):

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge compare
```

Key options:
- `--integrator {hht,newmark,explicit}`
- `--nlgeom` enables geometric nonlinearity (P-Δ)
- IDA amplitude range (in g): `--ag-min`, `--ag-max`, `--ag-step`
- Time-step controls: `--base-dt`, `--dt-min`
- Newton controls: `--max-iter`, `--tol`

Notes:
- All frame (beam/column) elements use a **Timoshenko** formulation by default.
- With `--nlgeom`, a **P-Δ (geometric stiffness)** contribution is included consistently in both the tangent stiffness
  and the internal force evaluation.
- For `--integrator explicit`, the solver estimates a critical explicit step (`dt_crit`) from the current tangent and
  automatically substeps each output step (reported as `dt_sub`, `n_substeps`) when needed.

SHM hinge note:
- The SHM beam hinge auto-scales the Bouc–Wen parameter `A` (when `bw_A <= 0`) so the **initial tangent stiffness** is
  approximately the desired elastic stiffness `K0` (avoids an overly soft elastic range).

#### Line search (recommended for fiber + implicit)

Backtracking line search is **disabled by default** and is only enabled when you pass `--line-search`.
When enabled, it applies to:
- Step 1 gravity preload (Newton)
- Step 2 implicit dynamics (HHT/Newmark Newton corrector)
- Fiber hinge local solve (the internal N≈0 constraint solve)

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --integrator hht --line-search
```

Fiber discretization (only used when `--beam-hinge fiber` or `compare`):

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --fiber-ny 24 --fiber-nz 18
```

### Problem 5 — RC fiber section → N–M interaction

```bash
PYTHONPATH=src python -m problems.problema5_fiber_section_interaction
```

---

## Outputs and logs (Abaqus-like)

Each run writes into its output directory:

- `journal.log` — chronological log (console + key events)
- `<jobname>.msg` — message file (warnings/errors, exceptions)
- `<jobname>.sta` — status file (step progress)
- `<jobname>.dat` — data summary file
- `run_info.json` / `run_info.txt` — run metadata snapshot

The console prints `JOB START` / `JOB END` blocks including wall/CPU time, FLOPs estimate, and a list of newly created files.

See: `examples/demo_job_infrastructure.py`.

---

## Troubleshooting (quick)

### “COLLAPSE (dt<dtmin)” during IDA

This means the analysis had to cut back the time step repeatedly (non-convergence) until `dt` fell below `--dt-min`.

Try (in this order):

1) Enable line search:

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --integrator hht --line-search
```

2) Increase Newton iterations:

```bash
... --max-iter 80
```

3) Start with a smaller base dt and/or allow smaller dt-min:

```bash
... --base-dt 0.001 --dt-min 0.000125
```

4) Print cutback diagnostics:

```bash
... --debug-cutback
```

For a full step-by-step checklist, see **`DEBUG_CHECKLIST.md`**.


### “Gravity-only Newton did not converge”

If you run gravity preload only:

```bash
PYTHONPATH=src python -m problems.problema4_portico --beam-hinge fiber --gravity
```

and you get a non-convergence error, the solver will automatically attempt **adaptive load substepping** (smaller gravity increments) by default.
If it still fails, try (in this order):

1) Enable line search:

```bash
... --line-search
```

2) Increase gravity iterations:

```bash
... --gravity-max-iter 120
```

3) Ramp gravity more smoothly:

```bash
... --gravity-steps 20
```

4) Enable verbose gravity output to see the substepping and residuals:

```bash
... --gravity-verbose
```

### Numba / JIT issues

Optional speedups exist behind an environment variable.
To disable JIT during debugging:

```bash
DC_USE_NUMBA=0 PYTHONPATH=src python -m problems.problema4_portico --integrator explicit
```

---

## Repository layout (high level)

- `plastic_hinge/` : N–M hinge return mapping + fiber section tools
- `src/dc_solver/` : FEM core, integrators, post-processing, job/reporting
- `src/problems/`  : reproducible problem scripts
