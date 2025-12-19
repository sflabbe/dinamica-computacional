# Plastic hinge N–M with associative flow (return mapping)

Self-contained implementation of a **2D axial force–bending moment (N–M) plastic hinge**
based on a **convex polygonal interaction surface** and a **closest-point / return-mapping** update
(**associative flow**).

## Contents

- `plastic_hinge/return_mapping.py`  
  Closest-point projection to a convex polytope `A s <= b` in 2D using an active-set/KKT approach.
- `plastic_hinge/hinge_nm.py`  
  Plastic hinge state update using a predictor–corrector:
  - elastic predictor `s_trial`
  - plastic corrector by projection (return mapping) in metric `W = K^{-1}`
  - plastic increment consistent with **associated flow** for polyhedral yield surfaces.
- `plastic_hinge/nm_surface.py`  
  Convex hull + half-space representation of an N–M interaction polygon.
- `plastic_hinge/rc_section.py`  
  Simple **rectangular RC** fiber section + quick interaction sampling.
- `plastic_hinge/fiber_section.py`  
  **Generic fiber section** interface (so you can build arbitrary RC geometries with your own fibers).

## Quick demo

```bash
python -m examples.demo_interaction_and_hinge
```

## dc_solver engine + Abaqus-like input

The canonical analysis engine lives in `src/dc_solver` and reads Abaqus-like
`.inp` files. A ready-to-run example is:

```bash
python -m dc_solver.run examples/abaqus_like/portal_6seg.inp
```

Quickstart:

```bash
pip install -e .[dev]
pytest
python -m dc_solver.run examples/abaqus_like/beam_cantilever_tipload.inp
```

## Abaqus-like log files (.sta/.msg/.dat)

Enable Abaqus-style logs for a run with `--abaqus-like-logs` (and optional
`--output-dir` for where the files are written):

```bash
python -m dc_solver.run examples/abaqus_like/beam_cantilever_tipload.inp --abaqus-like-logs --output-dir results/beam_demo
```

This produces three files:

- `<job>.sta`: status table by increment (STEP, INC, ATT, iteration counts, and times).
- `<job>.msg`: step/increment/iteration messages suitable for `tail -f`.
- `<job>.dat`: input echo, warnings, and a final summary.

Legal note: this is an Abaqus-like layout for familiarity; not affiliated with Dassault Systèmes.

## Notes

- This is intended for research/prototyping and coursework-level models.
- Replace the default elastic stiffness extraction (`hinge_factory.py`) with your preferred EA/EI calibration.

## Benchmarks

Two simple beam benchmarks with closed-form Euler–Bernoulli comparisons live in
`examples/abaqus_like/`:

- Cantilever with tip point load:
  - Tip deflection: \u03b4 = P L^3 / (3 E I)
  - Tip rotation: \u03b8 = P L^2 / (2 E I)
- Simply supported beam with midspan point load:
  - Midspan deflection: \u03b4 = P L^3 / (48 E I)

See `tests/test_beam_benchmarks_theory.py` for the exact comparisons.
Beam benchmark plots now show curvature; run:

```bash
python -m dc_solver.run examples/abaqus_like/beam_cantilever_tipload.inp
```

## Input format

Supported Abaqus-like cards, limitations, and examples are documented in
`docs/INPUT_FORMAT.md`.

## Example: Pórtico (Problema 4) — Beam elements + 6 rótulas

Ejecuta:

```bash
python -m examples.demo_portico_problema4
```

Genera:

- `problem4_IDA.png`
- `problem4_last_*.png` (ag, drift, Vbase–drift, M–θ por rótula, N–M ±M con gradiente temporal)
- `problem4_last_energy_balance.png`, `problem4_last_energy_residual*.png` (balance de energía)
- `problem4_dt_sensitivity_*.png` + `problem4_dt_sensitivity.csv` (sensitividad en dt)

**Criterio de colapso por drift (default): 10% (snapshot/etiquetas a 4%)**

## Development

```bash
pip install -e .[dev]
pytest -q
```
