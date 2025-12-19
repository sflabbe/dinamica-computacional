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

## New engine + Abaqus-like input

The newer analysis engine can read Abaqus-like `.inp` files via
`dinamica_computacional.io.abaqus_like.read_inp`. A ready-to-run example is:

```bash
python -m examples.demo_frame --input inputs/portal_problem4.inp --prefix problem4
```

Key input cards and parameters:

- `*HINGE_NM_PARAMS, ELSET=..., SURFACE=...`
  - `K0` (initial rotational stiffness)
  - `ALPHA_POST` (post-yield stiffness ratio; default `1e-4`)
  - `SURFACE` (name of the `*NMSURFACE` polygon)
- `*HINGE_MTHETA_SHM_PARAMS, ELSET=...`
  - `K0` (initial rotational stiffness)
  - `MY` (yield moment)
  - `ALPHA_POST` (post-yield stiffness ratio; default `0.02`)
  - `CK` (strength deterioration coefficient; default `2.0`)
  - `CMY` (yield strength deterioration coefficient; default `1.0`)
- `*HHT, ALPHA=...` with `t_start, t_end, dt` for HHT-alpha dynamic steps.

## Notes

- This is intended for research/prototyping and coursework-level models.
- Replace the default elastic stiffness extraction (`hinge_factory.py`) with your preferred EA/EI calibration.


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

# dinamica-computacional

## Development

```bash
pip install -e .[dev]
pytest -q
```
