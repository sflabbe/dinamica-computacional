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

## Notes

- This is intended for research/prototyping and coursework-level models.
- Replace the default elastic stiffness extraction (`hinge_factory.py`) with your preferred EA/EI calibration.


## Portal frame example (Problema 4)

```bash
python -m examples.demo_portico_problema4
```

This example uses a **beam-elements + concentrated hinges** formulation (6 hinges total) and runs an IDA until collapse by drift.


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
