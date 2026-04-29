# Dev Changelog

## [0.2.0] — 2026-04

### Fixed
- `FiberBeamHinge1D` (fiber_right): `kappa_factor=-2.0, moment_sign=+1.0` para
  coherencia de curvatura física entre extremo izquierdo y derecho de viga RC
  asimétrica. Antes, el extremo derecho resolvía `_solve_eps0` en el tramo
  incorrecto de la curva N(ε₀).
- `_scaled_displacements` en `post/plotting.py`: nodos auxiliares que comparten
  `dof_u` con su nodo padre ya no producen escala² en la visualización.
- Gravity plotting en modo `--state gravity`: `u_hist[0]` era `zeros` en vez de
  `u_grav`, causando que State 2 mostrara la geometría sin deformar.
- `Model.col_hinge_groups`: añadido `field(default_factory=list)` — el argumento
  ya no es requerido para modelos sin acoplamiento columna-bisagra.

### Added
- Modelo de hormigón CEB-90 en `plastic_hinge/fiber_section.py`:
  `_sigma_c_ceb90_tangent_scalar` con rama descendente (post-peak softening).
  Reemplaza el modelo parabólico-rectangular en `FiberSection2DStateful`.
  `E_ci` calculado automáticamente desde `fc` vía fórmula EC2/FIB.
- Tests de regresión: `test_problem4_fiber_gravity_symmetry.py` (Fase 0).

### Changed
- `pyproject.toml`: paquete renombrado de `plastic-hinge-nm` a `dc-solver`,
  versión bumpeada a `0.2.0`.
- `dc_solver/__init__.py`: API pública mínima exportada.
- Numba guards unificados: `DC_USE_NUMBA=0` desactiva todos los kernels JIT.

## Latest
- Implemented robust 2D projection for N–M return mapping (full edge/vertex candidates).
- Added consistent tangent + optional substepping to PlasticHingeNM and a new 2D N–M hinge element for columns.
- Reworked SHM beam hinge to a smooth hysteretic model with degradation/pinching and added new Problem 2/3 scripts plus tests.

### How to run
- Problem 2: `python -m problems.problema2_interaccion`
- Problem 3: `python -m problems.problema3_shm_verify`
- Problem 4: `python -m problems.problema4_portico`
