# Dev Changelog

## Latest
- Implemented robust 2D projection for N–M return mapping (full edge/vertex candidates).
- Added consistent tangent + optional substepping to PlasticHingeNM and a new 2D N–M hinge element for columns.
- Reworked SHM beam hinge to a smooth hysteretic model with degradation/pinching and added new Problem 2/3 scripts plus tests.

### How to run
- Problem 2: `python -m problems.problema2_interaccion`
- Problem 3: `python -m problems.problema3_shm_verify`
- Problem 4: `python -m problems.problema4_portico`
