# Hinge M-theta return mapping patch

## Problem

`tests/test_hinge_mtheta_return_mapping.py` expected exact elastic-perfectly-plastic
M-theta behavior while instantiating `SHMBeamHinge1D`.

That was a model-contract mismatch: `SHMBeamHinge1D` is a smooth Bouc-Wen
hysteresis model with optional degradation, not an exact 1D return-mapping hinge.

## Patch

- Added `BilinearMThetaHinge1D` for deterministic bilinear M-theta return mapping.
- Added `beam_bilinear` / `mtheta_bilinear` support in `RotSpringElement`.
- Rewrote the return-mapping test to use `BilinearMThetaHinge1D`.
- Added a separate SHM contract test that checks smooth Bouc-Wen behavior without
  pretending it caps exactly at `My`.
- Added a default factory for `Model.col_hinge_groups` so simple model fixtures can
  be constructed without explicitly passing column hinge groups.

## Verification

The test suite was run in chunks due to the notebook execution timeout:

- 58 project tests passed across `tests/test_*.py`.
- 2 smoke tests passed from `tools/smoke_test.py`.

Total checked by tests: 60 passed.
