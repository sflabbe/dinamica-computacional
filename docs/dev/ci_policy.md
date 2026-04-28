# CI policy

Default CI runs deterministic fast tests only.

## Required jobs

- `pytest fast` runs on the existing OS/Python matrix.
- It installs `.[test]`.
- It runs `python -m pytest -q -m "not slow"`.
- It sets `DC_USE_NUMBA=0` so optional local/global Numba installs cannot
  change the CI path or fail collection through cache setup.
- Tests that need auxiliary repos must be marked `optional_external` and kept
  out of default CI until a real optional job is added for them.

## Optional jobs

- `pytest slow` runs only from `workflow_dispatch` when `run_slow` is true.
- It runs `python -m pytest -q -m slow`.
- Slow tests should be real assertions, not skips standing in for unavailable systems.

## Markers

- `integration`: integration boundary covered inside this repo.
- `optional_external`: needs a sibling repo, editable install, or real upstream.
- `slow`: useful but too expensive for required fast CI.
