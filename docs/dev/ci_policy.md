# CI policy

The canonical CI path runs with `DC_USE_NUMBA=0`.
This disables all JIT kernels, including `plastic_hinge` and `dc_solver.kernels`.
`DC_FAST=1` is a backwards compatible local profiling opt in.
