# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:35:49.299192

**Commit**: `b4d374cc` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag opt1_after_v2 --integrator hht --beam-hinge shm --state ida`

## Baseline Validation

✅ Single-thread baseline validated.

## Problem Parameters

| Parameter | Value |
|-----------|-------|
| problem | Problem 4 Portal Frame |
| integrator | hht |
| beam_hinge | shm |
| state | ida |
| nlgeom | False |
| H | 3.0 |
| L | 5.0 |
| nseg | 6 |
| n_elem | 18 |
| n_hinge | 4 |
| ndof | 24 |
| ag_min | 0.1 |
| ag_max | 0.2 |
| ag_step | 0.1 |
| n_ida_runs | 2 |

## Timing Summary

- **Wall time**: 76.177 s
- **CPU time**: 71.030 s

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0041 | 77.1104 | 77.110390 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0002 | 76.1728 | 76.172828 |
| 3 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)` | 1 | 0.0053 | 65.0948 | 65.094756 |
| 4 | `/home/user/dinamica-computacional/src/dc_solver/integrators/__init__.py:24(solve_dynamic)` | 2 | 0.0003 | 64.9334 | 32.466681 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2045 | 64.9327 | 32.466330 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 24.3487 | 47.9295 | 0.001374 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.9798 | 24.0643 | 0.000024 |
| 8 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:975(plot_results)` | 1 | 0.0002 | 10.6835 | 10.683471 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.5095 | 7.6754 | 0.000043 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 21 | 0.0004 | 7.1241 | 0.339241 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 21 | 0.0012 | 7.1235 | 0.339216 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:41(update_column_yields)` | 10053 | 0.2170 | 7.0276 | 0.000699 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0212 | 6.9753 | 0.000697 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.8624 | 6.5598 | 0.000007 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 42 | 0.0001 | 5.9313 | 0.141220 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 9960/42 | 0.0290 | 5.9311 | 0.000595 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 42 | 0.0012 | 5.9308 | 0.141210 |
| 18 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:106(export_problem4_hinges)` | 1 | 0.0017 | 5.6596 | 5.659577 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:63(plot_time_gradient)` | 16 | 0.0007 | 5.4554 | 0.340964 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0011 | 5.0233 | 1.674424 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 24.3487 | 47.9295 | 0.000698 |
| 2 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.9798 | 24.0643 | 0.000010 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.8624 | 6.5598 | 0.000005 |
| 4 | `~:0(<built-in method numpy.array>)` | 2553437 | 2.6455 | 2.6477 | 0.000001 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.5095 | 7.6754 | 0.000014 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:18(rot2d)` | 989568 | 2.1910 | 4.0085 | 0.000002 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:15(moment_capacity_from_polygon)` | 20109 | 2.1039 | 2.6724 | 0.000105 |
| 8 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:81(_geom)` | 1978884 | 1.9586 | 2.3113 | 0.000001 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:179(eval_increment)` | 89792 | 1.3157 | 3.9357 | 0.000015 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2045 | 64.9327 | 0.602228 |
| 11 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 52 | 1.0544 | 1.0544 | 0.020276 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0212 | 6.9753 | 0.000102 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:89(dofs)` | 989568 | 0.8887 | 1.6372 | 0.000001 |
| 14 | `/usr/local/lib/python3.11/dist-packages/numpy/linalg/_linalg.py:363(solve)` | 14881 | 0.8614 | 1.1077 | 0.000058 |
| 15 | `~:0(<method 'fill' of 'numpy.ndarray' objects>)` | 1978164 | 0.7969 | 0.7969 | 0.000000 |
| 16 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:140(_My0_base)` | 449479 | 0.7015 | 1.0185 | 0.000002 |
| 17 | `~:0(<built-in method builtins.max>)` | 3626179 | 0.6733 | 0.6746 | 0.000000 |
| 18 | `/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py:748(dot)` | 4962339 | 0.6600 | 0.6600 | 0.000000 |
| 19 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 1855 | 0.5296 | 0.5319 | 0.000285 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:165(_bw_rhs)` | 451036 | 0.4759 | 0.5450 | 0.000001 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 77.1104 s (101.2% of total)
- **tottime**: 0.0041 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 76.1728 s (100.0% of total)
- **tottime**: 0.0002 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 65.0948 s (85.5% of total)
- **tottime**: 0.0053 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
