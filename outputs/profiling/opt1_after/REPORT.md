# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:33:42.359464

**Commit**: `b4d374cc` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag opt1_after --integrator hht --beam-hinge shm --state ida`

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

- **Wall time**: 81.553 s
- **CPU time**: 76.080 s

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0052 | 82.6539 | 82.653882 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0003 | 81.5477 | 81.547698 |
| 3 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)` | 1 | 0.0085 | 69.6612 | 69.661248 |
| 4 | `/home/user/dinamica-computacional/src/dc_solver/integrators/__init__.py:24(solve_dynamic)` | 2 | 0.0004 | 69.4815 | 34.740741 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.3324 | 69.4807 | 34.740338 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 25.3505 | 51.0954 | 0.001464 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 10.8028 | 26.4901 | 0.000027 |
| 8 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:975(plot_results)` | 1 | 0.0002 | 11.5047 | 11.504692 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.7266 | 8.1595 | 0.000045 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 21 | 0.0004 | 7.6659 | 0.365042 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 21 | 0.0013 | 7.6653 | 0.365014 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0825 | 7.5705 | 0.000757 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:41(update_column_yields)` | 10053 | 0.2322 | 7.5546 | 0.000751 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 5.1154 | 6.9149 | 0.000007 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 42 | 0.0001 | 6.3802 | 0.151909 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 9960/42 | 0.0326 | 6.3800 | 0.000641 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 42 | 0.0015 | 6.3797 | 0.151898 |
| 18 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:106(export_problem4_hinges)` | 1 | 0.0021 | 6.0611 | 6.061120 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:63(plot_time_gradient)` | 16 | 0.0009 | 5.8403 | 0.365017 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0015 | 5.4428 | 1.814262 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 25.3505 | 51.0954 | 0.000727 |
| 2 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 10.8028 | 26.4901 | 0.000011 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 5.1154 | 6.9149 | 0.000005 |
| 4 | `~:0(<built-in method numpy.array>)` | 2553437 | 2.8058 | 2.8081 | 0.000001 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.7266 | 8.1595 | 0.000015 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:18(rot2d)` | 989568 | 2.2962 | 4.2118 | 0.000002 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:15(moment_capacity_from_polygon)` | 20109 | 2.1781 | 2.7914 | 0.000108 |
| 8 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:81(_geom)` | 1978884 | 2.1246 | 2.4918 | 0.000001 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:179(eval_increment)` | 89792 | 1.3995 | 4.1273 | 0.000016 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.3324 | 69.4807 | 0.666192 |
| 11 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 52 | 1.1245 | 1.1245 | 0.021625 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0825 | 7.5705 | 0.000108 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:89(dofs)` | 989568 | 0.9361 | 1.7258 | 0.000001 |
| 14 | `/usr/local/lib/python3.11/dist-packages/numpy/linalg/_linalg.py:363(solve)` | 14881 | 0.9315 | 1.2020 | 0.000063 |
| 15 | `~:0(<method 'copy' of 'numpy.ndarray' objects>)` | 2122372 | 0.9314 | 0.9314 | 0.000000 |
| 16 | `~:0(<method 'fill' of 'numpy.ndarray' objects>)` | 1978164 | 0.8314 | 0.8314 | 0.000000 |
| 17 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:140(_My0_base)` | 449479 | 0.7345 | 1.0621 | 0.000002 |
| 18 | `~:0(<built-in method builtins.max>)` | 3626178 | 0.6992 | 0.7007 | 0.000000 |
| 19 | `/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py:748(dot)` | 4962339 | 0.6795 | 0.6795 | 0.000000 |
| 20 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 1855 | 0.5415 | 0.5439 | 0.000292 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 82.6539 s (101.4% of total)
- **tottime**: 0.0052 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 81.5477 s (100.0% of total)
- **tottime**: 0.0003 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 69.6612 s (85.4% of total)
- **tottime**: 0.0085 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
