# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:40:41.565835

**Commit**: `be6976fc` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag opt2_after --integrator hht --beam-hinge shm --state ida`

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

- **Wall time**: 56.264 s
- **CPU time**: 52.440 s

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0043 | 57.5875 | 57.587528 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0002 | 56.2601 | 56.260054 |
| 3 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)` | 1 | 0.0053 | 44.9749 | 44.974853 |
| 4 | `/home/user/dinamica-computacional/src/dc_solver/integrators/__init__.py:24(solve_dynamic)` | 2 | 0.0004 | 44.5130 | 22.256487 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2642 | 44.5122 | 22.256080 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:60(assemble)` | 34891 | 3.2372 | 27.4015 | 0.000785 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.4479 | 23.7300 | 0.000024 |
| 8 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:975(plot_results)` | 1 | 0.0002 | 10.8959 | 10.895911 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.5592 | 8.1438 | 0.000045 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:127(base_shear)` | 10003 | 1.1128 | 7.2901 | 0.000729 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 21 | 0.0004 | 7.2885 | 0.347071 |
| 12 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 21 | 0.0013 | 7.2879 | 0.347042 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:48(update_column_yields)` | 10053 | 0.2103 | 7.0193 | 0.000698 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.9989 | 6.7510 | 0.000007 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 42 | 0.0001 | 6.0798 | 0.144757 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 9960/42 | 0.0294 | 6.0797 | 0.000610 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 42 | 0.0014 | 6.0793 | 0.144746 |
| 18 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:106(export_problem4_hinges)` | 1 | 0.0019 | 5.7856 | 5.785650 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:63(plot_time_gradient)` | 16 | 0.0009 | 5.5810 | 0.348810 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0012 | 5.1095 | 1.703165 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.4479 | 23.7300 | 0.000010 |
| 2 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.9989 | 6.7510 | 0.000005 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:60(assemble)` | 34891 | 3.2372 | 27.4015 | 0.000093 |
| 4 | `~:0(<built-in method numpy.array>)` | 2553581 | 2.7179 | 2.7209 | 0.000001 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.5592 | 8.1438 | 0.000014 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:18(rot2d)` | 989568 | 2.1603 | 3.9803 | 0.000002 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:15(moment_capacity_from_polygon)` | 20109 | 2.0817 | 2.6356 | 0.000104 |
| 8 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:81(_geom)` | 1978884 | 1.9863 | 2.3344 | 0.000001 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:179(eval_increment)` | 89792 | 1.4816 | 4.3289 | 0.000017 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2642 | 44.5122 | 0.632084 |
| 11 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:127(base_shear)` | 10003 | 1.1128 | 7.2901 | 0.000111 |
| 12 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 52 | 1.0733 | 1.0733 | 0.020641 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:89(dofs)` | 989568 | 0.8652 | 1.6590 | 0.000001 |
| 14 | `/usr/local/lib/python3.11/dist-packages/numpy/linalg/_linalg.py:382(solve)` | 14881 | 0.8103 | 1.0632 | 0.000054 |
| 15 | `~:0(<method 'fill' of 'numpy.ndarray' objects>)` | 1978164 | 0.7979 | 0.7979 | 0.000000 |
| 16 | `~:0(<built-in method builtins.max>)` | 3629639 | 0.7918 | 0.7937 | 0.000000 |
| 17 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:140(_My0_base)` | 449479 | 0.7488 | 1.1601 | 0.000002 |
| 18 | `/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py:769(dot)` | 4962339 | 0.6826 | 0.6826 | 0.000000 |
| 19 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 1855 | 0.5560 | 0.5584 | 0.000300 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:165(_bw_rhs)` | 451036 | 0.4975 | 0.5718 | 0.000001 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 57.5875 s (102.4% of total)
- **tottime**: 0.0043 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 56.2601 s (100.0% of total)
- **tottime**: 0.0002 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 44.9749 s (79.9% of total)
- **tottime**: 0.0053 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
