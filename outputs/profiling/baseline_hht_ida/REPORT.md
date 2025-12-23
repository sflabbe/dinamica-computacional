# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:25:39.730804

**Commit**: `612b971f` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag baseline_hht_ida --integrator hht --beam-hinge shm --state ida`

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

- **Wall time**: 77.856 s
- **CPU time**: 72.610 s

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0053 | 78.8291 | 78.829129 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0002 | 77.8513 | 77.851255 |
| 3 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)` | 1 | 0.0073 | 66.1777 | 66.177677 |
| 4 | `/home/user/dinamica-computacional/src/dc_solver/integrators/__init__.py:24(solve_dynamic)` | 2 | 0.0004 | 66.0002 | 33.000112 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2722 | 65.9995 | 32.999735 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 25.0540 | 48.7426 | 0.001397 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:131(stiffness_and_force_global)` | 989082 | 10.0651 | 23.8071 | 0.000024 |
| 8 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:975(plot_results)` | 1 | 0.0002 | 11.2451 | 11.245149 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.4809 | 7.8751 | 0.000044 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 21 | 0.0004 | 7.4901 | 0.356671 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 21 | 0.0014 | 7.4895 | 0.356642 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:41(update_column_yields)` | 10053 | 0.2502 | 7.1028 | 0.000707 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0814 | 7.0224 | 0.000702 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:59(k_local)` | 989280 | 5.0063 | 6.8652 | 0.000007 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 42 | 0.0001 | 6.2081 | 0.147811 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 9960/42 | 0.0311 | 6.2079 | 0.000623 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 42 | 0.0014 | 6.2076 | 0.147799 |
| 18 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:106(export_problem4_hinges)` | 1 | 0.0022 | 5.9824 | 5.982364 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:63(plot_time_gradient)` | 16 | 0.0009 | 5.7446 | 0.359038 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0013 | 5.2621 | 1.754028 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 25.0540 | 48.7426 | 0.000718 |
| 2 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:131(stiffness_and_force_global)` | 989082 | 10.0651 | 23.8071 | 0.000010 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:59(k_local)` | 989280 | 5.0063 | 6.8652 | 0.000005 |
| 4 | `~:0(<built-in method numpy.array>)` | 2553437 | 2.6140 | 2.6162 | 0.000001 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.4809 | 7.8751 | 0.000014 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:14(rot2d)` | 989568 | 2.2352 | 4.0295 | 0.000002 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:15(moment_capacity_from_polygon)` | 20109 | 2.1990 | 2.7836 | 0.000109 |
| 8 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:43(_geom)` | 1978884 | 2.1332 | 2.4987 | 0.000001 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:179(eval_increment)` | 89792 | 1.3814 | 4.1185 | 0.000015 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.2722 | 65.9995 | 0.636121 |
| 11 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 52 | 1.1247 | 1.1247 | 0.021629 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0814 | 7.0224 | 0.000108 |
| 13 | `~:0(<built-in method numpy.zeros>)` | 2109248 | 1.0663 | 1.0663 | 0.000001 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:51(dofs)` | 989568 | 0.9004 | 1.5676 | 0.000001 |
| 15 | `/usr/local/lib/python3.11/dist-packages/numpy/linalg/_linalg.py:363(solve)` | 14881 | 0.8952 | 1.1546 | 0.000060 |
| 16 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:140(_My0_base)` | 449479 | 0.7459 | 1.0670 | 0.000002 |
| 17 | `~:0(<built-in method builtins.max>)` | 3626178 | 0.6853 | 0.6866 | 0.000000 |
| 18 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 1855 | 0.5418 | 0.5443 | 0.000292 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:165(_bw_rhs)` | 451036 | 0.5017 | 0.5733 | 0.000001 |
| 20 | `~:0(<built-in method math.hypot>)` | 1979298 | 0.3655 | 0.3655 | 0.000000 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 78.8291 s (101.2% of total)
- **tottime**: 0.0053 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 77.8513 s (100.0% of total)
- **tottime**: 0.0002 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 66.1777 s (85.0% of total)
- **tottime**: 0.0073 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
