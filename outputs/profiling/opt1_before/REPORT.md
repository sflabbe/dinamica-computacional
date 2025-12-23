# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:32:10.628996

**Commit**: `b4d374cc` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag opt1_before --integrator hht --beam-hinge shm --state ida`

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

- **Wall time**: 74.394 s
- **CPU time**: 69.390 s

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0043 | 75.3105 | 75.310472 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0002 | 74.3892 | 74.389173 |
| 3 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)` | 1 | 0.0057 | 63.3711 | 63.371114 |
| 4 | `/home/user/dinamica-computacional/src/dc_solver/integrators/__init__.py:24(solve_dynamic)` | 2 | 0.0003 | 63.2073 | 31.603666 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.1668 | 63.2067 | 31.603328 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 24.1926 | 46.7739 | 0.001341 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.7988 | 22.9566 | 0.000023 |
| 8 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:975(plot_results)` | 1 | 0.0001 | 10.6304 | 10.630444 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.3117 | 7.4195 | 0.000041 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 21 | 0.0004 | 7.1101 | 0.338576 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 21 | 0.0011 | 7.1096 | 0.338552 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:41(update_column_yields)` | 10053 | 0.2426 | 6.8192 | 0.000678 |
| 13 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0178 | 6.7145 | 0.000671 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.8058 | 6.5440 | 0.000007 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 42 | 0.0001 | 5.9074 | 0.140652 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 9960/42 | 0.0302 | 5.9072 | 0.000593 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 42 | 0.0012 | 5.9069 | 0.140642 |
| 18 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:106(export_problem4_hinges)` | 1 | 0.0016 | 5.6388 | 5.638820 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/post/hinge_exports.py:63(plot_time_gradient)` | 16 | 0.0007 | 5.4401 | 0.340003 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0011 | 4.9910 | 1.663665 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 34891 | 24.1926 | 46.7739 | 0.000693 |
| 2 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:182(stiffness_and_force_global)` | 989082 | 9.7988 | 22.9566 | 0.000010 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:97(k_local)` | 989280 | 4.8058 | 6.5440 | 0.000005 |
| 4 | `~:0(<built-in method numpy.array>)` | 2553437 | 2.5057 | 2.5078 | 0.000001 |
| 5 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:620(eval_trial)` | 179584 | 2.3117 | 7.4195 | 0.000013 |
| 6 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:18(rot2d)` | 989568 | 2.1814 | 3.8891 | 0.000002 |
| 7 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:15(moment_capacity_from_polygon)` | 20109 | 2.0802 | 2.6272 | 0.000103 |
| 8 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:81(_geom)` | 1978884 | 1.9602 | 2.2983 | 0.000001 |
| 9 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:179(eval_increment)` | 89792 | 1.3026 | 3.9002 | 0.000015 |
| 10 | `/home/user/dinamica-computacional/src/dc_solver/integrators/hht_alpha.py:125(hht_alpha_newton)` | 2 | 1.1668 | 63.2067 | 0.583413 |
| 11 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 52 | 1.0665 | 1.0665 | 0.020509 |
| 12 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:121(base_shear)` | 10003 | 1.0178 | 6.7145 | 0.000102 |
| 13 | `~:0(<built-in method numpy.zeros>)` | 2109248 | 0.9831 | 0.9831 | 0.000000 |
| 14 | `/home/user/dinamica-computacional/src/dc_solver/fem/frame2d.py:89(dofs)` | 989568 | 0.9099 | 1.5593 | 0.000001 |
| 15 | `/usr/local/lib/python3.11/dist-packages/numpy/linalg/_linalg.py:363(solve)` | 14881 | 0.8297 | 1.0678 | 0.000056 |
| 16 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:140(_My0_base)` | 449479 | 0.7113 | 1.0218 | 0.000002 |
| 17 | `~:0(<built-in method builtins.max>)` | 3626178 | 0.6541 | 0.6553 | 0.000000 |
| 18 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 1855 | 0.5633 | 0.5656 | 0.000304 |
| 19 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:165(_bw_rhs)` | 451036 | 0.4769 | 0.5447 | 0.000001 |
| 20 | `/home/user/dinamica-computacional/src/dc_solver/hinges/models.py:602(dofs)` | 359172 | 0.3545 | 0.6148 | 0.000001 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 75.3105 s (101.2% of total)
- **tottime**: 0.0043 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 74.3892 s (100.0% of total)
- **tottime**: 0.0002 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:747(run_incremental_amplitudes)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 63.3711 s (85.2% of total)
- **tottime**: 0.0057 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
