# HPC Profiling Report — Evidence-Based Analysis

**Generated**: 2025-12-23T02:23:07.587271

**Commit**: `612b971f` (⚠️  DIRTY)
**Command**: `python tools/profile_run.py --tag baseline_hht_gravity --integrator hht --beam-hinge shm --state gravity`

## Baseline Validation

✅ Single-thread baseline validated.

## Problem Parameters

| Parameter | Value |
|-----------|-------|
| problem | Problem 4 Portal Frame |
| integrator | hht |
| beam_hinge | shm |
| state | gravity |
| nlgeom | False |
| H | 3.0 |
| L | 5.0 |
| nseg | 6 |
| n_elem | 18 |
| n_hinge | 4 |
| ndof | 24 |
| gravity_steps | 10 |
| nsteps | 10 |
| dt | static |

## Timing Summary

- **Wall time**: 5.783 s
- **CPU time**: 5.390 s
- **Steps/sec**: 1.7

## Top Hotspots by Cumulative Time

| Rank | Function | ncalls | tottime | cumtime | percall (cum) |
|------|----------|--------|---------|---------|---------------|
| 1 | `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)` | 1 | 0.0001 | 6.6981 | 6.698086 |
| 2 | `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)` | 1 | 0.0003 | 5.7828 | 5.782764 |
| 3 | `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)` | 3 | 0.0013 | 5.5897 | 1.863220 |
| 4 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3334(savefig)` | 3 | 0.0001 | 4.6221 | 1.540685 |
| 5 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2051(print_figure)` | 3 | 0.0002 | 4.6220 | 1.540651 |
| 6 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:92(draw_wrapper)` | 6 | 0.0000 | 4.1179 | 0.686318 |
| 7 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:53(draw_wrapper)` | 5450/6 | 0.0164 | 4.1179 | 0.000756 |
| 8 | `/usr/local/lib/python3.11/dist-packages/matplotlib/figure.py:3237(draw)` | 6 | 0.0003 | 4.1178 | 0.686305 |
| 9 | `/usr/local/lib/python3.11/dist-packages/matplotlib/image.py:116(_draw_list_compositing_images)` | 58/6 | 0.0021 | 2.0932 | 0.036090 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_base.py:3161(draw)` | 52 | 0.0014 | 2.0928 | 0.040246 |
| 11 | `/usr/local/lib/python3.11/dist-packages/matplotlib/layout_engine.py:265(execute)` | 3 | 0.0000 | 1.9975 | 0.665843 |
| 12 | `/usr/local/lib/python3.11/dist-packages/matplotlib/_constrained_layout.py:63(do_constrained_layout)` | 3 | 0.0002 | 1.9975 | 0.665827 |
| 13 | `/usr/local/lib/python3.11/dist-packages/matplotlib/_constrained_layout.py:625(get_pos_and_bbox)` | 80 | 0.0007 | 1.9638 | 0.024548 |
| 14 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:1395(_get_tightbbox_for_layout_only)` | 240/80 | 0.0015 | 1.9580 | 0.008158 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_base.py:4508(get_tightbbox)` | 80 | 0.0052 | 1.9573 | 0.024466 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backend_bases.py:2042(<lambda>)` | 6 | 0.0000 | 1.7392 | 0.289874 |
| 17 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backends/backend_agg.py:434(print_png)` | 6 | 0.0000 | 1.7392 | 0.289865 |
| 18 | `/usr/local/lib/python3.11/dist-packages/matplotlib/backends/backend_agg.py:424(_print_pil)` | 6 | 0.0001 | 1.7392 | 0.289862 |
| 19 | `/usr/local/lib/python3.11/dist-packages/matplotlib/_constrained_layout.py:342(make_layout_margins)` | 6 | 0.0011 | 1.7180 | 0.286325 |
| 20 | `/usr/local/lib/python3.11/dist-packages/matplotlib/axis.py:1337(get_tightbbox)` | 208 | 0.0046 | 1.6483 | 0.007925 |

## Top Hotspots by Self Time (tottime)

| Rank | Function | ncalls | tottime | cumtime | percall (tot) |
|------|----------|--------|---------|---------|---------------|
| 1 | `~:0(<method 'encode' of 'ImagingEncoder' objects>)` | 12 | 0.4405 | 0.4405 | 0.036708 |
| 2 | `~:0(<built-in method matplotlib.ft2font.set_text>)` | 859 | 0.2278 | 0.2289 | 0.000265 |
| 3 | `/usr/local/lib/python3.11/dist-packages/matplotlib/text.py:358(_get_layout)` | 5208 | 0.1474 | 0.8292 | 0.000028 |
| 4 | `~:0(<method 'reduce' of 'numpy.ufunc' objects>)` | 70412 | 0.1409 | 0.1409 | 0.000002 |
| 5 | `~:0(<built-in method posix.stat>)` | 1367 | 0.1284 | 0.1284 | 0.000094 |
| 6 | `/usr/local/lib/python3.11/dist-packages/matplotlib/font_manager.py:700(__hash__)` | 33791 | 0.0843 | 0.1262 | 0.000002 |
| 7 | `/usr/lib/python3.11/inspect.py:863(cleandoc)` | 7941 | 0.0785 | 0.1268 | 0.000010 |
| 8 | `~:0(<built-in method builtins.getattr>)` | 295067 | 0.0767 | 0.1696 | 0.000000 |
| 9 | `/usr/lib/python3.11/copy.py:66(copy)` | 39474 | 0.0636 | 0.1494 | 0.000002 |
| 10 | `/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py:781(__getitem__)` | 112296 | 0.0581 | 0.1163 | 0.000001 |
| 11 | `~:0(<built-in method marshal.loads>)` | 306 | 0.0568 | 0.0568 | 0.000186 |
| 12 | `~:0(<built-in method builtins.isinstance>)` | 346441 | 0.0568 | 0.0760 | 0.000000 |
| 13 | `/usr/local/lib/python3.11/dist-packages/matplotlib/transforms.py:2432(get_affine)` | 11750/8449 | 0.0564 | 0.1125 | 0.000005 |
| 14 | `/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_base.py:890(<dictcomp>)` | 6983 | 0.0539 | 0.1108 | 0.000008 |
| 15 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:315(stale)` | 146326/120413 | 0.0527 | 0.1126 | 0.000000 |
| 16 | `/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py:1188(_update_props)` | 9247/8987 | 0.0522 | 0.2439 | 0.000006 |
| 17 | `/home/user/dinamica-computacional/src/dc_solver/fem/model.py:53(assemble)` | 72 | 0.0491 | 0.0924 | 0.000682 |
| 18 | `/usr/lib/python3.11/copy.py:128(deepcopy)` | 40848/828 | 0.0469 | 0.1028 | 0.000001 |
| 19 | `/usr/local/lib/python3.11/dist-packages/matplotlib/ticker.py:2159(_raw_ticks)` | 1560 | 0.0468 | 0.3305 | 0.000030 |
| 20 | `/usr/local/lib/python3.11/dist-packages/matplotlib/transforms.py:750(__init__)` | 19007 | 0.0452 | 0.0909 | 0.000002 |

## Detailed Hotspot Analysis (Top 3 by Cumulative Time)

### 1. `/home/user/dinamica-computacional/tools/profile_run.py:115(run_problem4_profiled)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 6.6981 s (115.8% of total)
- **tottime**: 0.0001 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 2. `/home/user/dinamica-computacional/src/problems/problema4_portico.py:1082(main)`

- **Category**: application logic
- **ncalls**: 1
- **cumtime**: 5.7828 s (100.0% of total)
- **tottime**: 0.0003 s (self)
- **Why hot**: Called 1 times; likely inside timestep or Newton loop.

### 3. `/home/user/dinamica-computacional/src/dc_solver/post/plotting.py:356(plot_structure_states)`

- **Category**: linear algebra dominated
- **ncalls**: 3
- **cumtime**: 5.5897 s (96.7% of total)
- **tottime**: 0.0013 s (self)
- **Why hot**: Called 3 times; likely inside timestep or Newton loop.

---

**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.
