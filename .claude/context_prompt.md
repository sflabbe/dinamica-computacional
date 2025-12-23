# Repository Context Prompt for Claude Code

Please gather complete context about this repository by exploring the following areas:

## 1. Project Overview & Documentation
- Read `README.md` - Main project documentation
- Read `quickstart.md` - Quick reference commands
- Read `DEBUG_CHECKLIST.md` - Troubleshooting guide
- Read `CHANGELOG_DEV.md` - Development changelog
- Read `CLEANUP_REPORT.md` - Recent cleanup activities
- Read `pyproject.toml` - Project dependencies and metadata

## 2. Core Architecture & Modules

### Plastic Hinge Module (`plastic_hinge/`)
Read all files to understand:
- `fiber_section.py` - Fiber section discretization (RC sections)
- `hinge_nm.py` - Coupled N-M plastic hinge implementation
- `return_mapping.py` - Plasticity return mapping algorithm
- `nm_surface.py` - N-M yield surface definition
- `rc_section.py` - Reinforced concrete section geometry
- `hinge_factory.py` - Factory for creating hinge instances
- `geometry.py` - Geometric utilities

### FEM Solver (`src/dc_solver/`)

#### Core FEM (`src/dc_solver/fem/`)
- `model.py` - Main FEM model class and assembly
- `frame2d.py` - 2D frame element (Timoshenko + P-Δ)
- `nodes.py` - Node management
- `utils.py` - FEM utilities

#### Time Integrators (`src/dc_solver/integrators/`)
- `hht_alpha.py` - HHT-α implicit integrator
- `newmark.py` - Newmark-β implicit integrator
- `explicit.py` - Explicit Velocity Verlet with auto-substepping

#### Hinges (`src/dc_solver/hinges/`)
- `models.py` - Hinge model implementations (SHM, fiber)
- `shm_calibration.py` - SHM Bouc-Wen calibration

#### Static Solvers (`src/dc_solver/static/`)
- `newton.py` - Newton-Raphson solver with line search

#### Materials (`src/dc_solver/materials/`)
- `elastic.py` - Elastic material models

#### Post-processing (`src/dc_solver/post/`)
- `hysteresis_gradient.py` - Gradient coloring for plots
- `hinge_exports.py` - Export hinge results
- `energy_balance.py` - Energy balance checks
- `plotting.py` - General plotting utilities
- `fiber_mesh_plot.py` - Fiber section visualization

#### Job Infrastructure (`src/dc_solver/job/`)
- `runner.py` - Abaqus-like job runner
- `journal.py` - Journal logging
- `console.py` - Console output
- `file_tracker.py` - Track created files
- `flops.py` - FLOP estimation

#### Kernels (`src/dc_solver/kernels/`)
- `hinge_jit.py` - JIT-compiled hinge routines
- `assemble_jit.py` - JIT-compiled assembly routines

#### Utilities (`src/dc_solver/utils/`)
- `gravity.py` - Gravity load utilities

#### I/O (`src/dc_solver/io/`)
- `abaqus_inp.py` - Abaqus INP file parser

## 3. Problem Scripts (`src/problems/`)
Read to understand usage patterns:
- `problema2_interaccion.py` - N-M interaction diagram generation
- `problema2_hinge_nm_verification.py` - N-M hinge verification
- `problema2_secciones_nm.py` - N-M section analysis
- `problema3_shm_verify.py` - SHM hinge verification
- `problema4_portico.py` - **Main portal frame solver** (gravity + IDA)
- `problema5_fiber_section_interaction.py` - RC fiber section N-M curves
- `problema6_portico_elastico.py` - Elastic reference solution

## 4. Test Suite (`tests/`)
Explore test files to understand:
- Test coverage and validation approach
- `fixtures.py` - Test fixtures
- `test_*.py` - Unit and integration tests
- Key tests: hinge behavior, FEM assembly, integrators, energy balance

## 5. Documentation (`docs/`)
- `INPUT_FORMAT.md` - Input file format specification
- `input_deck_v2.md` - Input deck version 2 documentation

## 6. Examples (`examples/`)
- `examples/README.md` - Examples overview
- `examples/demo_job_infrastructure.py` - Job infrastructure demo
- `examples/abaqus_like/` - Abaqus-like workflow examples
- `examples/portal_frame_v2/` - Portal frame examples with amplitudes

## 7. Key Architectural Patterns

After reading the above, understand:

### Time Integration Strategy
- HHT-α (default, implicit, energy-dissipative)
- Newmark-β (implicit, energy-conserving option)
- Explicit with automatic stability substepping

### Plasticity Models
- Coupled N-M yield surface with return mapping
- SHM (degrading hysteresis) for beam hinges
- RC fiber sections for realistic N-M interaction

### Nonlinear Geometry
- P-Δ effects via geometric stiffness
- Corotational formulation available

### Solution Strategy
- **Step 1**: Gravity preload (static Newton with adaptive load substepping)
- **Step 2**: Dynamic analysis / IDA (incremental dynamic analysis)

### Job Infrastructure (Abaqus-like)
- `journal.log` - Chronological event log
- `<job>.msg` - Warnings/errors
- `<job>.sta` - Step progress
- `<job>.dat` - Data summary
- `run_info.json/txt` - Metadata snapshot

### Performance Optimization
- Optional Numba JIT compilation (controlled by `DC_USE_NUMBA` env var)
- Optimized assembly routines

## 8. Command-Line Interface Patterns

Understand the CLI structure from `problema4_portico.py`:
- `--integrator {hht,newmark,explicit}`
- `--beam-hinge {shm,fiber,compare}`
- `--state {gravity,ida}`
- `--nlgeom` - Enable P-Δ effects
- `--line-search` - Enable backtracking line search
- `--gravity-verbose` - Detailed gravity output
- Time step controls: `--base-dt`, `--dt-min`
- Newton controls: `--max-iter`, `--tol`
- IDA amplitude range: `--ag-min`, `--ag-max`, `--ag-step`
- Fiber discretization: `--fiber-ny`, `--fiber-nz`
- Debug options: `--debug-cutback`

## 9. Recent Changes & Dead Code

Review:
- `CLEANUP_REPORT.md` - Recent dead code removal and archiving
- `legacy/` - Archived prototype code

## 10. Development Workflow

Understand:
- Installation: `python -m pip install -e .`
- Alternative: `PYTHONPATH=src python -m problems.problema4_portico ...`
- Testing: `pytest`
- Optional Numba: Install with `pip install -e ".[numba]"`

## Summary Questions to Answer

After gathering context, you should be able to answer:

1. **What is this repository for?**
   - 2D frame solver for structural dynamics with plastic hinges

2. **What are the main components?**
   - Plastic hinge models (N-M, SHM, fiber)
   - FEM solver with Timoshenko beams
   - Multiple time integrators
   - Abaqus-like job infrastructure

3. **What are the key problem scripts?**
   - Problema 2: N-M verification
   - Problema 3: SHM verification
   - Problema 4: Portal frame (main application)
   - Problema 5: Fiber section interaction
   - Problema 6: Elastic reference

4. **What are common troubleshooting steps?**
   - Enable line search (`--line-search`)
   - Adjust time step (`--base-dt`, `--dt-min`)
   - Increase iterations (`--max-iter`)
   - Try explicit integrator
   - Run gravity-only first
   - Check `DEBUG_CHECKLIST.md`

5. **How is the code organized?**
   - `plastic_hinge/`: Core plasticity algorithms
   - `src/dc_solver/`: FEM solver infrastructure
   - `src/problems/`: Runnable problem scripts
   - `tests/`: Test suite
   - `examples/`: Usage examples

6. **What are the key design patterns?**
   - Factory pattern for hinge creation
   - Abaqus-like job workflow
   - Pluggable time integrators
   - Optional JIT compilation
   - Adaptive substepping (gravity + explicit)

---

**Note**: This context should give you comprehensive understanding of the repository's purpose, architecture, usage patterns, and development workflow. You should now be able to assist with debugging, feature development, documentation, and general repository maintenance.
