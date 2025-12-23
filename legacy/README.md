# Legacy / Archived Code

This directory contains archived code that has been superseded by the modern framework but is preserved for historical reference and potential algorithm comparison.

## Archived Files

### `portico_shm.py` (1,260 lines)

**Original purpose:**
Standalone SDOF portal frame prototype with:
- Elastic-perfectly-plastic (EPP) or Bouc-Wen spring models
- Column and beam springs
- Energy-based degradation (SHM model)
- P-Delta geometric stiffness
- Multiple integrators (Velocity-Verlet, HHT-α)
- Various excitation types (sine, Ricker pulse, CSV input)

**Why archived:**
Completely replaced by the modern framework in:
- `src/problems/problema4_portico.py` — Full 2D frame solver with plastic hinges
- `src/dc_solver/` — Modular FEM core with beam elements, integrators, hinges
- `src/dc_solver/hinges/models.py` — Modern SHM hinge implementation

**Evidence of non-use:**
```bash
$ rg "import portico_shm|from portico_shm" .
# No matches

$ rg "portico_shm" . --files-with-matches
# Only self-reference
```

**Use cases for consulting this file:**
1. Understanding the original SHM degradation calibration approach
2. Comparing algorithm implementations (Bouc-Wen parameter scaling, etc.)
3. Historical context for early prototypes

**How to recover/use:**
```bash
# Copy back to root if needed for reference
cp legacy/portico_shm.py ./portico_shm_reference.py

# Execute directly (standalone, no dependencies on modern framework)
python legacy/portico_shm.py
```

---

### `rotula_plastica.py` (487 lines)

**Original purpose:**
Standalone N-M plastic hinge prototype with:
- RC fiber section analysis (concrete compression-only, elastic-plastic steel)
- N-M interaction surface computation (convex hull polygon)
- Return mapping for elasto-plastic flow (associated flow rule)
- Cyclic loading simulation with hysteresis plots

**Why archived:**
Completely replaced by:
- `plastic_hinge/` module — Modern fiber section and N-M hinge models
  - `plastic_hinge/fiber_section.py`
  - `plastic_hinge/hinge_nm.py`
  - `plastic_hinge/nm_surface.py`
  - `plastic_hinge/return_mapping.py`
  - `plastic_hinge/rc_section.py`
- `src/problems/problema2_*.py` — Modern verification problems for N-M interaction

**Evidence of non-use:**
```bash
$ rg "import rotula_plastica|from rotula_plastica" .
# No matches

$ rg "rotula_plastica" . --files-with-matches
# Only self-reference
```

**Use cases for consulting this file:**
1. Understanding the original N-M hinge algorithm
2. Comparing with modern `plastic_hinge/` implementation
3. Validation of material models (concrete/steel stress-strain)

**How to recover/use:**
```bash
# Copy back to root if needed
cp legacy/rotula_plastica.py ./rotula_plastica_reference.py

# Execute directly (standalone)
python legacy/rotula_plastica.py
```

---

## Relationship to Modern Framework

| Legacy File | Modern Replacement | Notes |
|-------------|-------------------|-------|
| `portico_shm.py` | `src/problems/problema4_portico.py`<br>`src/dc_solver/hinges/models.py` | Full 2D frame + modular hinge models |
| `rotula_plastica.py` | `plastic_hinge/` module<br>`src/problems/problema2_*.py` | Modular N-M hinge + verification problems |

---

## When to Delete

These files can be permanently deleted if:

1. ✅ Modern framework is fully validated and documented
2. ✅ No algorithm questions or historical comparison needs remain
3. ✅ No users have requested access to legacy implementations
4. ✅ Sufficient time has passed (e.g., 6+ months) without consultation

**Current status:** Archived (2025-12-23)
**Review date:** 2026-06-23 (6 months)

---

## Restoration Commands

If you need to restore these files to active use:

```bash
# Move back to root
git mv legacy/portico_shm.py portico_shm.py
git mv legacy/rotula_plastica.py rotula_plastica.py

# Commit restoration
git commit -m "restore: bring back legacy prototypes from archive"

# Update documentation to reflect their active status
```

---

**Archived by:** Claude Code Cleanup (2025-12-23)
**Branch:** `claude/cleanup-dead-code-GPbii`
**Evidence:** See `CLEANUP_REPORT.md` for full analysis
