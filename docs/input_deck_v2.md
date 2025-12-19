# Input Deck v2

This project supports a simplified Abaqus-like input deck. The v2 structure
promotes modular includes and consistent naming.

## Folder layout (recommended)

```
main.inp
geometry.inp
materials.inp
sections.inp
steps_gravity.inp
steps_dynamic.inp
amplitudes/
  AMP_*.inp
```

`main.inp` orchestrates the includes:

```
** INPUT DECK V2
** UNITS: SI (m, N, kg, s)
*Include, input=geometry.inp
*Include, input=materials.inp
*Include, input=sections.inp
*Include, input=steps_gravity.inp
*Include, input=steps_dynamic.inp
```

## Naming conventions

- Nsets:
  - `NS_BASE`, `NS_TOP_L`, `NS_TOP_R`, `NS_JOINT_L`, `NS_JOINT_R`, `NS_ALL`
- Elsets:
  - `ES_COL_L`, `ES_COL_R`, `ES_BEAM`, `ES_FRAME_ALL`

Avoid generic `Set-1`, `Set-2` names. Keep only referenced sets.

## Units policy

Always include a header comment declaring units, e.g.:

```
** UNITS: SI (m, N, kg, s)
```

Gravity should be applied with `g = 9.81` in SI.

## Supported keywords

The parser supports the following keywords (case-insensitive):

- `*Include` (relative paths)
- `*Part`, `*End Part`
- `*Assembly`, `*Instance`, `*End Instance`, `*End Assembly`
- `*Node`, `*Element`
- `*Nset`, `*Elset` (including `generate`)
- `*Material`, `*Elastic`, `*Density`
- `*Beam Section`
- `*Step`, `*Static`, `*Dynamic`, `*End Step`
- `*Boundary` (including `type=ACCELERATION` + `amplitude=...`)
- `*Dload` (GRAV)
- `*Amplitude` (TABULAR)
- `*Output`, `*Node Output`, `*Element Output` (accepted for deck structure)

## Ordering rules (v2)

1. Geometry + sets
2. Materials
3. Sections
4. Steps (gravity, then dynamic)
5. Amplitudes included in the step file that uses them

## Output requests

Include explicit output blocks for validation:

```
*Output, field
*Node Output, nset=NS_ALL
U, UR, RF
*Element Output, elset=ES_FRAME_ALL
S, E
```

## Normalizer/linter

Use `tools/inp_normalize.py` to rename sets, split includes, and detect unused
amplitudes. Example:

```bash
python tools/inp_normalize.py examples/Job-1.inp --output-dir normalized_v2 \
  --map Set-1=NS_BASE --map Set-2=NS_TOP_L
```

The normalizer writes a `main.inp` with includes and drops unused amplitudes
(best-effort).
