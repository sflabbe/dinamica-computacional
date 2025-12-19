# Abaqus-like input format (dc_solver)

`dc_solver` reads a focused subset of Abaqus-style `.inp` files for 2D frame
problems. The parser is intentionally small, so unsupported keywords are ignored.

## Supported keywords

- `*PART`, `*NODE`, `*ELEMENT`
  - Elements are 2-node frame/beam (`B21`) with 3 DOF per node (ux, uy, \u03b8z).
- `*NSET`, `*ELSET` (including `generate`)
- `*BEAM SECTION, ELSET=..., MATERIAL=..., SECTION=RECT`
  - Rectangle defined by `b, h`
- `*MATERIAL`, `*ELASTIC`, `*DENSITY`
- `*ASSEMBLY`, `*INSTANCE`, `*END INSTANCE`, `*END ASSEMBLY`
- `*STEP`, `*STATIC`, `*DYNAMIC`
- `*BOUNDARY` (displacement) and `*BOUNDARY, TYPE=ACCELERATION` (base accel)
- `*DLOAD` with `GRAV` for self-weight
- `*CLOAD` for concentrated nodal loads
- `*AMPLITUDE` (used by acceleration boundaries)

## Key limitations

- Only 2D frame elements with translational DOF 1–2 and rotation DOF 3.
- `*CLOAD` ignores `AMPLITUDE` for now (constant per step).
- Dynamic steps only support `*BOUNDARY, TYPE=ACCELERATION` with DOF 1 (ux).
- No contact, multi-part assemblies, or nonlinear materials.

## Example snippet

```text
*Part, name=CANTILEVER
*Node
1, 0.0, 0.0
2, 2.0, 0.0
*Element, type=B21, elset=BEAM
1, 1, 2
*Nset, nset=FIX
1
*Beam Section, elset=BEAM, material=STEEL, section=RECT
0.20, 0.33
*Material, name=STEEL
*Elastic
2.10e11, 0.3
*Step, name=Static
*Static
*Boundary
FIX, 1, 3
*Cload
2, 2, -10000.0
*End Step
```
