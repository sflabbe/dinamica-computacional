# Architecture

```text
app/pages -> app/components -> app/services -> dc_solver -> plastic_hinge
```

## Notes

- `app/services` is the pure orchestration layer used by Streamlit pages/components: it prepares inputs, calls core solver routines, and formats serializable outputs.
- `src/dc_solver` is framework-agnostic core logic (FEM model assembly, static/dynamic integration, modal analysis, post-processing helpers) and does not depend on Streamlit.
- `plastic_hinge` is preserved as legacy/research support code for hinge constitutive logic and related helpers.
- Modal workflow supports condensation of DOFs without mass as part of the eigensolver pipeline (tested in modal condensation tests).
- Steel/aluminum/RC sections are exposed as reusable property helpers to support experiments and app input setup.
