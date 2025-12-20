# Examples

This folder contains **Abaqus-like input decks** used by the input parser and automated tests.

For the verification / research-style scripts, run the `problems` modules instead:

```bash
python -m problems.problema2_interaccion
python -m problems.problema3_shm_verification
python -m problems.problema4_portico --beam-hinge compare --integrator hht
python -m problems.problema5_fiber_section_interaction --mode all
```

Outputs are written to `./outputs/`.
