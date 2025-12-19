# Abaqus-like logs example

Run a beam example and stream the message file:

```bash
python -m dc_solver.run examples/abaqus_like/beam_cantilever_tipload.inp \
  --abaqus-like-logs --output-dir results/beam_cantilever_logs

# In another terminal:
tail -f results/beam_cantilever_logs/beam_cantilever_tipload.msg
```

The run produces:

- `beam_cantilever_tipload.sta`
- `beam_cantilever_tipload.msg`
- `beam_cantilever_tipload.dat`
