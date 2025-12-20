# Dinámica Computacional — pórtico + rótulas plásticas (N–M) + fibras

Este repo contiene un solver estructural 2D (frame) con:
- rótulas elasto-plásticas **acopladas N–M** (superficie poligonal + return mapping),
- rótula tipo **SHM** para vigas (histeresis degradada),
- y un **Problema 5** para construir la curva de interacción N–M desde una **sección de fibras 2D**.

Incluye integración en el tiempo con selección por CLI:

- `hht` (HHT-α, default)
- `newmark` (Newmark-β)
- `explicit` (Velocity Verlet)

> Salidas: por defecto se guardan en `./outputs/`.

---

## Instalación rápida

Recomendado (editable install):

```bash
python -m pip install -U pip
python -m pip install -e .
```

Alternativa (sin instalar): varios scripts en `src/problems/` se pueden ejecutar directo con `python src/problems/...py`.

---

## Integradores (HHT / Newmark / Explicit)

En los problemas dinámicos y en el runner, usa:

```bash
--integrator hht
--integrator newmark
--integrator explicit
```

Ejemplos:

```bash
python -m problems.problema4_portico --integrator hht
python -m problems.problema4_portico --integrator newmark
python -m problems.problema4_portico --integrator explicit
```

Runner genérico:

```bash
python -m dc_solver.run inputs/portal_frame.inp --integrator newmark
```

---

## Ejecutar problemas

### Problema 2 — rótula N–M (axial / flexión / combinado)

```bash
python src/problems/problema2_interaccion.py
python src/problems/problema2_hinge_nm_verification.py
```

Exporta automáticamente:
- `problem2_*_paths*.png` (trayectorias N–M)
- `problem2_*_hysteresis*.png` (M–θ y N–ε)
- versiones `_gradient.png` con **gradiente de color por step** (para ver el tiempo).

> En **flexión pura** se controla `N≈0` resolviendo `Δε` por bisección (evita serrucho artificial por restricción axial).

### Problema 4 — pórtico (time-history) con rótulas

```bash
python -m problems.problema4_portico --integrator hht
```

Exporta:
- drift vs tiempo, Vb vs drift
- estados (U/S) en varios snapshots
- `problem4_hinge_hysteresis_gradient.png` + CSVs por rótula seleccionada
  (gradiente de color por **t [s]**)

### Problema 5 — sección RC por fibras 2D → curva N–M

```bash
python -m problems.problema5_fiber_section_interaction
```

Asume:
- acero **A420** (fy=420 MPa)
- hormigón **C20/25** (fc≈20 MPa)

y genera malla 2D de fibras + interacción N–M (convex hull).

---

## Ejecutar todo (2–3–4–5)

```bash
python -m problems.run_all_problems_2_3_4
```

---

## Estructura (alto nivel)

- `plastic_hinge/` : return mapping 2D + superficie poligonal N–M
- `src/dc_solver/` : FEM + integradores + post-proceso
- `src/problems/`  : scripts reproducibles por problema
---

## Aceleración opcional con Numba (JIT)

Si instalas **Numba**, se aceleran kernels numéricos que están en el hot-path:

- proyección / return mapping de la rótula N–M (`plastic_hinge/return_mapping.py`)
- integración de secciones por fibras (`plastic_hinge/fiber_section.py`)

Instalación (Python ≥ 3.10 recomendado):

```bash
python -m pip install -e ".[numba]"
```

Desactivar JIT (por si estás debuggeando o si Numba no anda bien en tu plataforma):

- PowerShell:
  ```powershell
  $env:DC_USE_NUMBA="0"
  python -m problems.problema4_portico --integrator explicit
  ```
- Bash:
  ```bash
  DC_USE_NUMBA=0 python -m problems.problema4_portico --integrator explicit
  ```

---

## Job Infrastructure & Reporting (Estilo Abaqus/CalculiX)

El repo ahora incluye infraestructura profesional de "JOB" con reporting robusto, inspirado en Abaqus, CalculiX y SAP2000.

### Características

- **JobRunner context manager**: orquesta todo el ciclo de vida de un análisis
- **Tracking de archivos**: detecta automáticamente todos los outputs generados
- **Verbosidad estilo Abaqus**: bloques `JOB START` / `JOB END` con métricas
- **Archivos de salida estándar**:
  - `.msg` — mensajes, warnings, iteraciones
  - `.sta` — status incremental (step/inc/time/dt)
  - `.dat` — resumen final + **JOB TOTALS** (CPU, FLOPs, etc.)
  - `journal.log` — log cronológico de eventos
  - `*_runinfo.json` y `*_runinfo.txt` — metadata completa
- **Serialización JSON-safe**: maneja automáticamente `np.ndarray`, `Path`, `datetime`, etc.
- **Estimación de FLOPs**: métricas de performance (GFLOP/s estimado)
- **Progress printing**: impresión incremental durante análisis largos

### Estructura de Código

```
src/dc_solver/
  job/
    runner.py          # JobRunner (context manager principal)
    file_tracker.py    # snapshot/diff de archivos
    journal.py         # journal.log writer
    console.py         # pretty printing (progress, headers)
    flops.py           # estimación de FLOPs
  reporting/
    run_info.py        # runinfo txt/json + to_jsonable()
    abaqus_like.py     # writers .msg/.sta/.dat
    events.py          # event definitions
  fem/                 # model, nodes, elements
  integrators/         # hht_alpha, newmark, explicit
  post/                # plotting, hinge_exports, fiber_mesh_plot
  materials/           # elastic, etc.
```

### Ejemplo de Uso

Ver: `examples/demo_job_infrastructure.py`

```python
from dc_solver.job import JobRunner, print_progress, should_print_progress

with JobRunner(job_name="mi_analisis", output_dir="outputs/run1", meta={...}) as job:
    # Configurar parámetros para estimación de FLOPs
    job.set_analysis_params(ndof=300, n_steps=10000, integrator="explicit")

    # Tu análisis aquí...
    for i in range(n_steps):
        # ... solve ...

        # Imprimir progreso periódicamente
        if should_print_progress(i, n_steps, print_every_pct=5.0):
            print_progress(i, n_steps, t=t_current, dt=dt, drift_peak=drift_max)

    # Marcar éxito
    job.mark_success()
```

Al salir del contexto, se generan automáticamente:
- `.msg`, `.sta`, `.dat`, `journal.log`
- `mi_analisis_runinfo.json` / `.txt` (con lista de archivos creados)
- Impresión de **JOB END** con tiempos, FLOPs, y lista de outputs

### Ejecutar el Demo

```bash
python examples/demo_job_infrastructure.py
```

Outputs en: `outputs/demo_job/<timestamp>__explicit__sdof/`

### Runinfo JSON-Safe

La función `to_jsonable()` en `reporting/run_info.py` convierte automáticamente:

- `np.ndarray` → `list`
- `np.float64`, `np.int64` → `float`, `int`
- `Path` → `str`
- `datetime` / `date` → isoformat string
- `set`, `tuple` → `list`
- `dict` con keys no-string → keys convertidas a `str`

Esto **elimina el error típico**:
```
TypeError: Object of type ndarray is not JSON serializable
```

**Todos** los problemas existentes (2, 3, 4, 5) ya usan `write_run_info()` internamente, que aplica `to_jsonable()` automáticamente.
