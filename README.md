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
