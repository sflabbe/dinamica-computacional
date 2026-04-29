# Problem 4 audit: orientación de rótulas fiber en viga

## Contexto

En Problem 4 se usa una viga con sección RC asimétrica (acero top/bottom distinto) y dos rótulas no lineales en los extremos. En esta configuración, un error de convención de signo entre extremos puede introducir deriva lateral artificial bajo carga gravitatoria simétrica.

## Por qué SHM puede esconder el problema

El modelo SHM tipo momento-rotación suele tener una respuesta casi impar en `M(θ)` y más “tolerante” a convenciones locales inconsistentes entre extremos. Por eso, un signo mal mapeado puede no explotar inmediatamente en métricas globales.

## Por qué fiber no perdona con RC asimétrica

En un modelo fiber, la curvatura gobierna deformaciones de fibras y, con armado asimétrico top/bottom, la respuesta no es simétrica por construcción. Un signo incorrecto de curvatura en un extremo cambia qué fibras entran en compresión/tracción y rompe la simetría física del estado gravitatorio.

## Convención requerida en la rótula derecha

La rótula derecha debe usar mapeo local coherente con la izquierda para que ambas “vean” la misma curvatura física cuando la gravedad es simétrica. En la práctica, eso implica signo opuesto en la conversión rotación→momento/curvatura del extremo derecho respecto del izquierdo.

## Invariante de regresión

Bajo gravedad simétrica, el invariante esperado es:

- convergencia del solve estático,
- drift promedio de techo prácticamente nulo,
- promedio de `ux` de nodos de techo prácticamente nulo.

Este invariante queda sellado por `tests/test_problem4_fiber_gravity_symmetry.py`.
