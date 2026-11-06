# Equipo, roles y propiedad de archivos

## 1. Integrantes

| # | Nombre | GitHub | Código | Rol | Respaldo |
|---|---|---|---|---|---|
| 1 | Isabella Ortiz Hernandez | `@Isabellaortizher-collab` | 2226215 | Datos y dataset | Juan Pablo |
| 2 | Juan Pablo Dávila Moncayo | `@Juanpablodavila` | 2216168 | Generador U-Net | Jeremy |
| 3 | Jeremy Santiago Caicedo Cabezas | `@JeremyCaicedo` | 2210721 | Discriminador y pérdidas | Juan Pablo |
| 4 | Cristian Fabian Muñoz Piedrahita | `@CristianMP355` | 2230865 | Entrenamiento | Juan Sebastian |
| 5 | Juan Sebastian Castillo Acevedo | `@Juansec01` | 2231921 | Evaluación y demo | Cristian |
| 6 | Isabela | `@Isabela1105` | _(completar)_ | CI/CD, integración e informe | Isabella Ortiz |

> **Pendiente:** confirmar nombre completo y código de `@Isabela1105`. Si el rol de Isabela debe
> ser otro, intercambiarlo antes de crear las ramas; el plan de commits es simétrico y se puede
> reasignar sin romper dependencias.

---

## 2. Propiedad de archivos

| Área | Dueño | Archivos |
|---|---|---|
| Datos | Isabella Ortiz | `src/gan/data/`, `scripts/build_dataset.py`, `scripts/check_dataset.py`, `tests/test_dataset.py`, `data/README.md`, `docs/DATASET.md` |
| Generador | Juan Pablo | `src/gan/models/blocks.py`, `src/gan/models/generator.py`, `tests/test_models.py` (generador), `docs/ARQUITECTURA.md` (U-Net) |
| Discriminador/pérdidas | Jeremy | `src/gan/models/discriminator.py`, `src/gan/losses.py`, `tests/test_losses.py`, `docs/ARQUITECTURA.md` (PatchGAN/pérdidas) |
| Entrenamiento | Cristian | `src/gan/train.py`, `src/gan/config.py`, `configs/`, `src/gan/utils/viz.py`, `tests/test_smoke_train.py` |
| Evaluación/demo | Juan Sebastian | `src/gan/evaluate.py`, `src/gan/infer.py`, `app/app.py`, `tests/test_metrics.py`, `tests/test_app.py`, `docs/img/` |
| CI/informe | Isabela | `.github/`, `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `requirements*.txt`, `report/`, `docs/PARTICIPACION.md`, `docs/CI_CD.md` |

**Regla:** si alguien necesita cambiar un archivo de otra área, abre un issue o pide revisión al
dueño. No se pisan archivos sin avisar.

---

## 3. Contratos entre áreas

| Interfaz | Dueño | Consumidor | Contrato |
|---|---|---|---|
| `build_dataset` | Isabella | Cristian | `(input, target)` float32 `(256,256,3)` en `[-1,1]` |
| `build_generator` | Juan Pablo | Cristian | `(B,256,256,3)` → `(B,256,256,3)` en `[-1,1]` |
| `build_discriminator` | Jeremy | Cristian | `[(B,256,256,3), (B,256,256,3)]` → `(B,30,30,1)` logits |
| `generator_loss` / `discriminator_loss` | Jeremy | Cristian | escalares |
| `train` | Cristian | Juan Sebastian | checkpoint + `metrics.csv` |
| `evaluate` | Juan Sebastian | Todos | `metrics.csv` + figuras |
| CI | Isabela | Todos | checks verdes antes de merge |

Los tests de forma son el árbitro. Si un contrato cambia, se actualiza el test y el documento
`docs/ARQUITECTURA.md` en el mismo PR.

---

## 4. Revisores

| PR de | Revisor principal | Segundo revisor (rutas críticas) |
|---|---|---|
| Isabella (datos) | Juan Pablo | Cristian |
| Juan Pablo (generador) | Jeremy | Cristian |
| Jeremy (pérdidas) | Juan Pablo | Cristian |
| Cristian (train) | Juan Sebastian | Juan Pablo |
| Juan Sebastian (eval/app) | Cristian | Isabela |
| Isabela (CI/docs) | Isabella Ortiz | Jeremy, Juan Sebastian |

**Regla:** nadie aprueba su propio PR. Las rutas `src/gan/data/`, `src/gan/losses.py` y
`src/gan/train.py` requieren 2 aprobaciones.

---

## 5. Responsabilidades transversales

| Actividad | Todos | Coordina |
|---|---|---|
| Escribir código de su módulo | Sí | — |
| Revisar PRs de otros | Sí (mín. 4) | Isabela |
| Escribir sección del informe | Sí | Isabela |
| Ensayo de defensa | Sí | Isabela |
| Actualizar `docs/PARTICIPACION.md` | Sí | Isabela |
| Reportar bloqueos el mismo día | Sí | Isabela |

---

## 6. Reuniones

| Reunión | Cuándo | Duración | Salida |
|---|---|---|---|
| Planeación | Lunes | 30 min | Issues de la semana asignados |
| Bloqueos | Miércoles | 15 min | Qué falta y quién ayuda |
| Demo + merge | Viernes | 45 min | PRs revisados/mergeados, demo |
| Ensayo defensa | Semana 4 (2 veces) | 30 min | Presentador aleatorio, preguntas |

Las decisiones técnicas se registran en `docs/DECISIONES.md` o en el issue, no solo en WhatsApp.

---

## 7. Evidencia de participación

Cada integrante debe tener:

- **14–15 commits significativos** (no triviales).
- **6 PRs autorados**.
- **4–5 revisiones** a PRs de otros.
- **1 sección del informe**.
- **1 bloque de defensa** (y capacidad de explicar todos).
- **1 ensayo** con presentador aleatorio.

Verificación:

```bash
git shortlog -sne --all
git log --author="Nombre" --oneline
```

En GitHub: **Insights → Contributors**, **Pull Requests → Author/Reviewer**, **Issues →
Assignee**.

---

## 8. Si alguien se atrasa

1. Avisa en el issue el mismo día.
2. Pide pairing al respaldo.
3. Toma una tarea de `docs/` o `tests/` mientras se desbloquea.
4. El respaldo puede tomar el PR, pero el autor original debe revisarlo y entenderlo.
5. No se deja un módulo sin dueño: si es necesario, se reasigna el issue.

---

## 9. Cambio de rol

Si el equipo decide intercambiar roles antes de empezar:

1. Actualizar esta tabla y `docs/PLAN_COMMITS.md`.
2. Actualizar `.github/CODEOWNERS`.
3. Crear las ramas con los nombres nuevos.
4. No cambiar roles después del día 7: genera conflictos y desordena la evidencia.
