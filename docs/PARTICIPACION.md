# Participación del equipo

Esta tabla se reconstruyó a partir del historial real de `main`
(`git log --format='%an %s' origin/main`), de las ramas remotas y de los notebooks de Colab.
Donde el **rol asignado** no coincide con **quién hizo el trabajo**, se deja escrito: la
evidencia manda sobre el plan.

Verificación:

```bash
git shortlog -sne origin/main
git log origin/main --author="Nombre" --oneline
```

En GitHub: **Insights → Contributors** y **Pull requests → Author / Reviewer**.

## 1. Tabla general

Estado de `main` al 26 de noviembre de 2026 (último merge `#57`). PRs mergeados por autor:
`gh pr list --state merged --author <usuario>` o *Pull requests → Closed* en GitHub.

| Integrante | GitHub | Rol asignado | PRs mergeados en `main` | Commits en `main` | Aporte verificable |
|---|---|---|---|---|---|
| Isabella Ortiz Hernandez | @Isabellaortizher-collab | Datos y dataset | #10, #11, #12, #15, #16, #17 | 6 | Dataset completo: degradaciones, build con manifiesto, estadísticas, conjunto real, alfa/EXIF, cargador `tf.data`. |
| Juan Pablo Dávila Moncayo | @JP-64-3010 | Generador U-Net | #56 | 1 | Subió el informe final en Word y PDF (`report/Informe_Pix2Pix.*`, #56). El generador lo implementó Jeremy (#18). |
| Jeremy Santiago Caicedo Cabezas | @JeremyCaicedo | Discriminador y pérdidas | #18, #21 | 3 | Bloques y generador U-Net (#18), discriminador PatchGAN y pérdidas (#21, issue #20), ajuste directo en `losses.py`. |
| Cristian Fabian Muñoz Piedrahita | @CristianMP355 | Entrenamiento | #25, #26, #27, #28, #29 | 5 | Bucle D→G, smoke test, integración real, métricas y ablación, checkpoints atómicos para Colab. |
| Juan Sebastian Castillo Acevedo | @Juansec01 | Evaluación, inferencia y demo | #30–#42, #46–#55, #57 + andamiaje inicial | 61 | Estructura inicial del repo; métricas, evaluación, inferencia y demo; corridas de 100 épocas, selección por validación, TTA, limpieza de polvo, ablación justa y experimentos (`docs/RESULTADOS.md` §6–7). |
| Isabela Cabezas Obregón | @Isabela1105 | CI/CD, integración e informe | #14 | 1 | `.gitattributes`; ejecutó la corrida base en Colab (20 épocas, `notebooks/pix2pix_entrenamiento_colab.ipynb`). Dueña del repositorio. |

## 2. Detalle por integrante

### Isabella Ortiz Hernandez (@Isabellaortizher-collab) · código 2226215

- **PRs:** #10 `feat(data): implementar dataset por pares y degradaciones sintéticas`,
  #11 `feat(data): crear script de construcción con manifiesto y validación`,
  #12 `feat(data): agregar estadísticas, galería, ficha y pruebas del dataset`,
  #15 `feat(data): agregar conjunto real y manejo de alfa y EXIF`,
  #16 `docs(data): actualizar cifras finales y cubrir casos borde`,
  #17 `feat(data): agregar cargador de datos para TensorFlow`.
- **Módulos:** `src/gan/data/`, `scripts/build_dataset.py`, `scripts/check_dataset.py`,
  `scripts/dataset_report.py`, `scripts/prepare_real.py`, `docs/DATASET.md`,
  `docs/dataset_stats.md`, `docs/img/dataset_grid.png`.
- **Resultado:** 152 fuentes → 456 pares (366 / 45 / 45), split 80/10/10 por fuente, sin fuga.
- **Informe y defensa:** §3 Dataset.

### Juan Pablo Dávila Moncayo (@JP-64-3010) · código 2216168

- **Rol asignado:** generador U-Net (B2–B6 en `docs/PLAN_COMMITS.md`).
- **PR:** #56 `docs(report): agregar informe final Pix2Pix` (commit `c2682fd`): informe final en
  Word y PDF, `report/Informe_Pix2Pix.docx` y `report/Informe_Pix2Pix.pdf`.
- **Código:** los PRs del generador (#18 en `main` y #22 en la rama `Jeremycaicedo-patch-1`) son
  de Jeremy y no se reasignan.
- **Informe y defensa asignados:** §4.2 Generador U-Net.

### Jeremy Santiago Caicedo Cabezas (@JeremyCaicedo) · código 2210721

- **PRs:** #18 `feat(generator): agregar bloques de construcción U-Net` (incluye
  `generator.py`), #21 `feat(model): implementar discriminador patchgan y pérdidas` (#20).
- **Commit directo:** `e682bc7` *Simplify label assignment in loss function*.
- **Ramas sin mergear:** `feat/generator-unet`, `feat/discriminator-losses`,
  `Jeremycaicedo-patch-1` (su contenido ya está en `main` por #18 y #21).
- **Módulos:** `src/gan/models/blocks.py`, `src/gan/models/generator.py`,
  `src/gan/models/discriminator.py`, `src/gan/losses.py`, `tests/test_models.py`,
  `tests/test_losses.py`.
- **Informe y defensa:** §2 Marco teórico, §4.2–4.4 (generador, PatchGAN y pérdidas).

### Cristian Fabian Muñoz Piedrahita (@CristianMP355) · código 2230865

- **PRs:** #25 `feat(train): implementar train_step D->G con checkpoints y TensorBoard`,
  #26 `test(train): smoke test de dos pasos en CPU`,
  #27 `feat(train): integrate real data loaders, models and losses`,
  #28 `feat(train): add metrics logging, D->G validation and ablation experiment`,
  #29 `fix(train): harden atomic checkpoints for Colab and update docs`.
- **Módulos:** `src/gan/train.py`, `src/gan/config.py`, `src/gan/utils/viz.py` (muestras y
  TensorBoard), `configs/`, `tests/test_smoke_train.py`, `tests/test_train_integration.py`,
  `docs/ARQUITECTURA.md` §5.
- **Colab:** el notebook original de base + ablación (empaquetaba sus salidas como
  `resultados_pix2pix_cristian.zip`) se unificó en `notebooks/pix2pix_colab.ipynb`.
- **Informe y defensa:** §5 Entrenamiento.

### Juan Sebastian Castillo Acevedo (@Juansec01) · código 2231921

- **Commits directos iniciales:** `98000b4` `chore: inicializar repositorio…` (43 archivos:
  CI, release, plantillas, CODEOWNERS, `pyproject.toml`, configs, docs y guías; corresponde
  al PR-F1 del plan) y `a82995c` (datos de Isabela en el README).
- **PRs de cierre** #30–#36 (abiertos desde `@Juansec01`, apilados; se mergean en orden):

| PR · rama | Commits | Qué entrega |
|---|---|---|
| [#30](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/30) `feat/16-metrics-inference` | 3 | PSNR/SSIM/L1 en [0, 1] por imagen (`src/gan/metrics.py`), inferencia desde checkpoint (`src/gan/infer.py`), tests. |
| [#31](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/31) `docs/17-baselines` | 1 | Protocolo, métricas, líneas base y plantilla de tabla (`docs/ARQUITECTURA.md` §6). |
| [#32](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/32) `feat/26-eval-app` | 5 | Evaluación del test con baselines y CSVs, grids, demo FastAPI + `web/`, Gradio de respaldo, tests. |
| [#33](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/33) `feat/33-results-app` | 4 | FID opcional, selector de ejemplos, `scripts/results_table.py`, `docs/DEMO.md`. |
| [#34](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/34) `fix/37-app-formats` | 3 | Fotos de 12 MP, alfa, EXIF, modos raros, relleno sin deformar; curvas reales y `docs/RESULTADOS.md`; notebooks de Colab. |
| [#35](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/35) `test/41-synthetic-sample` | 1 | `scripts/make_synthetic_sample.py` (PR-F3 pendiente): el CI deja de saltarse el smoke training. |
| [#36](https://github.com/Isabela1105/Trabajo-Final-de-Redes/pull/36) `docs/43-cierre-participacion` | 9 | README, participación, equipo, informe y referencias alineados con el trabajo real; notebooks de Colab unificados; `docs/GUIA_PRUEBAS.md`. |

- **Informe y defensa:** §6 Experimentos y resultados, §7 Demo.

### Isabela Cabezas Obregón (@Isabela1105) · código 2226013

- **PRs:** #14 `chore: agregar gitattributes para estadísticas de lenguajes`.
- **Colab:** ejecutó la corrida base de 20 épocas (notebook `Pix2Pix_metricas_perdidas_existentes`,
  en su Drive). Sus pesos, `metrics.csv`, logs y muestras están en su Drive en
  `pix2pix_outputs/metricas_loss_existente/`.
- **Rol:** dueña del repositorio y de la protección de `main`; responsable del tag `v1.0.0` y
  del release con el PDF del informe (`.github/workflows/release.yml`).
- **Pendiente de su plan:** F4 (quality gates), F5 (informe y release) y F6 (participación);
  el generador de datos sintéticos (F3) quedó cubierto en `test/41-synthetic-sample`.
- **Informe y defensa asignados:** §1 Introducción, §9 Conclusiones y anexos.

## 3. Evidencia de entrenamiento (Colab)

| Notebook | Estado | Resultado |
|---|---|---|
| `notebooks/pix2pix_entrenamiento_colab.ipynb` | ejecutado por Isabela Cabezas | 20 épocas en GPU, 522 s; val L1 0.1482, PSNR 20.26 dB, SSIM 0.5077 (`docs/RESULTADOS.md`). |
| `notebooks/pix2pix_colab.ipynb` | unifica el de base + ablación (falló en la celda 1) con el de métricas, más evaluación y demo | pendiente de correr la evaluación del test y la ablación. |

Los checkpoints, `metrics.csv`, logs y muestras están en la carpeta de Drive del equipo
(`archivos_deep_learning` y `pix2pix_outputs/metricas_loss_existente`), no en git.

## 4. Declaración

Cada integrante confirma que:

- [ ] Entiende y puede explicar el código de su módulo.
- [ ] Puede ejecutar el pipeline completo (dataset, smoke train, inferencia).
- [ ] Participó en al menos una revisión de otro módulo.
- [ ] Escribió o revisó una sección del informe.
- [ ] Participó en al menos un ensayo de defensa.
