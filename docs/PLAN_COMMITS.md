# Plan de entrega de commits por integrante

> **Cómo leer este documento.** Cada fila es un **Pull Request** (una rama). Dentro de cada PR
> hay 1–4 commits. Los IDs `PR-A1`, `PR-B2`, etc. son internos del plan; GitHub asignará los
> números reales. Los días son **días hábiles** de un plan de 4 semanas (28 días). Si la entrega
> tiene otra fecha, se comprime o se estira proporcionalmente.

**Equivalencias:** **A** = Isabella Ortiz (`@Isabellaortizher-collab`) · **B** = Juan Pablo Dávila
(`@Juanpablodavila`) · **C** = Jeremy Caicedo (`@JeremyCaicedo`) · **D** = Cristian Muñoz
(`@CristianMP355`) · **E** = Juan Sebastian Castillo (`@Juansec01`) · **F** = Isabela
(`@Isabela1105`). La tabla completa está en [`EQUIPO.md`](EQUIPO.md).

---

## 1. Reglas del plan

1. **Cada integrante hace su primer commit en los días 1–2.** La participación debe verse desde
   el inicio, no al final.
2. **Mínimo 12 commits significativos por persona.** Este plan propone **14–15**.
3. **Cada PR tiene un dueño y un revisor distinto.** Nadie aprueba su propio PR.
4. **Cada commit deja el repo ejecutable** (o al menos la rama compila y los tests pasan).
5. **No se mergea con CI rojo.**
6. **Los PRs de modelos/datos llevan captura o métrica** antes/después.
7. **Comentarios y docstrings en español**, identificadores en inglés. Cada archivo tiene un
   encabezado con propósito, autor y contrato de entrada/salida.
8. **Después del día 24 no se agregan features.** Solo informe, figuras, defensa y correcciones.
9. **Si alguien se atrasa:** avisa en el issue el mismo día, pide pairing y toma una tarea de
   `docs/` o `tests/` para no bloquear la cadena crítica.

---

## 2. Resumen de entregas

| Persona | Rol | PRs | Commits | Semana 1 | Semana 2 | Semana 3 | Semana 4 |
|---|---|---|---|---|---|---|---|
| **A** | Datos / dataset | 6 | 14 | 5 | 4 | 3 | 2 |
| **B** | Generador U-Net | 6 | 14 | 5 | 4 | 3 | 2 |
| **C** | Discriminador / pérdidas | 6 | 14 | 5 | 4 | 3 | 2 |
| **D** | Entrenamiento | 6 | 15 | 5 | 4 | 4 | 2 |
| **E** | Evaluación / demo | 6 | 15 | 5 | 4 | 4 | 2 |
| **F** | CI/CD / QA / informe | 6 | 15 | 6 | 4 | 3 | 2 |
| **Total** | | **36** | **87** | **31** | **24** | **20** | **12** |

> 36 PRs es un ritmo manejable: ~9 PRs por semana para 6 personas. Si un PR queda muy grande,
> se parte en dos, pero **no se acumula trabajo de 2 semanas en un solo PR**.

---

## 3. Persona A — Datos y dataset

**Archivos que posee:** `src/gan/data/`, `scripts/build_dataset.py`, `scripts/check_dataset.py`,
`tests/test_dataset.py`, `data/README.md`, `docs/DATASET.md`.
**Respaldo:** Persona B (para que alguien más pueda explicar el pipeline).

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-A1** | 1 | `docs/03-protocolo-dataset` | `docs(data): definir protocolo de recolección y criterios de calidad` | `docs/DATASET.md`, `data/README.md` | F |
| **PR-A2** | 2–4 | `feat/07-dataset-pairs` | `feat(data): implementar emparejamiento y split por fuente`<br>`feat(data): agregar degradaciones sintéticas configurables`<br>`test(data): validar formas, rangos y no fuga entre splits` | `src/gan/data/dataset.py`, `src/gan/data/degradations.py`, `tests/test_dataset.py` | B, D |
| **PR-A3** | 5 | `feat/09-build-dataset` | `feat(data): crear script de construcción con manifiesto y validación` | `scripts/build_dataset.py`, `scripts/check_dataset.py` | F |
| **PR-A4** | 8–11 | `feat/14-dataset-v2` | `feat(data): generar tres variantes por fuente y estadísticas`<br>`fix(data): corregir orden de degradaciones y semilla`<br>`docs(data): completar ficha del dataset y galería`<br>`test(data): agregar prueba de reproducibilidad del manifiesto` | `src/gan/data/`, `data/README.md`, `docs/img/dataset_grid.png`, `tests/test_dataset.py` | B, C, E |
| **PR-A5** | 15–18 | `feat/21-real-eval` | `feat(data): agregar conjunto real para evaluación cualitativa`<br>`docs(data): documentar licencias, sesgos y limitaciones`<br>`fix(data): manejar imágenes con alfa y EXIF` | `data/README.md`, `docs/DATASET.md`, `src/gan/data/dataset.py` | E, F |
| **PR-A6** | 22–24 | `docs/28-dataset-final` | `docs(data): actualizar cifras finales del dataset`<br>`test(data): cubrir casos borde y cobertura final` | `data/README.md`, `docs/DATASET.md`, `tests/test_dataset.py` | D |

**Entregable clave de A:** un comando reproducible que construye el dataset desde `data/raw/` y
deja `train/`, `val/`, `test/` con un manifiesto que impide fuga de datos.

---

## 4. Persona B — Generador U-Net

**Archivos que posee:** `src/gan/models/blocks.py`, `src/gan/models/generator.py`,
`docs/ARQUITECTURA.md` (sección generador), `tests/test_models.py` (generador).
**Respaldo:** Persona C.

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-B1** | 1 | `docs/04-contrato-modelos` | `docs(model): definir contrato de tensores y diagrama U-Net` | `docs/ARQUITECTURA.md` | C |
| **PR-B2** | 2–4 | `feat/10-generator-unet` | `feat(model): implementar bloques encoder y decoder`<br>`feat(model): implementar generador U-Net configurable`<br>`test(model): verificar formas 64/128/256 y rango tanh` | `src/gan/models/blocks.py`, `src/gan/models/generator.py`, `tests/test_models.py` | C, D |
| **PR-B3** | 5 | `docs/11-generator-layers` | `docs(model): agregar tabla de capas y conteo de parámetros` | `docs/ARQUITECTURA.md` | F |
| **PR-B4** | 8–11 | `refactor/17-generator-norm` | `refactor(model): parametrizar normalización y dropout`<br>`fix(model): corregir concatenación de skip connections`<br>`test(model): agregar prueba de gradientes del generador`<br>`docs(model): explicar decisiones de diseño y alternativas` | `src/gan/models/`, `tests/test_models.py`, `docs/ARQUITECTURA.md` | C, D |
| **PR-B5** | 15–19 | `feat/24-generator-tuning` | `feat(model): agregar inicialización y variantes de ngf`<br>`perf(model): reducir memoria en 256x256`<br>`docs(model): preparar guion de explicación del generador` | `src/gan/models/generator.py`, `docs/DEFENSA.md` | D, F |
| **PR-B6** | 22–24 | `fix/30-generator-borders` | `fix(model): manejar imágenes no cuadradas sin deformar`<br>`test(model): congelar pruebas de arquitectura` | `src/gan/models/generator.py`, `tests/test_models.py` | A |

**Entregable clave de B:** generador configurable que funciona con 64, 128 y 256, con tests de
forma y una explicación capa por capa lista para la defensa.

---

## 5. Persona C — Discriminador y pérdidas

**Archivos que posee:** `src/gan/models/discriminator.py`, `src/gan/losses.py`,
`docs/ARQUITECTURA.md` (sección discriminador/pérdidas), `tests/test_losses.py`.
**Respaldo:** Persona B.

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-C1** | 1 | `docs/05-contrato-discriminador` | `docs(model): definir contrato y diagrama del discriminador` | `docs/ARQUITECTURA.md` | B |
| **PR-C2** | 2–4 | `feat/12-patchgan-losses` | `feat(model): implementar discriminador PatchGAN`<br>`feat(loss): implementar pérdidas adversarial y L1`<br>`test(model): verificar parches 30x30 y pérdidas finitas` | `src/gan/models/discriminator.py`, `src/gan/losses.py`, `tests/test_models.py`, `tests/test_losses.py` | B, D |
| **PR-C3** | 5 | `docs/13-loss-theory` | `docs(model): explicar PatchGAN y pérdida adversarial` | `docs/ARQUITECTURA.md` | F |
| **PR-C4** | 8–11 | `feat/20-loss-options` | `feat(loss): agregar label smoothing configurable`<br>`test(loss): verificar valores con casos conocidos`<br>`feat(loss): permitir ponderación de L1 por configuración`<br>`docs(model): comparar BCE vs MSE y justificar la elección` | `src/gan/losses.py`, `tests/test_losses.py`, `docs/ARQUITECTURA.md` | B, D |
| **PR-C5** | 15–19 | `feat/27-gradient-penalty` | `feat(loss): agregar gradient penalty experimental`<br>`test(loss): probar estabilidad numérica de las pérdidas`<br>`docs(model): preparar guion de explicación de pérdidas` | `src/gan/losses.py`, `tests/test_losses.py`, `docs/DEFENSA.md` | D, F |
| **PR-C6** | 22–24 | `fix/32-loss-nan` | `fix(loss): evitar NaN con logits extremos`<br>`docs(model): congelar ecuaciones y notación` | `src/gan/losses.py`, `docs/ARQUITECTURA.md` | B |

**Entregable clave de C:** PatchGAN con salida `(B, 30, 30, 1)`, pérdidas implementadas y
probadas, y la justificación matemática de λ = 100.

---

## 6. Persona D — Entrenamiento

**Archivos que posee:** `src/gan/train.py`, `src/gan/config.py`, `configs/`,
`src/gan/utils/viz.py`, `tests/test_smoke_train.py`.
**Respaldo:** Persona E.

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-D1** | 1 | `chore/06-config` | `chore(config): agregar cargador y validación de YAML` | `src/gan/config.py`, `configs/base.yaml` | E |
| **PR-D2** | 2–4 | `feat/14-train-loop` | `feat(train): implementar train_step con tf.function`<br>`feat(train): agregar checkpoints y reanudación`<br>`feat(train): agregar TensorBoard y guardado de muestras` | `src/gan/train.py`, `src/gan/utils/viz.py` | B, C |
| **PR-D3** | 5 | `test/15-smoke-train` | `test(train): smoke test de dos pasos en CPU` | `tests/test_smoke_train.py`, `configs/smoke.yaml` | F |
| **PR-D4** | 8–11 | `feat/23-train-integration` | `feat(train): integrar dataset y modelo en el bucle`<br>`feat(train): agregar precisión mixta y clip de gradientes`<br>`feat(train): agregar early stopping por L1 de validación`<br>`docs(train): documentar hiperparámetros y reanudación` | `src/gan/train.py`, `configs/base.yaml`, `docs/ARQUITECTURA.md` | A, B, E |
| **PR-D5** | 15–20 | `feat/29-train-experiments` | `feat(train): registrar métricas en CSV y resumen`<br>`fix(train): corregir orden de pasos D/G`<br>`feat(train): agregar configuración de ablación sin GAN`<br>`docs(train): preparar guion del bucle de entrenamiento` | `src/gan/train.py`, `configs/`, `docs/DEFENSA.md` | C, E, F |
| **PR-D6** | 22–24 | `fix/34-colab-checkpoint` | `fix(train): estabilizar checkpoints en Colab`<br>`docs(train): congelar hiperparámetros finales` | `src/gan/train.py`, `docs/ARQUITECTURA.md` | E |

**Entregable clave de D:** un entrenamiento que corre de punta a punta, se reanuda tras una
desconexión de Colab y produce checkpoints + logs + muestras.

---

## 7. Persona E — Evaluación, inferencia y demo

**Archivos que posee:** `src/gan/evaluate.py`, `src/gan/infer.py`, `app/app.py`,
`tests/test_metrics.py`, `tests/test_app.py`, `docs/img/` (resultados).
**Respaldo:** Persona D.

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-E1** | 1 | `docs/06-protocolo-eval` | `docs(eval): definir métricas y protocolo de evaluación` | `docs/ARQUITECTURA.md` | B |
| **PR-E2** | 2–4 | `feat/16-metrics-inference` | `feat(eval): implementar PSNR, SSIM y L1`<br>`feat(infer): implementar inferencia sobre una imagen`<br>`test(eval): validar métricas con imágenes conocidas` | `src/gan/evaluate.py`, `src/gan/infer.py`, `tests/test_metrics.py` | C, D |
| **PR-E3** | 5 | `docs/17-baselines` | `docs(eval): describir líneas base y tablas de resultados` | `docs/ARQUITECTURA.md` | F |
| **PR-E4** | 8–11 | `feat/26-eval-app` | `feat(eval): evaluar test set y guardar CSV`<br>`feat(viz): generar grids input/target/predicción`<br>`feat(app): crear demo Gradio mínima`<br>`test(app): probar carga de modelo y predicción` | `src/gan/evaluate.py`, `src/gan/utils/viz.py`, `app/app.py`, `tests/test_app.py` | B, D, F |
| **PR-E5** | 15–20 | `feat/33-results-app` | `feat(eval): agregar FID opcional`<br>`feat(app): agregar ejemplos y selector de imágenes`<br>`docs(eval): documentar resultados cuantitativos y cualitativos`<br>`docs(app): documentar despliegue en Hugging Face Spaces` | `src/gan/evaluate.py`, `app/app.py`, `docs/ARQUITECTURA.md`, `README.md` | A, C, F |
| **PR-E6** | 22–24 | `fix/37-app-formats` | `fix(app): manejar imágenes grandes y formatos`<br>`docs(eval): congelar tablas y figuras finales` | `app/app.py`, `docs/ARQUITECTURA.md`, `docs/img/` | B |

**Entregable clave de E:** tabla PSNR/SSIM/L1 con baselines, grid de resultados y demo Gradio
que cualquiera puede correr con un comando.

---

## 8. Persona F — CI/CD, QA e informe

**Archivos que posee:** `.github/`, `README.md`, `CONTRIBUTING.md`, `pyproject.toml`,
`requirements*.txt`, `report/`, `docs/PARTICIPACION.md`, `docs/CI_CD.md`.
**Respaldo:** Persona A.

| PR | Días | Rama | Commits (mensaje exacto) | Archivos principales | Revisor |
|---|---|---|---|---|---|
| **PR-F1** | 1 | `chore/01-init` | `chore: inicializar repositorio con README, licencia y gitignore`<br>`ci: agregar workflow de lint, tests y smoke training` | `README.md`, `LICENSE`, `.gitignore`, `.github/workflows/ci.yml` | A |
| **PR-F2** | 2 | `chore/02-tooling` | `chore: agregar pre-commit y configuración de calidad`<br>`docs: agregar guía de contribución y plantillas` | `.pre-commit-config.yaml`, `pyproject.toml`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/` | B, C |
| **PR-F3** | 3 | `test/03-synthetic-data` | `test(ci): agregar generador de datos sintéticos para pruebas` | `scripts/make_synthetic_sample.py` | A |
| **PR-F4** | 8–11 | `ci/29-quality-gates` | `ci: agregar cache, cobertura y artefactos`<br>`ci: agregar verificación de título de PR`<br>`docs(ci): documentar protección de rama y flujo`<br>`chore: configurar Dependabot y etiquetas` | `.github/workflows/ci.yml`, `docs/CI_CD.md`, `.github/dependabot.yml` | D, E |
| **PR-F5** | 15–18 | `ci/37-report-release` | `ci: agregar build del informe en PDF`<br>`docs: actualizar README con resultados y demo`<br>`ci: agregar release automático en tags` | `.github/workflows/release.yml`, `report/Makefile`, `README.md` | B, E |
| **PR-F6** | 20–24 | `docs/40-participacion` | `docs: crear tabla de participación y evidencia`<br>`ci: congelar versiones y validar release`<br>`docs: revisión final y checklist de entrega` | `docs/PARTICIPACION.md`, `requirements*.txt`, `README.md` | A, C, E |

**Entregable clave de F:** `main` protegida, CI que bloquea merges rotos, release `v1.0.0` con
PDF y tabla de participación con evidencia.

---

## 9. Calendario de merges (ventanas)

Para evitar conflictos, los PRs se mergean en ventanas. Fuera de la ventana, el PR espera.

| Día | Ventana | PRs que entran | Responsable de la ventana |
|---|---|---|---|
| 1–2 | Fundaciones | F1, A1, B1, C1, D1, E1 | F |
| 3–5 | Herramientas + esqueletos | F2, F3, A2, B2, C2, D2, E2 | D |
| 6 | Cierre semana 1 | A3, B3, C3, D3, E3 | F |
| 8–11 | Integración | A4, B4, C4, D4, E4, F4 | D |
| 12 | Cierre semana 2 | Revisiones pendientes + demo interna | E |
| 15–20 | Experimentos | A5, B5, C5, D5, E5, F5 | D |
| 21 | Cierre semana 3 | Figuras, tabla de resultados, borrador del informe | F |
| 22–24 | Cierre de código | A6, B6, C6, D6, E6, F6 | F |
| 25 | **Congelamiento** | Tag `v1.0.0`, release, PDF | F |
| 26–28 | Solo docs/defensa | Correcciones de informe, ensayos, no código nuevo | Todos |

---

## 10. Cadena crítica y dependencias

```text
F1 (repo + CI)
  └─> D1 (config)
        ├─> A2/A3 (dataset) ──┐
        ├─> B2/B3 (generador) ─┼─> D2/D3 (train loop) ─> D4 (integración)
        └─> C2/C3 (D + losses) ┘                              │
                                                               ├─> entrenamiento real
E2/E3 (métricas + inferencia) ─────────────────────────────────┘        │
                                                                        ├─> E4/E5 (resultados + app)
                                                                        └─> informe + defensa
```

**Regla de desbloqueo:** si un PR depende de otro que no está mergeado, se puede avanzar con
**mocks** o datos sintéticos. Ejemplo: B y C pueden probar el generador y el discriminador con
tensores aleatorios antes de que A termine el dataset.

---

## 11. Definition of Done (para cada PR)

- [ ] El código compila y `pytest -q` pasa.
- [ ] `pre-commit run --all-files` pasa.
- [ ] Docstrings y comentarios en español; nombres en inglés.
- [ ] Tests nuevos o actualizados para el cambio.
- [ ] Documentación actualizada (si cambia una interfaz o hiperparámetro).
- [ ] Issue enlazado (`Closes #...`).
- [ ] Revisor asignado y aprobación.
- [ ] CI verde.
- [ ] Sin datos, pesos, `.env` ni archivos > 5 MB.
- [ ] Si cambia la salida visual: captura antes/después en el PR.

---

## 12. Evidencia de participación

Actualizar `docs/PARTICIPACION.md` al final de cada semana:

| Integrante | Módulos que posee | PRs autorados | PRs revisados | Commits | Sección del informe | Diapositivas |
|---|---|---|---|---|---|---|
| A | Datos, dataset, scripts | 6 | 4 | 14 | Dataset, ética | Dataset |
| B | Generador, bloques | 6 | 5 | 14 | Arquitectura (U-Net) | Generador |
| C | Discriminador, pérdidas | 6 | 5 | 14 | Marco teórico, pérdidas | Pérdidas |
| D | Entrenamiento, config | 6 | 5 | 15 | Entrenamiento, experimentos | Entrenamiento |
| E | Evaluación, inferencia, app | 6 | 4 | 15 | Resultados, demo | Resultados/demo |
| F | CI/CD, QA, informe | 6 | 5 | 15 | Introducción, conclusiones, anexos | Cierre |

Comandos de verificación:

```bash
git shortlog -sne --all
git log --author="Nombre" --oneline | wc -l
```

Y en GitHub: **Insights → Contributors**, **Pull requests → Author/Reviewer**, **Issues →
Assignee**.

---

## 13. Si el equipo va atrasado

Prioridad de recorte (de menor a mayor impacto):

1. **Opcional:** FID, gradient penalty, encuesta humana, despliegue en HF Spaces.
2. **Opcional:** ablación sin skip connections; dejar solo L1-only vs Pix2Pix.
3. **Reducir:** 256×256 → 128×128 y 100 → 50 épocas.
4. **Reducir:** 3 variantes por foto → 1; 500 fotos → 250.
5. **Nunca recortar:** dataset propio, entrenamiento, métricas, informe, participación de los 6,
   CI verde y defensa.

---

## 14. Primer día: checklist para cada integrante

```bash
# 1. Clonar
git clone <URL>
cd <repo>

# 2. Entorno
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt
pre-commit install

# 3. Sincronizar
git switch main
git pull --rebase

# 4. Rama propia (ejemplo Persona A)
git switch -c docs/03-protocolo-dataset

# 5. Primer commit pequeño pero real
# editar docs/DATASET.md
git add docs/DATASET.md
git commit -m "docs(data): definir protocolo de recolección y criterios de calidad (#3)"
git push -u origin HEAD

# 6. Abrir PR con la plantilla y asignar revisor
```

**Meta del día 1:** cada integrante tiene una rama y un PR abierto. Aunque sea un documento,
debe ser un aporte real, no un commit vacío.
