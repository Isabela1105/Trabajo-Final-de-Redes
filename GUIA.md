# Guía maestra — Miniproyecto de Redes Generativas

**Curso:** Redes Neuronales y Deep Learning
**Trabajo:** Aplicación de arquitecturas de modelos generativos a la generación de imágenes
**Equipo:** 6 integrantes
**Entrega:** repositorio + informe + sustentación

> Esta guía asume la recomendación técnica: **restauración de fotografías antiguas con Pix2Pix**.
> Si el grupo elige otro tema, se mantienen el flujo de trabajo, el CI/CD, el informe y el plan
> de commits; solo cambian la sección de datos y algunos detalles del modelo.

---

## 0. Cómo usar este kit

| Archivo | Para qué sirve |
|---|---|
| `GUIA.md` | Documento principal. Decisiones, flujo y checklist. |
| `docs/MODELOS.md` | Comparación de arquitecturas y justificación de Pix2Pix. |
| `docs/PLAN_COMMITS.md` | **Plan de commits por integrante**, semana por semana. |
| `docs/EQUIPO.md` | Nombres, usuarios, códigos, roles, propiedad de archivos y revisores. |
| `docs/MERGE_Y_REVISION.md` | Integración, orden de merge, revisión final y release. |
| `docs/REPARTO.md` | Roles, archivos asignados y respaldos. |
| `docs/CI_CD.md` | Configuración exacta de GitHub Actions y protección de `main`. |
| `docs/DATASET.md` | Cómo construir el dataset personalizado y su ficha. |
| `docs/INFORME.md` | Estructura del informe y responsables por sección. |
| `docs/DEFENSA.md` | Guion de sustentación y banco de preguntas. |
| `docs/REFERENCIA_CODIGO.md` | Código comentado de referencia. |
| `guias/` | Una guía por integrante (ramas, PRs, commits, defensa). |
| `plantillas/` | Archivos listos para copiar a la raíz del repositorio. |

**Orden de trabajo sugerido:**

1. Crear el repositorio en GitHub (público, con licencia MIT).
2. Copiar el contenido de `plantillas/` a la raíz del repo.
3. Copiar además `README.md` (este reemplaza al de la plantilla) y los documentos
   `docs/EQUIPO.md`, `docs/MERGE_Y_REVISION.md`, `docs/MODELOS.md`, `docs/PLAN_COMMITS.md`,
   `docs/DATASET.md`, `docs/CI_CD.md`, `docs/INFORME.md`, `docs/DEFENSA.md`,
   `docs/REFERENCIA_CODIGO.md` y la carpeta `guias/`.
4. Verificar que los usuarios reales ya estén en `CODEOWNERS`, `docs/PARTICIPACION.md` y README.
5. Proteger `main` y activar los checks de CI.
6. Crear los issues de la semana 1 y asignarlos.
7. Cada integrante crea su rama y hace su primer commit **en los primeros 2 días**.

---

## 1. Lo que pide el profesor (checklist literal)

Del PDF y de la imagen:

- [ ] Aplicación con modelos generativos en **TensorFlow/Keras o PyTorch**.
- [ ] **Dataset personalizado** construido por el grupo.
- [ ] Entrenamiento del modelo y capacidad de **explicar el código**.
- [ ] **Validación** del modelo (métricas + evidencia visual).
- [ ] **Explicación de la arquitectura** de red neuronal usada.
- [ ] Tema seleccionado de la lista (o propuesta aprobada).
- [ ] **Repositorio** con todos los archivos y link compartido.
- [ ] En el repo se debe apreciar la **participación de todos** en la generación de código.
- [ ] **Informe** entregado.
- [ ] Preparación para **sustentación**; el docente elige al azar a un integrante.
- [ ] No usar caballos→cebras, cebras→caballos, manzanas→naranjas ni naranjas→manzanas.

**Implicación crítica:** como el profesor puede elegir a cualquiera, **nadie puede ser solo
"el de los datos" o "el de la diapositiva"**. Todos deben poder ejecutar el pipeline, explicar
la arquitectura, las pérdidas, las métricas y una línea de código cualquiera. El plan de
commits y las revisiones cruzadas están diseñados para forzar eso.

---

## 2. Decisiones recomendadas

| Decisión | Recomendación | Por qué |
|---|---|---|
| Tema | **Restauración de fotografías antiguas** | Pares entrada/salida fáciles de generar, evaluación objetiva, demo vistosa. |
| Modelo | **Pix2Pix** (U-Net + PatchGAN) | GAN condicional para imagen→imagen con datos pareados. Explicable y reproducible. |
| Framework | **TensorFlow 2.x / Keras 3** | Es el recomendado en el enunciado y el tutorial oficial de Pix2Pix está en TF. |
| Dataset | Fotos propias + degradaciones sintéticas | Dataset personalizado, con *ground truth* para medir PSNR/SSIM. |
| Estrategia Git | **GitHub Flow** (ramas cortas + PR) | Mucho más simple que GitFlow para 6 personas y 4 semanas. |
| `main` | Protegida, sin pushes directos | Obliga a PR, revisión y CI verde. |
| Repo | **Público** | Evidencia de participación, Actions ilimitadas y link fácil para el docente. |
| Merge | **Squash merge** | Historial lineal y un commit limpio por PR. |
| Entrenamiento pesado | Colab / Kaggle | GitHub Actions no tiene GPU; el CI solo hace un *smoke test*. |

### ¿Por qué no otros modelos?

- **CycleGAN:** no necesita pares, pero entrena más lento, es inestable y la evaluación es
  subjetiva. Buena opción solo si el tema es día→noche o pintura→foto.
- **StyleGAN2:** requiere muchas imágenes, GPU y tiempo; no resuelve restauración con una
  entrada concreta.
- **SRGAN:** válido para super-resolución, pero añade *perceptual loss* y una arquitectura más
  difícil de defender en el tiempo disponible.
- **DCGAN:** genera desde ruido, no restaura una foto; no cumple bien el objetivo.

Detalle completo en `docs/MODELOS.md`.

---

## 3. Arquitectura del repositorio

```text
proyecto-gan/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # lint + tests + smoke training en cada PR
│   │   └── release.yml            # build del informe y release al crear un tag v*
│   ├── ISSUE_TEMPLATE/
│   │   └── tarea.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── app/
│   └── app.py                     # demo Gradio
├── configs/
│   ├── base.yaml                  # configuración real de entrenamiento
│   └── smoke.yaml                 # configuración mínima para CI
├── data/
│   ├── README.md                  # ficha y procedencia del dataset
│   ├── raw/.gitkeep               # fotos originales (ignoradas por git)
│   ├── processed/.gitkeep         # pares generados (ignorados por git)
│   └── sample/.gitkeep            # muestra sintética para tests/CI
├── docs/
│   ├── ARQUITECTURA.md
│   ├── DATASET.md
│   ├── DECISIONES.md              # registro corto de decisiones (ADR simplificado)
│   ├── DEFENSA.md
│   ├── EXPLICACION_CODIGO.md      # recorrido archivo → función → para qué sirve
│   ├── PARTICIPACION.md           # tabla persona → módulos → PRs → commits
│   └── img/                       # figuras generadas por el código
├── notebooks/
│   ├── 01_eda_dataset.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_resultados.ipynb
├── report/
│   ├── informe.md
│   ├── referencias.bib
│   └── figuras/                   # enlaces o copias de docs/img
├── scripts/
│   ├── build_dataset.py           # crea pares y splits
│   ├── check_dataset.py           # valida estructura y fuga de datos
│   └── make_synthetic_sample.py   # datos falsos para tests/CI
├── src/gan/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── degradations.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── generator.py
│   │   └── discriminator.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       └── viz.py
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   └── test_smoke_train.py
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

**Regla:** `data/raw`, `data/processed`, `outputs/` y los pesos (`*.keras`, `*.h5`, `*.pt`)
**no se suben a git**. Se comparten por release, Hugging Face Hub o Drive, y el README explica
cómo descargarlos.

---

## 4. Flujo de trabajo con 6 personas

### 4.1 Reglas de oro

1. **Nadie hace push directo a `main`.** Todo entra por Pull Request.
2. **Un issue = una rama = un PR.** PRs de máximo ~400 líneas cambiadas.
3. **El autor no mergea su propio PR.** Necesita al menos 1 aprobación (2 si toca
   `src/gan/data/`, `src/gan/losses.py` o `src/gan/train.py`).
4. **CI verde es obligatorio.** Si falla, no se mergea.
5. **Cada integrante hace su primer commit en los primeros 2 días.** No dejar la participación
   para el final.
6. **Cada semana, cada integrante:** al menos 1 PR propio, 1 revisión a otro y 1 aporte al
   informe o a la documentación.
7. **Todo el código se escribe en el repo.** No vale código en WhatsApp, Drive o correo.
8. **Pair programming se acredita** con `Co-authored-by:` en el commit.
9. **No se reescribe la historia de `main`.** Nada de `force push` a ramas compartidas.
10. **Antes de pedir revisión:** `pre-commit run --all-files`, `pytest -q` y probar el smoke.

### 4.2 Ramas

```text
main
├── feat/07-dataset-pairs          Isabella Ortiz
├── feat/10-generator-unet         Juan Pablo Dávila
├── feat/12-patchgan-losses        Jeremy Caicedo
├── feat/14-train-loop             Cristian Muñoz
├── feat/16-metrics-inference      Juan Sebastian Castillo
└── ci/02-workflow                 Isabela
```

Formato: `tipo/numero-issue-descripcion-corta`. Los números del plan son identificadores internos;
al crear el issue, usa el número real que asigna GitHub.

Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`.

### 4.3 Commits

Formato **Conventional Commits**:

```text
tipo(area): descripción en imperativo (#issue)
```

Ejemplos:

```text
feat(data): implementar emparejamiento y split por fuente (#7)
fix(model): corregir concatenación de skip connections (#17)
test(loss): verificar valores con casos conocidos (#20)
docs(train): documentar hiperparámetros y reanudación (#25)
ci: agregar cache y artefactos al workflow (#29)
```

- Un commit = un cambio coherente. No mezclar refactor + feature + docs.
- El mensaje explica **qué** y **por qué**, no "cambios".
- Si el commit rompe tests, se queda en la rama, nunca en `main`.

### 4.4 Pull Requests

Plantilla obligatoria (está en `plantillas/.github/PULL_REQUEST_TEMPLATE.md`):

- Qué cambia y por qué.
- Issue relacionado (`Closes #7`).
- Cómo se probó (comandos, capturas, métricas).
- Checklist: tests, lint, docs, comentarios, sin datos/pesos.
- Captura o GIF si cambia la salida visual del modelo.

Revisión:

- Revisar en menos de 24 h hábiles.
- Comentarios con `suggestion` cuando sea posible.
- Aprobar solo si se entiende el código; si no, pedir explicación. Eso también prepara la
  defensa.
- Rotar revisores: que nadie revise siempre a la misma persona.

### 4.5 Sincronización diaria

```bash
git switch main
git pull --rebase origin main
git switch mi-rama
git rebase main
# resolver conflictos, probar, continuar
git push --force-with-lease
```

`--force-with-lease` solo en ramas propias, nunca en `main`.

### 4.6 Evidencia de participación

El profesor puede revisar:

- **GitHub Insights → Contributors**: commits por persona.
- **`git shortlog -sne --all`**: autores y correos.
- **PRs autorados y revisados**.
- **Issues asignados y cerrados**.
- **`docs/PARTICIPACION.md`**: tabla con enlaces a PRs y commits.

Meta mínima: **12 commits significativos por persona** (el plan propone 14–16) y **mínimo 4
revisiones por persona**. Ningún integrante debe tener solo commits de `README` o de formato.

---

## 5. CI/CD: qué verifica y qué no

### 5.1 Filosofía

En un proyecto de ML, **CI no significa entrenar la GAN completa en cada push** (sería carísimo
y lento). El CI verifica que el código no se rompa; el entrenamiento real se hace en Colab/Kaggle
y se documenta. El CD publica el informe, los pesos y la demo cuando se crea un tag.

### 5.2 Pipeline de CI (cada PR y push a `main`)

1. **Lint:** `ruff check .`
2. **Formato:** `black --check .`
3. **Tests unitarios:** `pytest -q --cov=src`
4. **Datos sintéticos:** genera 8 imágenes de 64×64.
5. **Smoke training:** 2 pasos de entrenamiento en CPU con `configs/smoke.yaml`.
6. **Título de PR:** valida Conventional Commits.
7. **Artefactos:** guarda las imágenes y logs del smoke test.

Tiempo objetivo: **menos de 10 minutos**. Si tarda más, se cachea pip y se reduce el smoke.

### 5.3 CD (solo al crear un tag `v*`)

1. Construye el informe PDF con Pandoc.
2. Crea un GitHub Release con notas.
3. Adjunta el PDF y, si pesa poco, el generador en `.keras`.
4. Opcional: despliega la demo en Hugging Face Spaces (con aprobación manual).

### 5.4 Protección de `main`

Activar en **Settings → Branches → Add branch protection rule**:

- [x] Require a pull request before merging.
- [x] Require approvals: **1** (2 para rutas críticas si se usa CODEOWNERS).
- [x] Dismiss stale pull request approvals when new commits are pushed.
- [x] Require status checks to pass: `Calidad y pruebas` y `Título de PR`.
- [x] Require conversation resolution before merging.
- [x] Require linear history.
- [x] Do not allow bypassing the above settings.
- [x] Automatically delete head branches.

> Si el repo es privado y la cuenta no permite protection rules, la alternativa es repo público
> o GitHub Education/Team. Ver `docs/CI_CD.md`.

---

## 6. Plan de 4 semanas (resumen)

| Semana | Objetivo | Entregable verificable |
|---|---|---|
| 1 | Fundaciones | Repo, CI, dataset v1, generador, discriminador, pérdidas, train skeleton, métricas. |
| 2 | Integración | Primer entrenamiento real, checkpoints, evaluación, demo mínima, tests. |
| 3 | Experimentos | Entrenamiento completo, ablación, figuras, app desplegada, borrador del informe. |
| 4 | Cierre | Informe final, release `v1.0.0`, ensayo de defensa, checklist. |

El detalle por persona, rama, archivo, mensaje de commit, PR y revisor está en
**`docs/PLAN_COMMITS.md`**.

### Hitos

- **M1 (fin semana 1):** `pytest` verde y `smoke.yaml` corre en CI.
- **M2 (fin semana 2):** una imagen restaurada generada por el modelo (aunque sea mala).
- **M3 (fin semana 3):** tabla PSNR/SSIM + figuras + demo funcionando.
- **M4 (fin semana 4):** repo congelado, tag `v1.0.0`, informe PDF, defensa ensayada.

---

## 7. Dataset personalizado

### 7.1 Estrategia

- Recolectar **300–500 fotos propias** (campus, paisajes, objetos, familiares con consentimiento)
  o fotos de dominio público.
- Crear el par:
  - **target:** foto limpia.
  - **input:** la misma foto degradada sintéticamente (ruido, blur, JPEG, rayones, sepia,
    baja resolución).
- Generar **2–3 variantes por foto** para aumentar datos.
- **Dividir por foto fuente**, no por parche, para evitar fuga de datos.
- Guardar un **manifiesto CSV/JSON** con la semilla y los parámetros de degradación.

### 7.2 Reglas

- `train` 80 %, `val` 10 %, `test` 10 %, por fuente.
- El test no se toca hasta la evaluación final.
- Nada de personas sin consentimiento en un repo público.
- Documentar licencia y procedencia en `data/README.md`.

Receta detallada y ficha del dataset en `docs/DATASET.md`.

---

## 8. Entrenamiento y evaluación

### 8.1 Contratos de tensores (definirlos antes de programar)

```text
Generador:      (B, 256, 256, 3) -> (B, 256, 256, 3)   valores en [-1, 1]
Discriminador:  [(B,256,256,3), (B,256,256,3)] -> (B, 30, 30, 1) logits
Dataset:        (input, target), float32, [-1, 1], 256x256
```

Los tests verifican estas formas. Así B, C, D y E pueden trabajar en paralelo sin bloquearse.

### 8.2 Hiperparámetros iniciales

| Parámetro | Valor |
|---|---|
| Imagen | 256×256 |
| Batch | 1–4 (según GPU) |
| Optimizador | Adam, lr 2e-4, β1 0.5, β2 0.999 |
| Pérdida | BCEWithLogits + λ·L1, λ = 100 |
| Épocas | 100 (con early stopping por L1 de validación) |
| Checkpoint | Cada época, con reanudación |
| Mixed precision | Sí, en GPU |
| Semilla | 42 |

### 8.3 Evaluación

- **Cuantitativa:** PSNR, SSIM, L1 en el test set. Media ± desviación estándar.
- **Líneas base:** imagen de entrada, filtro clásico (mediana/Gaussiano), U-Net solo L1,
  Pix2Pix completo.
- **Ablación:** λ = 0 (sin GAN) vs λ = 100; con/sin skip connections si alcanza el tiempo.
- **Cualitativa:** grid `input | target | predicción` y galería de fallos.
- **Opcional:** FID y pequeña encuesta humana (1–5 de realismo).

### 8.4 Cómputo

| Configuración | Dónde | Tiempo aproximado |
|---|---|---|
| 64×64, 2 pasos (smoke) | CPU / CI | < 1 min |
| 128×128, 400 pares, 50 épocas | Colab T4 | 2–4 h |
| 256×256, 900 pares, 100 épocas | Colab T4 / Kaggle P100 | 8–12 h |

Guardar checkpoints en Drive cada época. Colab se desconecta; hay que poder reanudar.

---

## 9. Informe

Estructura completa y responsables en `docs/INFORME.md`. Resumen:

1. Portada y datos del curso.
2. Resumen.
3. Introducción: problema, motivación, objetivos.
4. Marco teórico: GAN, minimax, Pix2Pix, U-Net, PatchGAN, pérdidas, métricas.
5. Dataset: recolección, ética, degradaciones, estadísticas, splits.
6. Arquitectura: diagrama, tablas de capas, hiperparámetros, pseudocódigo.
7. Experimentos: setup, baselines, ablación, resultados, figuras.
8. Aplicación/demo: cómo ejecutar, capturas.
9. Discusión: limitaciones, fallos, overfitting, mode collapse.
10. Conclusiones y trabajo futuro.
11. Referencias IEEE.
12. Anexos: estructura del repo, participación, reflexiones individuales.

**Formato:** Markdown en `report/informe.md` + `referencias.bib`, exportado a PDF con Pandoc.
Eso deja evidencia en git y permite construir el PDF en CI. Si el curso exige Word, se exporta
también `.docx`.

**Regla:** el informe se escribe desde la semana 1, no al final. Cada sección tiene dueño y se
integra por PR.

---

## 10. Defensa

Guion de 15–20 minutos y banco de 20 preguntas en `docs/DEFENSA.md`.

Reglas:

- Todos preparan **todas** las secciones, aunque cada uno exponga una.
- Dos ensayos con **presentador aleatorio** (simula al docente).
- Demo de respaldo: video/GIF y galería precalculada, por si falla internet.
- Cada integrante debe poder explicar: formas de tensores, ecuaciones de pérdida, por qué
  PatchGAN, por qué λ=100, cómo se evita fuga de datos, métricas y limitaciones.

---

## 11. Riesgos y mitigación

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Algún integrante no participa | Media | Issues semanales, revisión cruzada, `PARTICIPACION.md`, alerta temprana. |
| La GAN no converge | Media | Baseline L1 primero, label smoothing, TTUR, checkpoints, reducir a 128×128. |
| No hay GPU | Alta | Colab/Kaggle; fallback 128×128 y menos épocas. |
| Dataset muy pequeño | Media | 2–3 degradaciones por foto, parches de 256, augmentación. |
| Fuga de datos | Media | Split por fuente + test bloqueado; test automatizado. |
| Conflictos de merge | Media | PRs pequeños, pull diario, un dueño por archivo. |
| Se acaba el tiempo | Media | Congelar código el día 24; el día 25–28 solo informe/defensa. |
| Repo privado sin branch protection | Baja | Repo público o GitHub Education. |

---

## 12. Checklist de entrega

### Repositorio
- [ ] `README.md` con descripción, equipo, link al informe y a la demo.
- [ ] `LICENSE` y `data/README.md` con procedencia/licencia del dataset.
- [ ] Todos los integrantes aparecen en Insights → Contributors.
- [ ] `docs/PARTICIPACION.md` con tabla y enlaces.
- [ ] `main` protegida y CI verde.
- [ ] Tag `v1.0.0` con release y notas.
- [ ] Pesos y dataset descargables (release/Hugging Face/Drive) o instrucciones claras.
- [ ] No hay datos personales, pesos ni archivos gigantes en git.

### Modelo
- [ ] `configs/base.yaml` reproduce el resultado final.
- [ ] `pytest` verde.
- [ ] `python -m src.gan.train --config configs/smoke.yaml --max-steps 2` funciona en CPU.
- [ ] Checkpoint del generador incluido o descargable.
- [ ] Script de inferencia documentado.
- [ ] Demo (Gradio) funcionando localmente y, si se pudo, desplegada.

### Informe
- [ ] PDF con figuras, tablas, referencias IEEE.
- [ ] Métricas con media ± std y comparación de baselines.
- [ ] Limitaciones y trabajo futuro.
- [ ] Anexo de participación.

### Defensa
- [ ] Diapositivas en el repo (`docs/presentacion.pdf` o enlace).
- [ ] Ensayo 1 y 2 con presentador aleatorio.
- [ ] Video/GIF de respaldo de la demo.
- [ ] Cada integrante resolvió el banco de preguntas.

---

## 13. Comandos útiles

```bash
# Clonar e instalar
git clone <URL>
cd <repo>
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# Rama nueva
git switch main
git pull --rebase
git switch -c feat/07-dataset-pairs

# Antes de commitear
pre-commit run --all-files
pytest -q

# Commit y push
git add -p
git commit -m "feat(data): implementar emparejamiento y split por fuente (#7)"
git push -u origin HEAD

# Actualizar rama con main
git fetch origin
git rebase origin/main

# Ver participación
git shortlog -sne --all
```

---

## 14. Referencias del enunciado

- Pix2Pix (TensorFlow): https://www.tensorflow.org/tutorials/generative/pix2pix
- Pix2Pix (paper): https://phillipi.github.io/pix2pix/
- CycleGAN: https://www.tensorflow.org/tutorials/generative/cyclegan
- StyleGAN2: https://github.com/NVlabs/stylegan2
- GANSynth: https://magenta.tensorflow.org/gansynth
- WaveGAN: https://chrisdonahue.com/wavegan_examples/
