# Restauración de fotografías con Pix2Pix

[![CI](https://github.com/Isabela1105/Trabajo-Final-de-Redes/actions/workflows/ci.yml/badge.svg)](https://github.com/Isabela1105/Trabajo-Final-de-Redes/actions/workflows/ci.yml)

**Curso:** Redes Neuronales y Deep Learning  
**Universidad:** Universidad Autónoma de Occidente  
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes  
**Entrega:** repositorio + informe + sustentación

---

## 1. Contexto

Este proyecto entrena una **GAN condicional (Pix2Pix)** para **restaurar fotografías degradadas**:
ruido, desenfoque, compresión JPEG, rayones, polvo, baja resolución y desvanecido de color.

El dataset es **personalizado**: recolectamos fotos propias o de dominio público y generamos los
pares `(input, target)` con un pipeline de degradaciones reproducible. La foto limpia es el
*ground truth*; la degradada es la entrada del modelo.

**Lo que se entrega:**

- Código comentado y reproducible en TensorFlow/Keras.
- Dataset propio documentado (ficha, manifiesto, splits, ética).
- Modelo entrenado y checkpoint descargable.
- Evaluación con PSNR, SSIM y L1 + comparación con baselines.
- Demo Gradio para restaurar una imagen.
- Informe en PDF y material de sustentación.
- Evidencia de participación de los 6 integrantes (commits, PRs, revisiones).

**Lo que NO se permite:** usar el dataset de caballos/cebras, manzanas/naranjas ni entrenar con
datos descargados sin construir el dataset propio.

---

## 2. Equipo

| Integrante | GitHub | Código | Rol |
|---|---|---|---|
| Isabella Ortiz Hernandez | [@Isabellaortizher-collab](https://github.com/Isabellaortizher-collab) | 2226215 | Datos y dataset |
| Juan Pablo Dávila Moncayo | [@Juanpablodavila](https://github.com/Juanpablodavila) | 2216168 | Generador U-Net |
| Jeremy Santiago Caicedo Cabezas | [@JeremyCaicedo](https://github.com/JeremyCaicedo) | 2210721 | Discriminador y pérdidas |
| Cristian Fabian Muñoz Piedrahita | [@CristianMP355](https://github.com/CristianMP355) | 2230865 | Entrenamiento |
| Juan Sebastian Castillo Acevedo | [@Juansec01](https://github.com/Juansec01) | 2231921 | Evaluación y demo |
| Isabela | [@Isabela1105](https://github.com/Isabela1105) | _(completar)_ | CI/CD, integración e informe |

> **Nota:** el nombre y código de la última fila se completan cuando el equipo confirme el dato.
> El rol se puede intercambiar antes de crear las ramas; el plan de commits está diseñado para
> que todos programen y todos puedan explicar cualquier módulo.

---

## 3. Arquitectura en una página

```text
Foto degradada (256×256×3, [-1,1])
        │
        ▼
┌─────────────────────────────┐
│ Generador U-Net             │
│ 8 bloques encoder           │
│ 8 bloques decoder + skip    │
│ salida tanh                 │
└─────────────────────────────┘
        │
        ▼
Foto restaurada (256×256×3, [-1,1])

Discriminador PatchGAN:
  concat(input, target) → Conv 64 → 128 → 256 → 512 → 1
  salida: 30×30 logits (cada celda evalúa un parche de 70×70)

Pérdidas:
  L_D = 0.5 · [BCE(D(x,y), 1) + BCE(D(x,G(x)), 0)]
  L_G = BCE(D(x,G(x)), 1) + 100 · L1(G(x), y)
```

| Componente | Contrato |
|---|---|
| Dataset | `(input, target)` float32 `(256,256,3)` en `[-1,1]` |
| Generador | `(B,256,256,3)` → `(B,256,256,3)` en `[-1,1]` |
| Discriminador | `[(B,256,256,3), (B,256,256,3)]` → `(B,30,30,1)` logits |
| Optimizador | Adam, lr 2e-4, β1 0.5 |
| Batch | 1–4 según GPU |
| Épocas | 100 con early stopping por L1 de validación |

Detalle completo: [`docs/MODELOS.md`](docs/MODELOS.md) y
[`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## 4. Resultados

> Completar al final de la semana 3. No inventar cifras.

| Modelo | PSNR ↑ | SSIM ↑ | L1 ↓ |
|---|---|---|---|
| Entrada (sin restaurar) | — | — | — |
| Filtro clásico (mediana/Gaussiano) | — | — | — |
| U-Net solo L1 | — | — | — |
| **Pix2Pix (λ = 100)** | — | — | — |

Figuras: [`docs/img/results_grid.png`](docs/img/) · Fallos:
[`docs/img/failures.png`](docs/img/)

---

## 5. Demo

```bash
python app/app.py
```

O en Hugging Face Spaces: `<enlace pendiente>`

![Demo](docs/img/app.png)

---

## 6. Inicio rápido

### 6.1 Clonar e instalar

```bash
git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
cd Trabajo-Final-de-Redes
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```

### 6.2 Smoke test (CPU, segundos)

```bash
python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
pytest -q
```

Si esto funciona, el pipeline está sano. El entrenamiento real va en Colab/Kaggle.

---

## 7. Dataset

El dataset no se versiona en git. Para reconstruirlo:

```bash
# 1. Colocar fotos originales en data/raw/
# 2. Generar pares y splits por fuente
python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42

# 3. Validar estructura y fuga de datos
python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv
```

- Splits: **80 % train / 10 % val / 10 % test**, por foto fuente (no por parche).
- 2–3 variantes de degradación por foto.
- Manifiesto con semilla y parámetros.
- Ficha completa: [`docs/DATASET.md`](docs/DATASET.md) y [`data/README.md`](data/README.md).

---

## 8. Entrenamiento

```bash
# Entrenamiento principal
python -m src.gan.train --config configs/base.yaml

# Reanudar tras una desconexión de Colab
python -m src.gan.train --config configs/base.yaml --resume outputs/base/checkpoints
```

Salidas:

```text
outputs/base/
├── checkpoints/     # G, D y optimizadores
├── samples/         # grids cada N épocas
├── logs/            # TensorBoard
└── metrics.csv      # pérdidas y métricas
```

Hardware recomendado: Colab T4 o Kaggle P100. En Windows, TensorFlow ≥ 2.11 no tiene GPU nativa;
usar WSL2, Colab o Kaggle.

---

## 9. Evaluación e inferencia

```bash
# Evaluar el test set
python -m src.gan.evaluate \
  --config configs/base.yaml \
  --checkpoint outputs/base/checkpoints/generator.keras \
  --split test

# Restaurar una imagen
python -m src.gan.infer \
  --checkpoint outputs/base/checkpoints/generator.keras \
  --input ruta/a/foto.jpg \
  --output ruta/a/restaurada.png
```

Métricas: PSNR, SSIM, L1. Opcional: FID y encuesta humana.

---

## 10. Estructura del repositorio

```text
.
├── .github/
│   ├── workflows/ci.yml          # lint + tests + smoke training
│   ├── workflows/release.yml     # informe + release en tags v*
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── ISSUE_TEMPLATE/tarea.md
│   └── CODEOWNERS
├── app/app.py                    # demo Gradio
├── configs/
│   ├── base.yaml                 # entrenamiento real
│   └── smoke.yaml                # CI / CPU
├── data/                         # no versionado (excepto README y sample)
├── docs/
│   ├── ARQUITECTURA.md
│   ├── CI_CD.md
│   ├── DATASET.md
│   ├── DECISIONES.md
│   ├── DEFENSA.md
│   ├── EQUIPO.md
│   ├── EXPLICACION_CODIGO.md
│   ├── INFORME.md
│   ├── PARTICIPACION.md
│   └── img/
├── notebooks/
├── report/
│   ├── informe.md
│   └── referencias.bib
├── scripts/
│   ├── build_dataset.py
│   ├── check_dataset.py
│   └── make_synthetic_sample.py
├── src/gan/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── utils/
├── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## 11. Flujo de trabajo con 6 personas

### 11.1 Reglas de oro

1. **Nadie hace push directo a `main`.** Todo entra por Pull Request.
2. **Un issue = una rama = un PR.** PRs de máximo ~400 líneas.
3. **El autor no mergea su propio PR.** Mínimo 1 aprobación; 2 en rutas críticas.
4. **CI verde obligatorio.**
5. **Primer commit en los primeros 2 días.**
6. **Cada semana:** 1 PR propio + 1 revisión + 1 aporte al informe/docs.
7. **Todo el código vive en el repo.** Nada de código solo en WhatsApp/Drive.
8. **Pair programming se acredita** con `Co-authored-by:`.
9. **No se reescribe la historia de `main`.**
10. **Después del día 24 no se agregan features.**

### 11.2 Ramas por integrante

| Integrante | Ramas principales |
|---|---|
| Isabella Ortiz (@Isabellaortizher-collab) | `docs/03-protocolo-dataset`, `feat/07-dataset-pairs`, `feat/09-build-dataset`, `feat/14-dataset-v2`, `feat/21-real-eval`, `docs/28-dataset-final` |
| Juan Pablo (@Juanpablodavila) | `docs/04-contrato-modelos`, `feat/10-generator-unet`, `docs/11-generator-layers`, `refactor/17-generator-norm`, `feat/24-generator-tuning`, `fix/30-generator-borders` |
| Jeremy (@JeremyCaicedo) | `docs/05-contrato-discriminador`, `feat/12-patchgan-losses`, `docs/13-loss-theory`, `feat/20-loss-options`, `feat/27-gradient-penalty`, `fix/32-loss-nan` |
| Cristian (@CristianMP355) | `chore/06-config`, `feat/14-train-loop`, `test/15-smoke-train`, `feat/23-train-integration`, `feat/29-train-experiments`, `fix/34-colab-checkpoint` |
| Juan Sebastian (@Juansec01) | `docs/06-protocolo-eval`, `feat/16-metrics-inference`, `docs/17-baselines`, `feat/26-eval-app`, `feat/33-results-app`, `fix/37-app-formats` |
| Isabela (@Isabela1105) | `chore/01-init`, `chore/02-tooling`, `test/03-synthetic-data`, `ci/29-quality-gates`, `ci/37-report-release`, `docs/40-participacion` |

### 11.3 Commits

Formato **Conventional Commits**:

```text
tipo(area): descripción en imperativo (#issue)
```

Ejemplos:

```text
feat(data): implementar emparejamiento y split por fuente (#7)
feat(model): implementar generador U-Net configurable (#10)
feat(loss): implementar pérdidas adversarial y L1 (#12)
feat(train): integrar dataset y modelo en el bucle (#23)
feat(eval): implementar PSNR, SSIM y L1 (#16)
ci: agregar workflow de lint, tests y smoke training (#2)
```

### 11.4 Pull Requests

- Título con el mismo formato del commit.
- Descripción con la plantilla de `.github/PULL_REQUEST_TEMPLATE.md`.
- `Closes #<issue>`.
- Revisor asignado distinto del autor.
- Captura/métrica si cambia la salida visual.
- Squash merge; borrar la rama.

### 11.5 Orden de merge

| Semana | PRs | Objetivo |
|---|---|---|
| 1 | F1, A1, B1, C1, D1, E1, F2, F3, A2, B2, C2, D2, E2, A3, B3, C3, D3, E3 | Fundaciones: repo, CI, contratos, dataset v1, modelos, train skeleton, métricas |
| 2 | A4, B4, C4, D4, E4, F4 | Integración: primer entrenamiento, evaluación, demo mínima, quality gates |
| 3 | A5, B5, C5, D5, E5, F5 | Experimentos: entrenamiento completo, ablación, resultados, app, borrador del informe |
| 4 | A6, B6, C6, D6, E6, F6 | Cierre: correcciones, informe final, release `v1.0.0`, ensayo de defensa |

Detalle exacto de commits, PRs y revisores:
[`docs/PLAN_COMMITS.md`](docs/PLAN_COMMITS.md).

---

## 12. CI/CD

Cada PR ejecuta:

1. `ruff check .`
2. `black --check .`
3. `pytest -q --cov=src`
4. Generación de 8 imágenes sintéticas de 64×64.
5. Smoke training de 2 pasos en CPU.
6. Validación del título del PR.

`main` está protegida: PR obligatorio, 1 aprobación, checks verdes, historial lineal.

Al crear un tag `v*`, el workflow de release construye el informe en DOCX/PDF y publica un
GitHub Release.

Configuración y solución de problemas: [`docs/CI_CD.md`](docs/CI_CD.md).

---

## 13. Documentación

| Documento | Contenido |
|---|---|
| [`GUIA.md`](GUIA.md) | Guía maestra: requisitos, decisiones, riesgos, checklist. |
| [`docs/MODELOS.md`](docs/MODELOS.md) | Comparación de arquitecturas y justificación de Pix2Pix. |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Diagramas, tablas de capas, contratos, pérdidas. |
| [`docs/DATASET.md`](docs/DATASET.md) | Recolección, degradaciones, splits, ética. |
| [`docs/PLAN_COMMITS.md`](docs/PLAN_COMMITS.md) | Plan de commits por rol. |
| [`docs/CI_CD.md`](docs/CI_CD.md) | GitHub Actions, branch protection, release. |
| [`docs/INFORME.md`](docs/INFORME.md) | Estructura del informe y responsables. |
| [`docs/DEFENSA.md`](docs/DEFENSA.md) | Guion y banco de preguntas. |
| [`docs/EXPLICACION_CODIGO.md`](docs/EXPLICACION_CODIGO.md) | Recorrido archivo → función → para qué sirve. |
| [`docs/PARTICIPACION.md`](docs/PARTICIPACION.md) | Evidencia de participación. |
| [`docs/REFERENCIA_CODIGO.md`](docs/REFERENCIA_CODIGO.md) | Código comentado de referencia. |
| [`docs/MERGE_Y_REVISION.md`](docs/MERGE_Y_REVISION.md) | Cómo integrar y revisar todo al final. |

### Guías individuales

| Integrante | Rol | Guía |
|---|---|---|
| Isabella Ortiz | Datos y dataset | [`guias/GUIA-Isabellaortizher-collab.md`](guias/GUIA-Isabellaortizher-collab.md) |
| Juan Pablo Dávila | Generador U-Net | [`guias/GUIA-Juanpablodavila.md`](guias/GUIA-Juanpablodavila.md) |
| Jeremy Caicedo | Discriminador y pérdidas | [`guias/GUIA-JeremyCaicedo.md`](guias/GUIA-JeremyCaicedo.md) |
| Cristian Muñoz | Entrenamiento | [`guias/GUIA-CristianMP355.md`](guias/GUIA-CristianMP355.md) |
| Juan Sebastian Castillo | Evaluación y demo | [`guias/GUIA-Juansec01.md`](guias/GUIA-Juansec01.md) |
| Isabela | CI/CD, integración e informe | [`guias/GUIA-Isabela1105.md`](guias/GUIA-Isabela1105.md) |

Índice y orden de lectura: [`guias/README.md`](guias/README.md).

---

## 14. Informe y sustentación

- Informe: [`report/informe.md`](report/informe.md) → PDF en `report/informe.pdf`.
- Estructura y responsables: [`docs/INFORME.md`](docs/INFORME.md).
- Guion de defensa y 24 preguntas: [`docs/DEFENSA.md`](docs/DEFENSA.md).

**Regla:** el docente elige al azar a un integrante. Todos deben poder explicar la arquitectura,
las pérdidas, el dataset, el entrenamiento, las métricas y una línea de código cualquiera.

---

## 15. Participación

| Integrante | Commits objetivo | PRs | Revisiones | Informe | Defensa |
|---|---|---|---|---|---|
| Isabella Ortiz | 14 | 6 | 4 | Dataset | Datos |
| Juan Pablo Dávila | 14 | 6 | 5 | Arquitectura U-Net | Generador |
| Jeremy Caicedo | 14 | 6 | 5 | Marco teórico/pérdidas | Pérdidas |
| Cristian Muñoz | 15 | 6 | 5 | Entrenamiento/experimentos | Train loop |
| Juan Sebastian Castillo | 15 | 6 | 4 | Resultados/demo | Resultados |
| Isabela | 15 | 6 | 5 | Intro/conclusiones/anexos | CI/cierre |

Verificación:

```bash
git shortlog -sne --all
```

En GitHub: **Insights → Contributors**, **Pull Requests → Author/Reviewer**, **Issues → Assignee**.

Tabla detallada: [`docs/PARTICIPACION.md`](docs/PARTICIPACION.md).

---

## 16. Checklist de entrega

- [ ] Repo público con los 6 integrantes como collaborators.
- [ ] `main` protegida y CI verde.
- [ ] Dataset construido y documentado.
- [ ] Generador, discriminador, pérdidas y train loop implementados.
- [ ] Entrenamiento real con checkpoint descargable.
- [ ] Métricas PSNR/SSIM/L1 + baselines.
- [ ] Demo Gradio funcionando.
- [ ] Informe PDF con figuras, tablas y referencias IEEE.
- [ ] `docs/PARTICIPACION.md` con enlaces a PRs/commits.
- [ ] Tag `v1.0.0` con release.
- [ ] Dos ensayos de defensa con presentador aleatorio.

---

## 17. Licencia

Código: MIT (ver [`LICENSE`](LICENSE)).
Dataset: ver [`data/README.md`](data/README.md). Las fotos de terceros conservan su licencia.
