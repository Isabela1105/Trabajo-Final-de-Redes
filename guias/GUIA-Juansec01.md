# Guía individual — Juan Sebastian Castillo Acevedo

**GitHub:** `@Juansec01`
**Código:** 2231921
**Rol:** Evaluación, inferencia y demo
**Respaldo:** Cristian Fabián Muñoz (`@CristianMP355`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `CONTRIBUTING.md`, `docs/DEFENSA.md` y `docs/REFERENCIA_CODIGO.md`.

---

## 1. Tu misión

Convertir el modelo entrenado en **evidencia medible y demostrable**:

- Implementar PSNR, SSIM y L1 con tests de valores conocidos.
- Evaluar el test set y guardar `metrics.csv`.
- Generar figuras `input | target | predicción` y galería de fallos.
- Implementar inferencia sobre una imagen nueva.
- Construir la demo Gradio (local y, si alcanza, Hugging Face Spaces).
- Documentar resultados, baselines y limitaciones.

Sin tu módulo el proyecto no tiene números ni demo. Eres el puente entre "el modelo entrenó" y
"aquí está la prueba de que funciona".

---

## 2. Archivos que posees

```text
src/gan/evaluate.py
src/gan/infer.py
app/app.py
tests/test_metrics.py
tests/test_app.py
docs/img/results_grid.png
docs/img/failures.png
docs/img/app.png
docs/ARQUITECTURA.md          # sección evaluación/resultados
docs/DEFENSA.md               # guion de resultados y demo
README.md                     # secciones Resultados y Demo (coordinar con Isabela)
```

No edites `train.py`, el generador, el discriminador ni las pérdidas sin avisar.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `docs/06-protocolo-eval` | PR-E1 | `docs(eval): definir métricas y protocolo de evaluación` | Juan Pablo |
| 1 | 2–4 | `feat/16-metrics-inference` | PR-E2 | `feat(eval): implementar PSNR, SSIM y L1`<br>`feat(infer): implementar inferencia sobre una imagen`<br>`test(eval): validar métricas con imágenes conocidas` | Jeremy, Cristian |
| 1 | 5 | `docs/17-baselines` | PR-E3 | `docs(eval): describir líneas base y tablas de resultados` | Isabela |
| 2 | 8–11 | `feat/26-eval-app` | PR-E4 | `feat(eval): evaluar test set y guardar CSV`<br>`feat(viz): generar grids input/target/predicción`<br>`feat(app): crear demo Gradio mínima`<br>`test(app): probar carga de modelo y predicción` | Juan Pablo, Cristian, Isabela |
| 3 | 15–20 | `feat/33-results-app` | PR-E5 | `feat(eval): agregar FID opcional`<br>`feat(app): agregar ejemplos y selector de imágenes`<br>`docs(eval): documentar resultados cuantitativos y cualitativos`<br>`docs(app): documentar despliegue en Hugging Face Spaces` | Isabella, Jeremy, Isabela |
| 4 | 22–24 | `fix/37-app-formats` | PR-E6 | `fix(app): manejar imágenes grandes y formatos`<br>`docs(eval): congelar tablas y figuras finales` | Juan Pablo |

**Total:** 6 PRs, 15 commits.

---

## 4. Día 1: cómo empezar

```bash
git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
cd Trabajo-Final-de-Redes

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install

git switch main
git pull --rebase
git switch -c docs/06-protocolo-eval
```

Escribe en `docs/ARQUITECTURA.md` el protocolo:

```text
Evaluación sobre split test, imágenes en [0,1] para PSNR/SSIM.
Métricas: PSNR (dB), SSIM, L1 medio. Reportar media ± std por imagen.
Baselines: entrada sin restaurar, filtro de mediana, U-Net solo L1.
```

Commit:

```bash
git add docs/ARQUITECTURA.md
git commit -m "docs(eval): definir métricas y protocolo de evaluación (#6)"
git push -u origin HEAD
```

---

## 5. Semana por semana

### Semana 1 — Métricas e inferencia

**`evaluate.py`:**

```python
def psnr(y_true, y_pred):
    # y en [0,1]; max_val=1.0
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 20.0 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def l1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))
```

Cuidado con los rangos: el modelo trabaja en `[-1,1]`, pero PSNR/SSIM se calculan en `[0,1]`.
Desnormaliza antes de medir. Un error aquí invalida toda la tabla.

**`infer.py`:**

- Cargar `generator.keras`.
- Leer imagen (soportar JPG/PNG/WebP), convertir a RGB.
- Redimensionar a 256×256 **sin deformar** (padding si es necesario).
- Normalizar a `[-1,1]`, predecir, desnormalizar a `[0,1]`, guardar PNG.
- Si la imagen original era más grande, devolver la predicción al tamaño original.

**Tests:**

- PSNR de una imagen consigo misma → `inf` o valor muy alto.
- SSIM de una imagen consigo misma → `1.0`.
- L1 de imágenes iguales → `0.0`.
- Imagen gris → RGB (3 canales).
- `infer` corre con un checkpoint de smoke.

**Criterio de aceptación:**

```bash
pytest tests/test_metrics.py -q
python -m src.gan.infer --checkpoint outputs/smoke/generator.keras --input data/sample/input/000.png --output /tmp/out.png
```

### Semana 2 — Evaluación + demo mínima

- Recorrer el test set y escribir una fila por imagen en `outputs/eval/metrics.csv`:
  `file, psnr, ssim, l1`.
- Resumen: media ± std, y por tipo de degradación si el manifiesto lo permite.
- Figuras:
  - `docs/img/results_grid.png`: 4–8 filas `input | target | predicción`.
  - `docs/img/failures.png`: los 4 peores casos por L1.
- `app/app.py` con Gradio:
  - `gr.Image(type="pil")` de entrada.
  - Botón "Restaurar".
  - Slider opcional de tamaño.
  - Salida `gr.Image`.
  - Ejemplos precargados desde `data/sample/`.
- Test de la app sin levantar servidor: importar la función de predicción y llamarla con una
  imagen pequeña. No testear la UI completa.

**Criterio de aceptación:**

```bash
python -m src.gan.evaluate --config configs/base.yaml --checkpoint outputs/base/checkpoints/generator.keras --split test
python app/app.py
pytest tests/test_app.py -q
```

### Semana 3 — Resultados y app final

- Completar la tabla del README:

  | Modelo | PSNR ↑ | SSIM ↑ | L1 ↓ |
  |---|---|---|---|
  | Entrada (sin restaurar) | | | |
  | Filtro clásico (mediana) | | | |
  | U-Net solo L1 | | | |
  | **Pix2Pix (λ = 100)** | | | |

- Baseline "sin restaurar" = comparar `input` contra `target`. Es obligatorio: sin él no se sabe
  si el modelo aporta algo.
- FID opcional; si no se alcanza, decirlo en el informe.
- Subir la app a Hugging Face Spaces **sin** el checkpoint en git: usar Release, HF Hub o Drive.
- Documentar el despliegue paso a paso.

### Semana 4 — Cierre

- Manejar imágenes grandes (redimensionar antes de pasar por el modelo), formatos con alfa,
  EXIF rotado y escala de grises.
- Congelar tablas y figuras finales; que coincidan con el informe.
- Verificar que `docs/img/` no tenga archivos innecesariamente pesados (> 2 MB por figura).
- Revisar que la demo funcione sin internet.

---

## 6. Cómo medir sin engañarse

| Error común | Consecuencia | Cómo evitarlo |
|---|---|---|
| Calcular PSNR en `[-1,1]` | Números inflados o negativos | Desnormalizar a `[0,1]` |
| Evaluar en train | Métricas irreales | Usar siempre `split=test` |
| Comparar solo contra el target | No hay referencia de mejora | Incluir baseline sin restaurar |
| Elegir las mejores imágenes | Sesgo de selección | Reportar media y peores casos |
| Medir por parche | No representa la imagen | Medir por imagen completa |

---

## 7. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-C4 | Jeremy | Que las pérdidas no se confundan con métricas; que L1 del entrenamiento sea el mismo L1 de evaluación. |
| PR-D4 | Cristian | Que el checkpoint guarde el generador con la misma arquitectura usada en inferencia. |
| PR-A5 | Isabella | Que el conjunto real de evaluación no se mezcle con train. |
| PR-F5 | Isabela | Que el README publique las cifras reales y no placeholders. |

---

## 8. Qué debes poder explicar en la defensa

1. ¿Qué mide PSNR y qué no?
2. ¿Qué mide SSIM y por qué se complementa con PSNR?
3. ¿Por qué L1 y no L2 para la tabla?
4. ¿Cómo se calculan las métricas (rango, tamaño, split)?
5. ¿Por qué incluir el baseline sin restaurar?
6. ¿Cuál es el peor caso y por qué falla?
7. ¿Cómo funciona la demo de punta a punta?
8. ¿Qué limitaciones tiene el modelo (alucinación, color, texturas)?
9. ¿Por qué no se puede confiar solo en las métricas?

---

## 9. Entregables

- [ ] `evaluate.py` con PSNR, SSIM, L1 y CSV.
- [ ] `infer.py` que funciona con una imagen nueva.
- [ ] `app/app.py` con Gradio y ejemplos.
- [ ] `tests/test_metrics.py` y `tests/test_app.py` verdes.
- [ ] `docs/img/results_grid.png`, `failures.png`, `app.png`.
- [ ] Tabla de resultados con baselines.
- [ ] Documentación de despliegue.
- [ ] 15 commits y 6 PRs.
- [ ] 4 revisiones a otros.
- [ ] Sección **Resultados y demo** del informe.

---

## 10. Checklist final

- [ ] `pytest -q` pasa.
- [ ] Las métricas se calculan en `[0,1]` y sobre `split=test`.
- [ ] La tabla del README coincide con `metrics.csv`.
- [ ] La demo corre sin internet.
- [ ] Hay video/GIF de respaldo de la demo.
- [ ] Las figuras están en el informe y en `docs/img/`.
- [ ] Tu nombre aparece en `docs/PARTICIPACION.md`.
- [ ] Puedes explicar la tabla y la demo sin mirar apuntes.
