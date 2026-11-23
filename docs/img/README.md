# Imágenes del proyecto

Esta carpeta contiene las figuras que se usan en el README, el informe y las diapositivas.
Las imágenes **no** se generan solas: se producen al correr la evaluación y la app.

## Archivos

| Archivo | Estado | Quién lo genera | Cómo |
|---|---|---|---|
| `dataset_grid.png` | ✅ | Isabella Ortiz | `scripts/dataset_report.py` (pares `input \| target`). |
| `curvas_entrenamiento.png` | ✅ | Juan Sebastian (con el registro de la corrida de Colab) | `python scripts/plot_training_log.py docs/resultados/colab_base_20ep_log.csv docs/img/curvas_entrenamiento.png` |
| `results_grid.png` | pendiente de correr | Juan Sebastian | `python -m src.gan.evaluate --checkpoint outputs/base/checkpoints` → `outputs/base/eval/results_grid.png` (columnas `entrada \| mediana \| Pix2Pix \| objetivo`). |
| `failures.png` | pendiente de correr | Juan Sebastian | Mismo comando: las 4 imágenes de test con menor PSNR. |
| `app.png` | pendiente de correr | Juan Sebastian | Captura de `python -m app.server --checkpoint …` con una imagen restaurada. |

Las dos figuras de evaluación se copian aquí desde `outputs/base/eval/` con
`notebooks/pix2pix_colab.ipynb` (sección 5); no se dibujan a mano.

## Reglas

- PNG o JPG, ancho máximo 1600 px, peso máximo 2 MB por figura (`dataset_grid.png` se dejó
  como lo subió su autora).
- Nombres en minúsculas y sin espacios.
- Si una figura cambia, se reemplaza en el mismo PR que cambió el resultado.
- Las figuras del informe deben ser **las mismas** que las del README: no se aceptan números
  distintos en cada documento.
- No subir imágenes con datos personales de terceros sin consentimiento.
