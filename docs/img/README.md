# Imágenes del proyecto

Esta carpeta contiene las figuras que se usan en el README, el informe y las diapositivas.
Las imágenes **no** se generan solas: se producen al correr la evaluación y la app.

## Archivos esperados

| Archivo | Quién lo genera | Cómo |
|---|---|---|
| `dataset_grid.png` | Isabella Ortiz | `python scripts/build_dataset.py --preview docs/img/dataset_grid.png` o captura de 8–16 pares `input \| target`. |
| `results_grid.png` | Juan Sebastian | `python -m src.gan.evaluate ... --save-grid docs/img/results_grid.png` (4–8 filas `input \| target \| predicción`). |
| `failures.png` | Juan Sebastian | Los 4 peores casos por L1 del test set. |
| `app.png` | Juan Sebastian | Captura de la demo Gradio con una imagen cargada. |
| `generator.png` | Juan Pablo | Diagrama U-Net (draw.io, Excalidraw o similar), exportado a PNG. |
| `training_curves.png` | Cristian | Curvas de `loss_g`, `loss_d`, `l1`, `val_l1` desde `metrics.csv`. |

## Reglas

- PNG o JPG, ancho máximo 1600 px, peso máximo 2 MB por figura.
- Nombres en minúsculas y sin espacios.
- Si una figura cambia, se reemplaza en el mismo PR que cambió el resultado.
- Las figuras del informe deben ser **las mismas** que las del README: no se aceptan números
  distintos en cada documento.
- No subir imágenes con datos personales de terceros sin consentimiento.
