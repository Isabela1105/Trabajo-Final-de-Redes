# Notebooks de Colab

El entrenamiento real se hace en Colab (GPU). El código vive en `src/`; los notebooks solo
montan Drive, preparan rutas y llaman a los módulos del repositorio.

| Notebook | Qué hace | Estado |
|---|---|---|
| `pix2pix_entrenamiento_colab.ipynb` | Corrida base de 20 épocas con `configs/base.yaml` y la versión de `losses.py` de `feat/discriminator-losses`. Guarda `metrics.csv`, checkpoints, logs y muestras en Drive. | Ejecutado, con salidas (522 s, val L1 0.1482, PSNR 20.26 dB, SSIM 0.5077). |
| `pix2pix_base_y_ablacion_colab.ipynb` | Corrida base + ablación sin GAN (`use_gan: false`, `λ = 1000`) y curvas. | Corregido: la primera ejecución falló al no encontrar `datasetv1.zip`. Falta correrlo. |
| `evaluacion_y_demo_colab.ipynb` | Evalúa el split `test`, genera la tabla y las figuras finales y abre la demo Gradio. | Listo para correr con el checkpoint de Drive. |

Los pesos, el dataset y `outputs/` **no** se suben a git; quedan en el Drive del equipo.
