# Notebooks de Colab

El entrenamiento real se hace en Colab (GPU). El código vive en `src/`; los notebooks solo
montan Drive, preparan rutas y llaman a los módulos del repositorio.

| Notebook | Qué hace | Estado |
|---|---|---|
| `pix2pix_entrenamiento_colab.ipynb` | Corrida base de 20 épocas con `configs/base.yaml`. Guarda `metrics.csv`, checkpoints, logs y muestras en Drive. | **Ejecutado**, con salidas (522 s, val L1 0.1482, PSNR 20.26 dB, SSIM 0.5077). Es la evidencia de la corrida. |
| `pix2pix_base_y_ablacion_colab.ipynb` | Corrida base + ablación sin GAN (`use_gan: false`, `λ = 1000`), curvas y resumen. | **Corregido**; la primera ejecución falló en la celda 1. Es el que se usa para la ablación. |
| `evaluacion_y_demo_colab.ipynb` | Evalúa el split `test`, genera la tabla y las figuras finales y abre la demo Gradio (respaldo; la principal es `python -m app.server`). | Listo para correr con el checkpoint de Drive. |

Los pesos, el dataset y `outputs/` **no** se suben a git; quedan en el Drive del equipo.

## Revisión de los dos notebooks originales

### `pix2pix_entrenamiento_colab.ipynb` (métricas con las pérdidas existentes)

Funcionó de punta a punta. Se deja **sin modificar** porque es la prueba de la corrida.

| Observación | Efecto | Qué se hizo |
|---|---|---|
| `git clone` falló con *destination path already exists* y no se hizo `git pull`. | La corrida usó el código que ya estaba en la sesión de Colab, no necesariamente el último `main`. | El notebook de ablación ahora hace `git pull --ff-only`. |
| Reemplaza `src/gan/losses.py` por la versión de `feat/discriminator-losses`. | Ninguno: solo difiere de `main` en un `if` equivalente. | Documentado en `docs/RESULTADOS.md`. |
| `epochs: 20` en lugar de las 100 de `base.yaml`. | El early stopping (paciencia 15) nunca actuó; la L1 aún bajaba. | Documentado como limitación; la corrida de 100 épocas queda como trabajo futuro. |
| La pérdida de D llega a ≈ 0 desde la época 2. | El generador aprende casi solo por L1. | Analizado en `docs/RESULTADOS.md` §1.2 y en el informe §5.5. |
| Las gráficas usan `plt.show()` y no se guardan. | No quedaron imágenes de las curvas. | `scripts/plot_training_log.py` las rehace desde el registro impreso → `docs/img/curvas_entrenamiento.png`. |
| Celdas repetidas que releen y reescriben el mismo YAML. | Solo ruido. | Sin cambios (evidencia). |

### `pix2pix_base_y_ablacion_colab.ipynb` (antes "Te damos la bienvenida a Colab")

Falló en la primera celda: `AssertionError: No existe /content/drive/MyDrive/Redes-Pix2Pix/datasetv1.zip`.
Por eso **no hay corrida de ablación**. Correcciones aplicadas:

| Problema | Corrección |
|---|---|
| Ruta fija del zip; el equipo lo tenía como `/content/datasetv1(1).zip`. | Busca el zip en las 4 rutas usadas y dice dónde subirlo si no lo encuentra. |
| Repo viejo si ya existía la carpeta. | `git pull --ff-only` después de clonar. |
| Salidas en `/content/resultados_pix2pix`: se pierden si Colab se desconecta y `RESUME = True` no encuentra nada. | Salidas en Drive (`Redes-Pix2Pix/resultados_pix2pix/{base,ablation}`). |
| La descarga final comprimía también los checkpoints (cientos de MB cada uno). | Zip ligero sin checkpoints; los pesos se quedan en Drive. |
| El resumen final fallaba si la ablación aún no se había corrido. | Omite la corrida que falte con un aviso. |
