# Notebooks de Colab

El entrenamiento real se hace en Colab (GPU). El código vive en `src/`; los notebooks solo
montan Drive, preparan rutas y llaman a los módulos del repositorio.

| Notebook | Para qué | Estado |
|---|---|---|
| **`pix2pix_colab.ipynb`** | **Notebook de trabajo.** Dataset, entrenamiento base, ablación, experimento de nitidez (3b/5b, opcional), curvas, evaluación del test con tabla y figuras automáticas, demo y descarga. | Listo. Con las opciones por defecto solo evalúa el checkpoint que ya existe. |
| `pix2pix_entrenamiento_colab.ipynb` | **Evidencia.** La corrida base de 20 épocas tal como la ejecutó Isabela Cabezas, con sus salidas. | Ejecutado (522 s; val L1 0.1482, PSNR 20.26 dB, SSIM 0.5077). No se edita. |

Los pesos, el dataset y `outputs/` **no** se suben a git; quedan en el Drive del equipo.

## Uso rápido de `pix2pix_colab.ipynb`

1. *Entorno de ejecución → Cambiar tipo → GPU*.
2. Sección 0: ajustar `ENTRENAR_BASE`, `ENTRENAR_ABLACION`, `EPOCAS_BASE`, `RESUME`.
3. *Ejecutar todas*. Si se desconecta: `RESUME = True` y volver a ejecutar.

Guía completa: [`docs/GUIA_PRUEBAS.md`](../docs/GUIA_PRUEBAS.md).

## Por qué se unificaron los dos notebooks del equipo

| | Base + ablación ("Te damos la bienvenida a Colab") | Métricas ("Pix2Pix_metricas_perdidas_existentes") |
|---|---|---|
| ¿Corrió? | No: `AssertionError` en la celda 1, no encontró `datasetv1.zip` | Sí, completo |
| Qué entrena | Base y ablación sin GAN | Solo base |
| Épocas | 100 (`base.yaml`) | 20 |
| Código | `main` + `requirements.txt` | `main` con `losses.py` de `feat/discriminator-losses` (equivalente); el `git clone` falló y no hubo `git pull` |
| Salidas | `/content` (se pierden al desconectarse; `RESUME` inútil) | Drive |
| Reanudar | `RESUME` | No |
| Gráficas | PNG solo de la base | `plt.show()`, no se guardaron |
| Descarga | Zip con checkpoints (cientos de MB) | Ninguna |
| Orden | Limpio | Celdas repetidas |

**Decisión:**

- El de **métricas** se conserva intacto como `pix2pix_entrenamiento_colab.ipynb`, porque es la
  prueba de la corrida que respalda las cifras del informe.
- El de **base + ablación** y el de evaluación se reemplazan por `pix2pix_colab.ipynb`, que toma:
  - del primero: estructura, instalación desde `requirements.txt`, `RESUME` y la ablación;
  - del segundo: salidas en Drive y dataset desempaquetado en `/content`;
  - además: búsqueda del zip en las rutas del equipo, `git pull` siempre, curvas en PNG para
    base y ablación, evaluación del test con `src/gan/evaluate.py`, tabla automática en README
    e informe, demo y un zip de descarga sin checkpoints.

## Observaciones sobre la corrida registrada

| Observación | Consecuencia |
|---|---|
| 20 épocas en lugar de 100 | El early stopping (paciencia 15) nunca actuó; la L1 de validación aún bajaba. |
| Pérdida de D ≈ 0 desde la época 2 | El generador aprendió casi solo por el término L1 (`docs/RESULTADOS.md` §1.2). |
| Sin `git pull` tras el clon fallido | La versión exacta del código no quedó registrada; `pix2pix_colab.ipynb` imprime el commit. |
| Curvas no guardadas | Se rehicieron con `scripts/plot_training_log.py` → `docs/img/curvas_entrenamiento.png`. |
