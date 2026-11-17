"""Propósito: generar muestras visuales y escribir escalares/imágenes en TensorBoard.

Autor: Cristian Fabian Muñoz (@CristianMP355)
Entradas: batches de imágenes en [-1, 1], ruta de salida y paso global (época).
Salidas: PNG `input | target | predicción` y eventos de TensorBoard.
Referencia: docs/PLAN_COMMITS.md (PR-D2) y docs/ARQUITECTURA.md §5.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

# backend Agg: indispensable en Colab/CI sin servidor gráfico.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402


def denormalize(images: tf.Tensor) -> tf.Tensor:
    """Pasa imágenes de [-1, 1] a [0, 1] recortando posibles desbordamientos."""
    return tf.clip_by_value((tf.cast(images, tf.float32) + 1.0) / 2.0, 0.0, 1.0)


def _to_uint8(images: tf.Tensor) -> tf.Tensor:
    """Convierte un batch en [-1, 1] a uint8 para TensorBoard."""
    return tf.image.convert_image_dtype(denormalize(images), tf.uint8)


def _atomic_savefig(fig: "plt.Figure", path: Path) -> None:
    """Guarda la figura en un temporal y la renombra: evita PNGs corruptos si Colab se cae."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(tmp, dpi=110, bbox_inches="tight", pad_inches=0.05)
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        if tmp.exists():
            tmp.unlink()


def save_sample_grid(
    input_batch: tf.Tensor,
    target_batch: tf.Tensor,
    pred_batch: tf.Tensor,
    path: str | Path,
    max_samples: int = 4,
) -> Path:
    """Guarda una cuadrícula con columnas `input | target | predicción`.

    Solo se usa desde el bucle de entrenamiento (modo eager); nunca dentro de `tf.function`.
    """
    n = min(max_samples, int(tf.shape(input_batch)[0].numpy()))
    if n <= 0:
        raise ValueError("El batch de muestras está vacío.")

    columns = [denormalize(input_batch)[:n], denormalize(target_batch)[:n], denormalize(pred_batch)[:n]]
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 3 * n), squeeze=False)

    for row in range(n):
        for col, (title, images) in enumerate(zip(("input", "target", "predicción"), columns)):
            ax = axes[row][col]
            ax.imshow(images[row].numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=10)

    fig.tight_layout()
    output = Path(path)
    _atomic_savefig(fig, output)
    return output


class TensorBoardLogger:
    """Envoltorio del writer de TensorBoard: escalares e imágenes por época."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.create_file_writer(str(self.log_dir))

    def log_scalars(self, step: int, **values: float) -> None:
        """Escribe todas las pérdidas/métricas recibidas en un mismo paso."""
        with self._writer.as_default():
            for name, value in values.items():
                tf.summary.scalar(name, float(value), step=int(step))
            self._writer.flush()

    def log_images(
        self,
        step: int,
        input_batch: tf.Tensor,
        target_batch: tf.Tensor,
        pred_batch: tf.Tensor,
        max_outputs: int = 4,
    ) -> None:
        """Registra las tres imágenes por muestra para compararlas en TensorBoard."""
        with self._writer.as_default():
            tf.summary.image("input", _to_uint8(input_batch), step=int(step), max_outputs=max_outputs)
            tf.summary.image("target", _to_uint8(target_batch), step=int(step), max_outputs=max_outputs)
            tf.summary.image(
                "prediction", _to_uint8(pred_batch), step=int(step), max_outputs=max_outputs
            )
            self._writer.flush()

    def close(self) -> None:
        """Cierra el writer para que no queden eventos sin volcar."""
        self._writer.flush()
        self._writer.close()
