"""Métricas de evaluación: PSNR, SSIM y L1 por imagen.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Regla de oro: las imágenes del modelo viven en [-1, 1], pero TODAS las métricas se calculan
en [0, 1]. Por eso primero se llama `to_unit_range` y después se mide.

    PSNR = 10 · log10(MAX² / MSE)      con MAX = 1      (dB, más alto es mejor)
    SSIM = luminancia · contraste · estructura          (0–1, más alto es mejor)
    L1   = mean(|target − predicción|)                  (0–1, más bajo es mejor)

Cada función devuelve un arreglo con UN valor por imagen del lote. El promedio y la
desviación estándar se calculan después, en `src/gan/evaluate.py`; nunca se promedian
parches ni se reporta solo la mejor imagen.

Nota sobre el CSV de entrenamiento: `train.py` mide en [-1, 1]. El PSNR sale idéntico (el
MSE y MAX² se escalan ambos por 4), pero el L1 sale el doble que aquí y el SSIM cambia un
poco. Por eso la tabla final se genera con este módulo y no con `metrics.csv`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

# Con MSE = 0 (imagen idéntica) el PSNR sería infinito; se acota a 100 dB para que el
# promedio siga siendo finito y la tabla no muestre `inf`.
MSE_MINIMO = 1e-10
PSNR_MAXIMO = 10.0 * np.log10(1.0 / MSE_MINIMO)


def to_unit_range(images: tf.Tensor | np.ndarray) -> tf.Tensor:
    """Pasa imágenes de [-1, 1] a [0, 1] y recorta desbordamientos."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    return tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)


def _as_batch(images: tf.Tensor | np.ndarray) -> tf.Tensor:
    """Asegura forma (B, H, W, C) y float32 en [0, 1]."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    if images.shape.rank == 3:
        images = images[tf.newaxis]
    if images.shape.rank != 4:
        raise ValueError(f"Se esperaba (B, H, W, C) o (H, W, C), no {images.shape}")
    return images


def _check_pair(pred: tf.Tensor, target: tf.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Predicción {pred.shape} y objetivo {target.shape} no coinciden")


def psnr(pred: tf.Tensor | np.ndarray, target: tf.Tensor | np.ndarray) -> np.ndarray:
    """PSNR por imagen en dB. Entradas en [0, 1]."""
    pred, target = _as_batch(pred), _as_batch(target)
    _check_pair(pred, target)
    mse = tf.reduce_mean(tf.square(pred - target), axis=[1, 2, 3])
    mse = tf.maximum(mse, MSE_MINIMO)
    return (10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0)).numpy()


def ssim(pred: tf.Tensor | np.ndarray, target: tf.Tensor | np.ndarray) -> np.ndarray:
    """SSIM por imagen (ventana gaussiana 11×11 de TensorFlow). Entradas en [0, 1]."""
    pred, target = _as_batch(pred), _as_batch(target)
    _check_pair(pred, target)
    return tf.image.ssim(pred, target, max_val=1.0).numpy()


def l1(pred: tf.Tensor | np.ndarray, target: tf.Tensor | np.ndarray) -> np.ndarray:
    """Error absoluto medio por imagen. Entradas en [0, 1]."""
    pred, target = _as_batch(pred), _as_batch(target)
    _check_pair(pred, target)
    return tf.reduce_mean(tf.abs(pred - target), axis=[1, 2, 3]).numpy()


def all_metrics(
    pred: tf.Tensor | np.ndarray, target: tf.Tensor | np.ndarray
) -> dict[str, np.ndarray]:
    """Calcula las tres métricas por imagen. Entradas en [0, 1]."""
    return {"psnr": psnr(pred, target), "ssim": ssim(pred, target), "l1": l1(pred, target)}
