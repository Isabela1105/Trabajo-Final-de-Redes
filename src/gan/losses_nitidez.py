"""Pérdidas del experimento de nitidez: L1 + adversarial + perceptual (VGG16) y label smoothing.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

No reemplaza a `src/gan/losses.py` (la entrega v1.0.0 sigue usando ese módulo). Se activa
solo con `configs/nitidez.yaml` (`model.losses_module: src.gan.losses_nitidez`).

Por qué existe:
- La L1 empuja al generador a "promediar" las posibles soluciones, y un promedio se ve borroso.
- La pérdida perceptual compara características de una red ya entrenada (VGG16 en ImageNet)
  en vez de píxeles: castiga bordes y texturas perdidos aunque el color medio sea correcto
  (Johnson et al., 2016).
- `discriminator_loss` acepta `label_smoothing` (la de `losses.py` no), así que el
  `train.label_smoothing` del YAML por fin tiene efecto: D deja de dominar tan rápido.

Contrato con `train.py`: `generator_loss` devuelve `(total, adversarial, l1)`, igual que las
pérdidas de humo, para que `metrics.csv` siga separando las partes. La perceptual va dentro de
`total` (y de `loss_g`), no se reporta aparte.

Variables de entorno (opcionales):
- `PIX2PIX_LAMBDA_PERC`: peso de la perceptual (por defecto 10).
- `PIX2PIX_VGG_WEIGHTS`: `imagenet` (por defecto) o `none` (pesos aleatorios, solo pruebas
  sin internet).
"""

from __future__ import annotations

import os

import tensorflow as tf

from src.gan.losses import adversarial_loss, l1_loss

# Capa de VGG16 usada: block3_conv3 (64×64×256 con entrada 256×256). Capas tempranas miden
# bordes y texturas finas, que es justo lo que falta; las profundas miden "qué objeto es".
CAPA_VGG = "block3_conv3"
LAMBDA_PERCEPTUAL = float(os.environ.get("PIX2PIX_LAMBDA_PERC", "10"))

_extractor: tf.keras.Model | None = None


def _build_extractor() -> tf.keras.Model:
    """VGG16 congelada hasta `CAPA_VGG`, siempre en float32 (aunque haya mixed precision)."""
    weights = os.environ.get("PIX2PIX_VGG_WEIGHTS", "imagenet")
    weights = None if weights.lower() == "none" else weights
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    try:
        vgg = tf.keras.applications.VGG16(
            include_top=False, weights=weights, input_shape=(None, None, 3)
        )
        model = tf.keras.Model(vgg.input, vgg.get_layer(CAPA_VGG).output, name="vgg_perceptual")
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
    model.trainable = False
    return model


def get_extractor() -> tf.keras.Model:
    """Construye la VGG una sola vez, en modo eager aunque se llame dentro de `tf.function`."""
    global _extractor
    if _extractor is None:
        with tf.init_scope():
            _extractor = _build_extractor()
    return _extractor


def _vgg_features(images: tf.Tensor) -> tf.Tensor:
    """Imágenes en [-1, 1] → características VGG16 (preprocesado oficial de Keras)."""
    pixels = (tf.cast(images, tf.float32) + 1.0) * 127.5
    return get_extractor()(tf.keras.applications.vgg16.preprocess_input(pixels), training=False)


def perceptual_loss(target: tf.Tensor, gen_output: tf.Tensor) -> tf.Tensor:
    """Distancia L1 entre características VGG, normalizada por su magnitud.

    Dividir por la magnitud media de las características del objetivo deja la pérdida en un
    rango parecido al de la L1 de píxeles (≈ 0–1), así `LAMBDA_PERCEPTUAL` no depende de la
    escala interna de VGG.
    """
    real = _vgg_features(target)
    fake = _vgg_features(gen_output)
    scale = tf.stop_gradient(tf.reduce_mean(tf.abs(real))) + 1e-6
    return tf.reduce_mean(tf.abs(real - fake)) / scale


def discriminator_loss(real_logits, fake_logits, label_smoothing: float = 0.0):
    """Como `losses.discriminator_loss`, pero con etiquetado suave para lo real.

    Con `label_smoothing = 0.1`, D apunta a 0.9 (no 1.0) en las imágenes reales: no puede
    volverse arbitrariamente seguro, su gradiente hacia G no se apaga y la competencia dura.
    """
    real_labels = tf.ones_like(real_logits) * (1.0 - label_smoothing)
    real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real_logits)
    )
    fake = adversarial_loss(fake_logits, target_is_real=False)
    return 0.5 * (real + fake)


def generator_loss(fake_logits, gen_output, target, lambda_l1=100.0):
    """adversarial + λ_L1 · L1 + λ_perc · perceptual. Devuelve `(total, adversarial, l1)`."""
    adversarial = adversarial_loss(fake_logits, target_is_real=True)
    l1 = l1_loss(target, gen_output)
    total = adversarial + lambda_l1 * l1
    if LAMBDA_PERCEPTUAL > 0:
        total = total + LAMBDA_PERCEPTUAL * perceptual_loss(target, gen_output)
    return total, adversarial, l1
