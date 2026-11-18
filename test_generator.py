"""Tests del generador U-Net: fallan si alguien rompe el contrato."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

from src.gan.models.generator import build_generator, niveles_por_defecto


def _lote(lado, batch=2):
    # Imágenes aleatorias en [-1, 1], como las que entrega el dataset.
    return tf.random.uniform((batch, lado, lado, 3), -1.0, 1.0, seed=1)


@pytest.mark.parametrize("lado", [64, 128, 256])
def test_forma_y_rango_en_64_128_256(lado):
    # Contrato: (B, H, W, 3) -> (B, H, W, 3) en [-1, 1]. ngf=8 para que corra rápido.
    g = build_generator(image_size=lado, ngf=8)
    salida = g(_lote(lado), training=False).numpy()
    assert salida.shape == (2, lado, lado, 3)
    assert np.isfinite(salida).all()
    assert salida.min() >= -1.0 and salida.max() <= 1.0  # tanh


def test_batch_variable():
    g = build_generator(image_size=64, ngf=8)
    for batch in (1, 3, 5):
        assert g(_lote(64, batch), training=False).shape[0] == batch


@pytest.mark.parametrize("norm", ["instance", "batch", None])
def test_todos_los_parametros_reciben_gradiente(norm):
    # No basta con que el gradiente exista (no sea None): debe ser distinto de cero.
    # Con InstanceNorm en el cuello de botella de 1x1 el gradiente sería exactamente 0
    # y este test lo detectaría.
    g = build_generator(image_size=64, ngf=8, norm=norm)
    x = _lote(64)
    with tf.GradientTape() as cinta:
        perdida = tf.reduce_mean(tf.abs(g(x, training=True) - x))
    grads = cinta.gradient(perdida, g.trainable_variables)
    sin_gradiente = [
        v.name for v, gr in zip(g.trainable_variables, grads) if gr is None
    ]
    en_cero = [
        v.name
        for v, gr in zip(g.trainable_variables, grads)
        if gr is not None and float(tf.reduce_max(tf.abs(gr))) == 0.0
    ]
    assert not sin_gradiente, sin_gradiente
    assert not en_cero, en_cero


@pytest.mark.parametrize("lado", [64, 128, 256])
def test_skip_connections_una_por_nivel(lado):
    # Hay (niveles - 1) concatenaciones: una por cada salida del encoder salvo el
    # cuello de botella. Keras falla al construir si los tamaños no coinciden.
    g = build_generator(image_size=lado, ngf=8)
    concatenaciones = [c for c in g.layers if isinstance(c, layers.Concatenate)]
    assert len(concatenaciones) == niveles_por_defecto(lado) - 1


def test_dropout_solo_en_los_3_primeros_bloques_del_decoder():
    g = build_generator(image_size=64, ngf=8, dropout=0.5)
    assert len([c for c in g.layers if isinstance(c, layers.Dropout)]) == 3
    sin_dropout = build_generator(image_size=64, ngf=8, dropout=0.0)
    assert not [c for c in sin_dropout.layers if isinstance(c, layers.Dropout)]


def test_dropout_solo_actua_al_entrenar():
    g = build_generator(image_size=64, ngf=8, dropout=0.5)
    x = _lote(64)
    # En inferencia el resultado es determinista...
    assert np.allclose(g(x, training=False).numpy(), g(x, training=False).numpy())
    # ... y al entrenar el dropout cambia la salida entre llamadas.
    assert not np.allclose(g(x, training=True).numpy(), g(x, training=True).numpy())


def test_parametros_ngf64_8_niveles():
    # U-Net de Pix2Pix con ngf=64 y 8 niveles: unos 54 M de parámetros.
    # Solo se construye el modelo, no se ejecuta.
    parametros = build_generator(image_size=256, ngf=64).count_params()
    assert 50_000_000 < parametros < 58_000_000


def test_niveles_por_defecto():
    assert [niveles_por_defecto(n) for n in (64, 128, 256)] == [6, 7, 8]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 100},  # no es divisible por 2**niveles
        {"image_size": 64, "niveles": 7},  # 64 / 2**7 no es entero
        {"image_size": 64, "niveles": 0},
        {"image_size": 64, "norm": "layer"},
    ],
)
def test_configuraciones_invalidas(kwargs):
    with pytest.raises(ValueError):
        build_generator(ngf=8, **kwargs)
