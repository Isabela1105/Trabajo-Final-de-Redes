"""Tests del discriminador PatchGAN: fallan si alguien rompe el contrato."""

import numpy as np
import pytest
import tensorflow as tf

from src.gan.models.discriminator import build_discriminator


@pytest.fixture(scope="module")
def discriminador():
    # Se construye una sola vez para todo el archivo (ahorra tiempo en CI).
    return build_discriminator(image_size=256, ndf=64)


def _par_aleatorio(batch=2):
    # Dos imágenes aleatorias en [-1, 1], como las que entrega el generador.
    x = tf.random.uniform((batch, 256, 256, 3), -1.0, 1.0, seed=1)
    y = tf.random.uniform((batch, 256, 256, 3), -1.0, 1.0, seed=2)
    return x, y


def test_salida_es_30x30(discriminador):
    # Contrato: (B, 256, 256, 3) + (B, 256, 256, 3) -> (B, 30, 30, 1).
    x, y = _par_aleatorio(batch=2)
    salida = discriminador([x, y], training=False)
    assert salida.shape == (2, 30, 30, 1)


def test_recibe_dos_imagenes_de_3_canales(discriminador):
    # Es condicional: una entrada para la degradada y otra para real/generada.
    formas = [tuple(entrada.shape) for entrada in discriminador.inputs]
    assert formas == [(None, 256, 256, 3), (None, 256, 256, 3)]


def test_la_ultima_capa_no_tiene_activacion(discriminador):
    # Devuelve logits: la sigmoide la pone la pérdida (BCE con logits).
    capa_final = discriminador.get_layer("logits_parches")
    assert capa_final.get_config()["activation"] == "linear"


def test_salida_finita(discriminador):
    # Sin NaN ni infinitos con entradas normales.
    x, y = _par_aleatorio()
    salida = discriminador([x, y], training=False).numpy()
    assert np.isfinite(salida).all()


def test_es_condicional(discriminador):
    # Si cambia la condición x (con la misma y), la salida debe cambiar.
    x, y = _par_aleatorio()
    otra_x = tf.random.uniform((2, 256, 256, 3), -1.0, 1.0, seed=99)
    salida_1 = discriminador([x, y], training=False).numpy()
    salida_2 = discriminador([otra_x, y], training=False).numpy()
    assert not np.allclose(salida_1, salida_2)


def test_numero_de_parametros_razonable(discriminador):
    # PatchGAN de 70x70 con ndf=64 tiene ~2.77 M de parámetros.
    assert 2_500_000 < discriminador.count_params() < 3_000_000
