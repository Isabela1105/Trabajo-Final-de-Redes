"""Tests de las pérdidas con valores conocidos (sin NaN, escalares, lambda)."""

import math

import numpy as np
import pytest
import tensorflow as tf

from src.gan.losses import (
    adversarial_loss,
    discriminator_loss,
    generator_loss,
    l1_loss,
)

# Con logits = 0, la sigmoide da 0.5 y BCE = -ln(0.5) = ln(2), sea la etiqueta
# 0 o 1. Es el valor de un discriminador "indeciso".
LN2 = math.log(2.0)

FORMA_LOGITS = (2, 30, 30, 1)
FORMA_IMAGEN = (2, 256, 256, 3)


def _logits(valor):
    # Mapa de logits constante con la forma de la salida de D.
    return tf.fill(FORMA_LOGITS, float(valor))


def _imagen(valor):
    # Imagen constante con la forma de las que maneja el generador.
    return tf.fill(FORMA_IMAGEN, float(valor))


@pytest.mark.parametrize("es_real", [True, False])
def test_adversarial_con_logits_cero_vale_ln2(es_real):
    perdida = adversarial_loss(_logits(0), target_is_real=es_real)
    assert float(perdida) == pytest.approx(LN2, rel=1e-5)


def test_discriminator_loss_indeciso_vale_ln2():
    perdida = discriminator_loss(_logits(0), _logits(0))
    assert float(perdida) == pytest.approx(LN2, rel=1e-5)


def test_discriminator_loss_perfecto_es_casi_cero():
    # D acierta: real -> +20 (prob ~1), falsa -> -20 (prob ~0).
    perdida = discriminator_loss(_logits(20), _logits(-20))
    assert float(perdida) < 1e-6


def test_discriminator_loss_enganado_es_grande():
    # D se equivoca por completo: BCE(-20, 1) = BCE(20, 0) = ~20.
    perdida = discriminator_loss(_logits(-20), _logits(20))
    assert float(perdida) == pytest.approx(20.0, rel=1e-3)


def test_l1_valor_conocido():
    # |0.2 - (-0.3)| = 0.5 en todos los píxeles -> media = 0.5.
    assert float(l1_loss(_imagen(0.2), _imagen(-0.3))) == pytest.approx(0.5)


def test_generator_loss_sin_error_l1_solo_queda_la_adversarial():
    # G perfecto en L1 (salida == objetivo) y D indeciso -> solo ln(2).
    perdida = generator_loss(_logits(0), _imagen(0.1), _imagen(0.1))
    assert float(perdida) == pytest.approx(LN2, rel=1e-5)


def test_lambda_multiplica_solo_el_l1():
    # Diferencia de 0.5 por píxel -> L1 = 0.5. Con lambda = 100: ln2 + 50.
    gen, objetivo = _imagen(0.0), _imagen(0.5)
    con_lambda_100 = generator_loss(_logits(0), gen, objetivo, lambda_l1=100.0)
    con_lambda_0 = generator_loss(_logits(0), gen, objetivo, lambda_l1=0.0)
    assert float(con_lambda_100) == pytest.approx(LN2 + 50.0, rel=1e-5)
    # Con lambda = 0 la parte adversarial queda intacta (ln2).
    assert float(con_lambda_0) == pytest.approx(LN2, rel=1e-5)


def test_las_perdidas_devuelven_escalares():
    # Contrato con train.py: tensores de forma () y tipo float32.
    d = discriminator_loss(_logits(0), _logits(0))
    g = generator_loss(_logits(0), _imagen(0.0), _imagen(0.0))
    for perdida in (d, g):
        assert perdida.shape == ()
        assert perdida.dtype == tf.float32


@pytest.mark.parametrize("extremo", [-1e4, 1e4])
def test_perdidas_finitas_con_logits_extremos(extremo):
    d = discriminator_loss(_logits(extremo), _logits(-extremo))
    g = generator_loss(_logits(extremo), _imagen(1.0), _imagen(-1.0))
    assert np.isfinite(float(d))
    assert np.isfinite(float(g))
