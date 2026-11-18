"""Pérdidas de Pix2Pix (TensorFlow/Keras).

Ecuaciones (ver docs/ARQUITECTURA.md, sección 4):

    L_D = 0.5 * [ BCE(D(x, y), 1) + BCE(D(x, G(x)), 0) ]
    L_G = BCE(D(x, G(x)), 1) + lambda * L1(G(x), y)        con lambda = 100

donde x es la foto degradada, y el objetivo real y G(x) la salida del
generador. D devuelve logits (sin sigmoide) con forma (B, 30, 30, 1).

Contrato con train.py: discriminator_loss y generator_loss devuelven un
ESCALAR (tensor de forma ()). No cambiar el orden de los argumentos ni el
tipo de retorno sin actualizar los tests y avisar en el PR.
"""

import tensorflow as tf

# BCE "con logits": aplica la sigmoide y la entropía cruzada en un solo paso.
# Es más estable que hacer sigmoide + BCE por separado (evita log(0) y los
# gradientes pobres). Promedia sobre el batch y sobre los 30 x 30 parches.
_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def adversarial_loss(logits, target_is_real):
    """BCE entre los logits de D y la etiqueta que queremos (1 real, 0 falsa).

    Args:
        logits: salida de D, forma (B, 30, 30, 1).
        target_is_real: True si los parches deberían clasificarse como reales.
    """
    # Una etiqueta por parche: un mapa de unos o de ceros con la forma de los
    # logits (tf.ones_like / tf.zeros_like copian la forma y el tipo).
    if target_is_real:
        etiquetas = tf.ones_like(logits)
    else:
        etiquetas = tf.zeros_like(logits)
    return _bce(etiquetas, logits)


def l1_loss(target, gen_output):
    """Error absoluto medio entre el objetivo real y lo generado.

    Es la parte que obliga al generador a ser FIEL a la imagen original.
    Se usa L1 y no L2 porque L2 tiende a promediar y deja imágenes borrosas.
    """
    return tf.reduce_mean(tf.abs(target - gen_output))


def discriminator_loss(real_logits, fake_logits):
    """Pérdida de D: debe dar 1 a lo real y 0 a lo generado.

    Args:
        real_logits: D(x, y), lo que D opina de la imagen real.
        fake_logits: D(x, G(x)), lo que D opina de la imagen generada.

    Returns:
        Escalar. El factor 0.5 hace que D aprenda más lento que G, como en
        el artículo original de Pix2Pix.
    """
    perdida_real = adversarial_loss(real_logits, target_is_real=True)
    perdida_falsa = adversarial_loss(fake_logits, target_is_real=False)
    return 0.5 * (perdida_real + perdida_falsa)


def generator_loss(fake_logits, gen_output, target, lambda_l1=100.0):
    """Pérdida total de G: engañar a D + ser fiel al objetivo (L1).

    Args:
        fake_logits: D(x, G(x)), lo que D opina de la imagen generada.
        gen_output: G(x), la imagen restaurada, (B, 256, 256, 3).
        target: y, la imagen original sin degradar, (B, 256, 256, 3).
        lambda_l1: peso del L1 (clave lambda_l1 de base.yaml). Multiplica
            SOLO al L1, nunca a la pérdida adversarial.

    Returns:
        Escalar: adversarial + lambda_l1 * L1.
    """
    # Para G queremos que D diga "real" (1) sobre lo generado: por eso la
    # etiqueta es 1 aunque la imagen sea falsa ("non-saturating loss").
    adversarial = adversarial_loss(fake_logits, target_is_real=True)
    l1 = l1_loss(target, gen_output)
    return adversarial + lambda_l1 * l1
