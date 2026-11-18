"""Discriminador PatchGAN condicional de Pix2Pix (TensorFlow/Keras).

Contrato de este módulo (ver docs/ARQUITECTURA.md, secciones 3 y 4):

    entradas : x -> foto degradada               (B, 256, 256, 3), rango [-1, 1]
               y -> objetivo real o G(x)         (B, 256, 256, 3), rango [-1, 1]
    salida   : mapa de logits, uno por parche    (B, 30, 30, 1), SIN sigmoide

El discriminador no da una sola decisión por imagen: da una cuadrícula de
30 x 30 decisiones. Cada una mira un parche de 70 x 70 píxeles de la entrada
(de ahí el nombre PatchGAN).
"""

import tensorflow as tf
from tensorflow.keras import Input, Model, layers


def _bloque_conv(x, filtros, stride, inicializador, usar_norm=True):
    """Bloque básico: Conv 4x4 -> (BatchNorm) -> LeakyReLU(0.2).

    Con padding="same" y stride=2 la imagen se reduce a la mitad
    (256 -> 128 -> 64 -> 32).
    """
    x = layers.Conv2D(
        filtros,
        kernel_size=4,
        strides=stride,
        padding="same",
        kernel_initializer=inicializador,
        # Si después viene BatchNorm el sesgo (bias) sobra, porque BatchNorm
        # ya tiene su propio desplazamiento aprendible.
        use_bias=not usar_norm,
    )(x)

    if usar_norm:
        # Normaliza cada canal para que el entrenamiento sea más estable.
        x = layers.BatchNormalization()(x)

    # LeakyReLU deja pasar un poco de señal negativa (pendiente 0.2). Así el
    # discriminador no "apaga" neuronas y siempre hay gradiente hacia G.
    return layers.LeakyReLU(0.2)(x)


def build_discriminator(image_size: int = 256, ndf: int = 64) -> Model:
    """Construye el discriminador PatchGAN condicional.

    Args:
        image_size: lado de la imagen cuadrada (clave image_size de base.yaml).
        ndf: número de filtros de la primera capa (clave ndf de base.yaml).
            Las capas siguientes lo multiplican por 2, 4 y 8.

    Returns:
        Un modelo Keras con dos entradas [x, y] y una salida de logits con
        forma (B, 30, 30, 1) cuando image_size = 256.
    """
    # Pesos iniciales ~ Normal(0, 0.02), como en el paper de Pix2Pix.
    inicializador = tf.random_normal_initializer(0.0, 0.02)

    # 1) Dos entradas: la foto degradada (la condición) y la imagen a juzgar.
    entrada = Input(shape=(image_size, image_size, 3), name="degradada")
    objetivo = Input(shape=(image_size, image_size, 3), name="real_o_generada")

    # 2) Las unimos por canales: 3 + 3 = 6 canales. Así el discriminador ve
    #    la condición y puede decidir si y es una restauración coherente CON
    #    ESA x, no solo si y "parece una foto real" (por eso es condicional).
    x = layers.Concatenate()([entrada, objetivo])  # (B, 256, 256, 6)

    # 3) Tres bloques con stride 2: cada uno reduce el tamaño a la mitad.
    #    El primero no lleva BatchNorm (igual que en el paper).
    x = _bloque_conv(x, ndf, 2, inicializador, usar_norm=False)  # (B,128,128, 64)
    x = _bloque_conv(x, ndf * 2, 2, inicializador)  # (B, 64, 64, 128)
    x = _bloque_conv(x, ndf * 4, 2, inicializador)  # (B, 32, 32, 256)

    # 4) Cuarto bloque con stride 1. Aquí NO usamos padding="same" porque
    #    queremos que el tamaño baje un poco: 1 de relleno a cada lado (34) y
    #    una convolución 4x4 sin relleno da 34 - 4 + 1 = 31.
    x = layers.ZeroPadding2D(padding=1)(x)  # (B, 34, 34, 256)
    x = layers.Conv2D(
        ndf * 8,
        kernel_size=4,
        strides=1,
        kernel_initializer=inicializador,
        use_bias=False,
    )(
        x
    )  # (B, 31, 31, 512)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 5) Capa final: un solo filtro y SIN activación. Devuelve logits, no
    #    probabilidades: la sigmoide la aplica la pérdida (BCE con logits),
    #    que es más estable numéricamente. 33 - 4 + 1 = 30.
    x = layers.ZeroPadding2D(padding=1)(x)  # (B, 33, 33, 512)
    logits = layers.Conv2D(
        1,
        kernel_size=4,
        strides=1,
        kernel_initializer=inicializador,
        name="logits_parches",
    )(
        x
    )  # (B, 30, 30, 1)

    return Model(
        inputs=[entrada, objetivo],
        outputs=logits,
        name="patchgan_discriminator",
    )
