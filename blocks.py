"""Bloques de construcción del generador U-Net (TensorFlow/Keras).

Un bloque encoder reduce la imagen a la mitad y un bloque decoder la duplica.
El generador (generator.py) los apila y une con skip connections.
"""

from tensorflow.keras import initializers, layers

# Normalizaciones aceptadas. None significa "sin normalización".
NORMALIZACIONES = ("instance", "batch", None)


def crear_inicializador():
    """Pesos iniciales ~ Normal(0, 0.02), como en el paper de Pix2Pix."""
    return initializers.RandomNormal(mean=0.0, stddev=0.02)


def normalizar(x, tipo):
    """Aplica la normalización elegida.

    - "instance": normaliza cada imagen y cada canal por separado. No depende del
      tamaño del lote, por eso es estable con lotes de 1 a 4 imágenes (BatchNorm no).
      En Keras es GroupNormalization con un grupo por canal (groups=-1).
    - "batch": BatchNormalization clásica.
    - None: no hace nada.
    """
    if tipo not in NORMALIZACIONES:
        raise ValueError(f"norm debe ser uno de {NORMALIZACIONES}, no {tipo!r}")
    if tipo == "instance":
        return layers.GroupNormalization(groups=-1)(x)
    if tipo == "batch":
        return layers.BatchNormalization()(x)
    return x


def bloque_encoder(x, filtros, inicializador, norm="instance"):
    """Encoder: Conv 4x4 stride 2 -> (normalización) -> LeakyReLU(0.2).

    El stride 2 reduce el alto y el ancho a la mitad (256 -> 128 -> 64 ...).
    """
    x = layers.Conv2D(
        filtros,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=inicializador,
        # Si hay normalización, el sesgo sobra: la normalización ya tiene el suyo.
        use_bias=(norm is None),
    )(x)
    x = normalizar(x, norm)
    # LeakyReLU deja pasar un poco de señal negativa y evita neuronas "muertas".
    return layers.LeakyReLU(0.2)(x)


def bloque_decoder(x, filtros, inicializador, norm="instance", dropout=0.0):
    """Decoder: ConvTranspose 4x4 stride 2 -> normalización -> (Dropout) -> ReLU.

    El stride 2 duplica el alto y el ancho (1 -> 2 -> 4 ...). El dropout se usa
    solo en los primeros bloques del decoder: mete ruido que reemplaza al vector z
    de las GAN clásicas y evita que el generador memorice.
    """
    x = layers.Conv2DTranspose(
        filtros,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=inicializador,
        use_bias=(norm is None),
    )(x)
    x = normalizar(x, norm)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return layers.ReLU()(x)
