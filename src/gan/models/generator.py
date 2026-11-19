"""Generador U-Net de Pix2Pix (TensorFlow/Keras).

Contrato de este módulo (ver docs/ARQUITECTURA.md, sección 2):

    entrada : foto degradada       (B, H, W, 3), rango [-1, 1], batch variable
    salida  : imagen restaurada    (B, H, W, 3), rango [-1, 1] (tanh)

Funciona con H = W = 64, 128 o 256. El discriminador, el entrenamiento y la
evaluación dependen de este contrato: si cambia, hay que actualizar los tests y
ARQUITECTURA.md en el mismo PR.
"""

from tensorflow.keras import Input, Model, layers

from src.gan.models.blocks import (
    NORMALIZACIONES,
    bloque_decoder,
    bloque_encoder,
    crear_inicializador,
)

# Máximo de reducciones a la mitad: 256 -> 1 necesita 8. Más niveles no caben en GPUs
# pequeñas y no aportan para imágenes de 256 x 256.
MAX_NIVELES = 8


def niveles_por_defecto(image_size):
    """Cuántas veces hay que dividir entre 2 para llegar a 1 x 1 (tope: 8).

    256 -> 8, 128 -> 7, 64 -> 6.
    """
    return min(MAX_NIVELES, image_size.bit_length() - 1)


def build_generator(image_size=256, ngf=64, niveles=None, norm="instance", dropout=0.5):
    """Construye el generador U-Net.

    Args:
        image_size: lado de la imagen cuadrada (clave image_size de base.yaml).
        ngf: filtros de la primera capa (clave ngf de base.yaml). Los niveles
            siguientes lo duplican hasta un máximo de 8 * ngf.
        niveles: número de bloques del encoder. Por defecto, los necesarios para
            llegar a 1 x 1. image_size debe ser divisible por 2 ** niveles.
        norm: "instance" (por defecto), "batch" o None.
        dropout: tasa de dropout de los 3 primeros bloques del decoder (0 = sin dropout).

    Returns:
        Modelo Keras con una entrada (B, image_size, image_size, 3) y una salida
        con la misma forma, en [-1, 1].
    """
    if niveles is None:
        niveles = niveles_por_defecto(image_size)
    if not 1 <= niveles <= MAX_NIVELES:
        raise ValueError(f"niveles debe estar entre 1 y {MAX_NIVELES}, no {niveles}")
    if image_size % (2**niveles) != 0:
        raise ValueError(f"image_size={image_size} no es divisible por 2**{niveles}={2**niveles}")
    if norm not in NORMALIZACIONES:
        raise ValueError(f"norm debe ser uno de {NORMALIZACIONES}, no {norm!r}")

    inicializador = crear_inicializador()
    entrada = Input(shape=(image_size, image_size, 3), name="entrada_degradada")

    # 1) ENCODER. Cada bloque reduce el tamaño a la mitad y guardamos su salida para
    #    usarla después como skip connection. Los filtros se duplican hasta 8 * ngf.
    #    El primer bloque y el último (cuello de botella) van SIN normalización:
    #    el cuello de botella mide 1 x 1 y InstanceNorm lo dejaría en 0 (con un solo
    #    valor por canal, valor - media = 0), anulando la señal y el gradiente.
    filtros = [min(ngf * 2**i, ngf * 8) for i in range(niveles)]
    x = entrada
    saltos = []
    for i, f in enumerate(filtros):
        sin_norma = i == 0 or i == niveles - 1
        x = bloque_encoder(x, f, inicializador, norm=None if sin_norma else norm)
        saltos.append(x)

    # 2) DECODER con skip connections. La última salida del encoder (cuello de botella)
    #    es el punto de partida; las demás se usan en orden inverso: la más profunda
    #    primero. Cada skip tiene el mismo tamaño que la salida del bloque decoder,
    #    así que se pueden concatenar por canales. Esto lleva los detalles finos del
    #    encoder directo al decoder, sin pasar por el cuello de botella.
    saltos = list(reversed(saltos[:-1]))
    for i, salto in enumerate(saltos):
        x = bloque_decoder(
            x,
            salto.shape[-1],
            inicializador,
            norm=norm,
            dropout=dropout if i < 3 else 0.0,  # dropout solo en los 3 primeros
        )
        x = layers.Concatenate()([x, salto])

    # 3) Último bloque del decoder: recupera el tamaño original. No hay skip porque el
    #    encoder no tiene ninguna salida con la resolución de la entrada.
    x = bloque_decoder(x, ngf, inicializador, norm=norm)

    # 4) Salida: Conv 3x3 a 3 canales con tanh, que limita los valores a [-1, 1]
    #    (el mismo rango en que se normalizan las imágenes del dataset).
    salida = layers.Conv2D(
        3,
        kernel_size=3,
        padding="same",
        kernel_initializer=inicializador,
        activation="tanh",
        name="salida_tanh",
    )(x)

    return Model(inputs=entrada, outputs=salida, name="unet_generator")
