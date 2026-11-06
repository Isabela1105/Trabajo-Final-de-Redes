# Referencia de código comentada

> **Uso:** este archivo es material de estudio y punto de partida. No se copia tal cual al repo:
> cada dueño lo adapta a su módulo, lo prueba y lo explica. Las formas y los comentarios son los
> que se esperan en la entrega.

Requisitos: TensorFlow 2.16 / Keras 3, Python 3.11. Los nombres de variables están en inglés y
los comentarios en español, como pide `CONTRIBUTING.md`.

---

## 1. Generador U-Net

```python
"""Generador U-Net para Pix2Pix.

Autor: Juan Pablo D�vila (@Juanpablodavila)
Contrato:
    entrada: (B, image_size, image_size, 3) en [-1, 1]
    salida:  (B, image_size, image_size, 3) en [-1, 1]
"""

import tensorflow as tf
from tensorflow.keras import layers

# Inicialización estándar de GAN: media 0, desviación 0.02.
# Evita que todas las capas arranquen con la misma escala y acelera la convergencia.
INIT = tf.random_normal_initializer(0.0, 0.02)


def downsample(filters: int, size: int = 4, norm: str = "instance", apply_norm: bool = True):
    """Bloque encoder: convolución stride 2 + normalización + LeakyReLU."""
    block = tf.keras.Sequential(name=f"down_{filters}")
    block.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=INIT,
            use_bias=False,  # la normalización ya tiene sesgo
        )
    )
    if apply_norm:
        # InstanceNorm en vez de BatchNorm: con batch 1-4 BatchNorm es inestable.
        if norm == "instance":
            block.add(layers.InstanceNormalization())
        else:
            block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def upsample(filters: int, size: int = 4, norm: str = "instance", dropout: float = 0.0):
    """Bloque decoder: convolución transpuesta + normalización + dropout + ReLU."""
    block = tf.keras.Sequential(name=f"up_{filters}")
    block.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=INIT,
            use_bias=False,
        )
    )
    if norm == "instance":
        block.add(layers.InstanceNormalization())
    else:
        block.add(layers.BatchNormalization())
    if dropout > 0:
        block.add(layers.Dropout(dropout))
    block.add(layers.ReLU())
    return block


def build_generator(
    image_size: int = 256,
    ngf: int = 64,
    n_downsample: int = 8,
    norm: str = "instance",
    dropout: float = 0.5,
) -> tf.keras.Model:
    """Construye la U-Net.

    Los filtros crecen hasta ngf*8 y luego se quedan constantes, como en el paper.
    Para 256x256 se usan 8 niveles; para el smoke test de 64x64, 4 niveles.
    """
    # Ejemplo con ngf=64, n=8: [64, 128, 256, 512, 512, 512, 512, 512]
    filters = [min(ngf * (2**i), ngf * 8) for i in range(n_downsample)]

    # El primer bloque no normaliza: la imagen de entrada ya está en [-1, 1].
    down_stack = [
        downsample(f, 4, norm=norm, apply_norm=(i > 0)) for i, f in enumerate(filters)
    ]

    # El decoder tiene n-1 bloques; el último upsampling se hace con la capa de salida.
    up_stack = [
        upsample(f, 4, norm=norm, dropout=dropout if i < 3 else 0.0)
        for i, f in enumerate(reversed(filters[:-1]))
    ]

    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")
    x = inputs
    skips = []

    # Encoder: guardamos la salida de cada bloque para concatenarla después.
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Decoder: en cada nivel se concatena el skip correspondiente.
    # skips[:-1] porque el bottleneck no se concatena consigo mismo.
    for up, skip in zip(up_stack, reversed(skips[:-1])):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    # Salida 3 canales en [-1, 1]. tanh es el estándar para imágenes normalizadas así.
    outputs = layers.Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        kernel_initializer=INIT,
        activation="tanh",
        name="output_image",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="generator")
```

---

## 2. Discriminador PatchGAN

```python
"""Discriminador PatchGAN condicional.

Autor: Jeremy Caicedo (@JeremyCaicedo)
Contrato:
    entradas: [input (B,H,W,3), target (B,H,W,3)] en [-1, 1]
    salida:   (B,30,30,1) logits para H=W=256
"""

import tensorflow as tf
from tensorflow.keras import layers

INIT = tf.random_normal_initializer(0.0, 0.02)


def discriminator_block(filters: int, stride: int, norm: bool = True):
    """Bloque del discriminador: Conv + InstanceNorm opcional + LeakyReLU."""
    block = tf.keras.Sequential(name=f"disc_{filters}_s{stride}")
    block.add(
        layers.Conv2D(
            filters,
            4,
            strides=stride,
            padding="same",
            kernel_initializer=INIT,
            use_bias=False,
        )
    )
    if norm:
        block.add(layers.InstanceNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def build_discriminator(
    image_size: int = 256,
    ndf: int = 64,
    norm: str = "instance",
) -> tf.keras.Model:
    """PatchGAN de 70x70.

    Para 256x256:
      256 -> 128 -> 64 -> 32 (tres stride 2)
      ZeroPadding -> Conv stride 1 -> 31
      ZeroPadding -> Conv stride 1 -> 30
    Cada celda de la salida 30x30 representa un parche de 70x70.
    """
    input_image = layers.Input(shape=(image_size, image_size, 3), name="input_image")
    target_image = layers.Input(shape=(image_size, image_size, 3), name="target_image")

    # Condicional: el discriminador ve la entrada y la salida juntas.
    x = layers.Concatenate()([input_image, target_image])  # (B,H,W,6)

    x = discriminator_block(ndf, 2, norm=False)(x)  # 128
    x = discriminator_block(ndf * 2, 2)(x)          # 64
    x = discriminator_block(ndf * 4, 2)(x)          # 32

    # Las dos últimas convoluciones usan stride 1 y padding válido para producir
    # el mapa de parches (30x30) en vez de un único valor por imagen.
    x = layers.ZeroPadding2D()(x)                   # 34
    x = layers.Conv2D(
        ndf * 8, 4, strides=1, padding="valid", kernel_initializer=INIT, use_bias=False
    )(x)                                            # 31
    x = layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)                   # 33
    logits = layers.Conv2D(1, 4, strides=1, padding="valid", kernel_initializer=INIT)(x)

    return tf.keras.Model(inputs=[input_image, target_image], outputs=logits, name="discriminator")
```

---

## 3. Pérdidas

```python
"""Pérdidas de Pix2Pix.

Autor: Jeremy Caicedo (@JeremyCaicedo)
"""

import tensorflow as tf

# from_logits=True porque el discriminador devuelve logits, no probabilidades.
# Aplicar sigmoid dos veces es un error clásico: la pérdida se vuelve inestable.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_logits, fake_logits, label_smoothing: float = 0.0):
    """BCE real + BCE fake, promediadas.

    label_smoothing=0.1 hace que el discriminador no se vuelva demasiado confiado.
    """
    real_labels = tf.ones_like(real_logits) * (1.0 - label_smoothing)
    fake_labels = tf.zeros_like(fake_logits)
    real_loss = bce(real_labels, real_logits)
    fake_loss = bce(fake_labels, fake_logits)
    return 0.5 * (real_loss + fake_loss)


def generator_loss(fake_logits, fake_image, target_image, lambda_l1: float = 100.0):
    """Adversarial + L1 ponderado.

    Devuelve (total, adversarial, l1) para poder registrar cada término en TensorBoard.
    """
    adv_loss = bce(tf.ones_like(fake_logits), fake_logits)
    l1_loss = tf.reduce_mean(tf.abs(target_image - fake_image))
    total = adv_loss + lambda_l1 * l1_loss
    return total, adv_loss, l1_loss
```

---

## 4. Dataset

```python
"""Carga de pares (input, target) y normalización.

Autor: Isabella Ortiz (@Isabellaortizher-collab)
Contrato:
    cada imagen -> float32 (image_size, image_size, 3) en [-1, 1]
"""

import tensorflow as tf


def load_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Lee, redimensiona y normaliza una imagen a [-1, 1]."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0  # [0,255] -> [-1,1]


def load_pair(input_path: tf.Tensor, target_path: tf.Tensor, image_size: int):
    """Carga el par y aplica la misma augmentación a ambos."""
    input_image = load_image(input_path, image_size)
    target_image = load_image(target_path, image_size)

    # Flip horizontal consistente: si se aplica solo a uno, el par deja de ser válido.
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def build_dataset(input_paths, target_paths, image_size: int, batch_size: int, training: bool):
    """Crea un tf.data.Dataset con shuffle solo en entrenamiento."""
    ds = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(input_paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda i, t: load_pair(i, t, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

---

## 5. Paso de entrenamiento

```python
"""Bucle de entrenamiento de Pix2Pix.

Autor: Cristian Mu�oz (@CristianMP355)
"""

import tensorflow as tf
from src.gan.losses import discriminator_loss, generator_loss


@tf.function
def train_step(
    input_image,
    target_image,
    generator,
    discriminator,
    gen_optimizer,
    disc_optimizer,
    lambda_l1: float = 100.0,
    label_smoothing: float = 0.0,
):
    """Un paso de optimización: primero D, luego G (con D congelado).

    Los dos GradientTape son independientes para que los gradientes de G no
    actualicen D y viceversa.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        # El discriminador evalúa el par real y el par generado.
        disc_real = discriminator([input_image, target_image], training=True)
        disc_fake = discriminator([input_image, gen_output], training=True)

        gen_total, gen_adv, gen_l1 = generator_loss(
            disc_fake, gen_output, target_image, lambda_l1
        )
        disc_loss = discriminator_loss(disc_real, disc_fake, label_smoothing)

    # Gradientes solo respecto a las variables de cada modelo.
    gen_grads = gen_tape.gradient(gen_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Clip opcional: evita explosión de gradientes en las primeras épocas.
    gen_grads = [tf.clip_by_norm(g, 1.0) for g in gen_grads]
    disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_total, gen_adv, gen_l1, disc_loss
```

> **Orden de actualización:** en este esquema, el paso de G usa los logits de D calculados
> *antes* de actualizar D. Es el esquema estándar de Pix2Pix y es defendible. Si se quiere ser
> estricto, se recalcula `disc_fake` después de actualizar D, pero cuesta un forward extra.

---

## 6. Checkpoints y reanudación

```python
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator,
    discriminator=discriminator,
)
manager = tf.train.CheckpointManager(
    checkpoint,
    directory="outputs/base/checkpoints",
    max_to_keep=3,
)

# Guardar al final de cada época.
manager.save(checkpoint_number=epoch)

# Reanudar:
# checkpoint.restore(manager.latest_checkpoint)
# if manager.latest_checkpoint:
#     print(f"Restaurado desde {manager.latest_checkpoint}")
```

---

## 7. Errores clásicos y cómo evitarlos

| Error | Síntoma | Corrección |
|---|---|---|
| Normalizar train en [-1,1] e inferencia en [0,1] | Salida gris/negra | Un solo contrato y test de rango. |
| `sigmoid` + `from_logits=True` | Pérdida estancada | Usar logits crudos. |
| Flip solo a la entrada | Pares desalineados | Aplicar la misma augmentación a ambos. |
| Split por parche | Métricas infladas | Split por fuente. |
| D actualizado dentro del tape de G | Gradientes incorrectos | Tapes separados. |
| `BatchNorm` con batch 1 | Inestabilidad | InstanceNorm. |
| Guardar solo G | No se puede reanudar | Guardar G, D y optimizadores. |
| `tanh` sin normalizar | Salida saturada | Normalizar a [-1,1]. |
