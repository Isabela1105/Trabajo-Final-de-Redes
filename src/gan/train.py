"""Propósito: bucle de entrenamiento de Pix2Pix con D→G estricto, checkpoints y trazas.

Autor: Cristian Fabian Muñoz (@CristianMP355)
Entradas: `Config` (ver config.py) y, opcionalmente, modelos/pérdidas inyectados por tests.
Salidas: checkpoints autocontenidos, `metrics.csv`, muestras PNG y eventos de TensorBoard.
Referencia: docs/PLAN_COMMITS.md (PR-D2), docs/ARQUITECTURA.md §5 y GUIA-CristianMP355.md.

Nota de propiedad: este archivo NO define la arquitectura real. Los builders se resuelven por
rutas siempre que sea posible. Los modelos `build_smoke_*` son mínimos y solo se usan cuando el
YAML lo autoriza explícitamente (`allow_smoke_fallback`), para que CI/tests corran sin esperar a
los PR-B2/PR-C2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.gan.config import Config, load_config
from src.gan.data.dataset import build_dataset
from src.gan.data.tf_dataset import make_dataset
from src.gan.utils.viz import TensorBoardLogger, save_sample_grid

DEFAULT_GENERATOR_BUILDER = "src.gan.models.generator:build_generator"
DEFAULT_DISCRIMINATOR_BUILDER = "src.gan.models.discriminator:build_discriminator"
DEFAULT_LOSSES_MODULE = "src.gan.losses"

METRICS_FIELDS = ("epoch", "split", "loss_g", "loss_d", "l1", "psnr", "ssim")

TensorBatch = tuple[tf.Tensor, tf.Tensor]


# ---------------------------------------------------------------------------
# Componentes mínimos de smoke (NUNCA se usan en el resultado principal)
# ---------------------------------------------------------------------------
def build_smoke_generator(
    image_size: int = 64,
    ngf: int = 8,
    n_downsample: int = 4,
    norm: str = "instance",
    dropout: float = 0.5,
) -> tf.keras.Model:
    """Generador de una sola convolución, usado solo en pruebas de 2 pasos."""
    del ngf, n_downsample, norm, dropout  # la forma la fija `image_size`
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
    outputs = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh", name="output_image")(
        inputs
    )
    return tf.keras.Model(inputs, outputs, name="smoke_generator")


def build_smoke_discriminator(
    image_size: int = 64, ndf: int = 8, norm: str = "instance"
) -> tf.keras.Model:
    """Discriminador mínimo condicional, usado solo en pruebas de 2 pasos."""
    del image_size, ndf, norm
    input_image = tf.keras.Input(shape=(None, None, 3), name="input_image")
    target_image = tf.keras.Input(shape=(None, None, 3), name="target_image")
    x = tf.keras.layers.Concatenate()([input_image, target_image])
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    logits = tf.keras.layers.Conv2D(1, 3, padding="same")(x)
    return tf.keras.Model([input_image, target_image], logits, name="smoke_discriminator")


def smoke_generator_loss(
    fake_logits: tf.Tensor,
    fake_image: tf.Tensor,
    target_image: tf.Tensor,
    lambda_l1: float = 100.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Equivalente mínimo de `generator_loss` con BCE desde logits y L1."""
    adv = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_logits), logits=fake_logits
        )
    )
    l1 = tf.reduce_mean(tf.abs(target_image - fake_image))
    return adv + lambda_l1 * l1, adv, l1


def smoke_discriminator_loss(
    real_logits: tf.Tensor, fake_logits: tf.Tensor, label_smoothing: float = 0.0
) -> tf.Tensor:
    """Equivalente mínimo de `discriminator_loss`; devuelve un escalar."""
    real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_logits) * (1.0 - label_smoothing), logits=real_logits
        )
    )
    fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_logits), logits=fake_logits
        )
    )
    return 0.5 * (real + fake)


# ---------------------------------------------------------------------------
# Resolución de dependencias (inyección + importación configurable)
# ---------------------------------------------------------------------------
def _import_object(spec: str) -> Any:
    """Importa `modulo:atributo` indicado en el YAML."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Especificación inválida: {spec!r}. Use 'modulo:funcion'.")
    return getattr(importlib.import_module(module_name), attribute)


def _call_builder(builder: Callable[..., Any], config: Config) -> Any:
    """Pasa al builder solo los argumentos que acepta, sin hardcodear su firma.

    El YAML usa `model.n_downsample` (nombre del plan de commits), pero el builder real del
    generador recibe ese mismo concepto como `niveles`. Traducimos aquí el alias para respetar
    la firma de Juan Pablo sin tocar su archivo; los builders que no usan el parámetro lo
    ignoran igual que antes.
    """
    candidates: dict[str, Any] = {
        "image_size": config.image_size,
        **dict(config.model),
    }
    if "niveles" in inspect.signature(builder).parameters:
        candidates["niveles"] = candidates.get("n_downsample")
    signature = inspect.signature(builder)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    kwargs = (
        candidates
        if accepts_kwargs
        else {k: v for k, v in candidates.items() if k in signature.parameters}
    )
    return builder(**kwargs)


def _call_discriminator_loss(
    loss_fn: Callable[..., Any],
    real_logits: tf.Tensor,
    fake_logits: tf.Tensor,
    label_smoothing: float,
) -> tf.Tensor:
    """Adapta el etiquetado suave a la firma de la pérdida disponible.

    La pérdida real de Jeremy es `discriminator_loss(real_logits, fake_logits)` y no expone
    `label_smoothing`; la de smoke sí lo acepta. Como está prohibido modificar su módulo,
    pasamos el argumento solo cuando la función lo declara. Con el valor por defecto (0.0)
    el resultado es idéntico; si alguien pide smoothing con la pérdida real, avisamos en vez
    de romper la llamada.
    """
    parameters = inspect.signature(loss_fn).parameters
    if "label_smoothing" in parameters:
        return loss_fn(real_logits, fake_logits, label_smoothing)
    if label_smoothing:
        print(
            "[train] AVISO: discriminator_loss no acepta label_smoothing; "
            "se ignora. Solicitar el cambio al responsable de src/gan/losses.py."
        )
    return loss_fn(real_logits, fake_logits)


def _default_l1_loss(target: tf.Tensor, gen_output: tf.Tensor) -> tf.Tensor:
    """L1 de reserva, usada solo para descomponer/reportar la pérdida total."""
    return tf.reduce_mean(tf.abs(tf.cast(target, tf.float32) - tf.cast(gen_output, tf.float32)))


def resolve_components(config: Config) -> tuple[Any, Any, Callable[..., Any], Callable[..., Any]]:
    """Devuelve (generator, discriminator, generator_loss, discriminator_loss).

    Si faltan los módulos de otros integrantes y el YAML no autoriza el fallback, se lanza un
    error claro pidiendo abrir un issue en vez de editar archivos ajenos.
    """
    model_cfg = config.model
    allow_fallback = bool(config.train.get("allow_smoke_fallback", False))

    def load(spec: str, fallback: Callable[..., Any] | None = None) -> Callable[..., Any]:
        try:
            return _import_object(spec)
        except (ImportError, AttributeError, ValueError) as error:
            if fallback is not None and allow_fallback:
                return fallback
            raise RuntimeError(
                f"No se pudo importar {spec!r}. Falta un módulo de otro integrante; "
                "reporta el bloqueo en GitHub en vez de modificarlo."
            ) from error

    generator_fn = load(
        model_cfg.get("generator_builder", DEFAULT_GENERATOR_BUILDER), build_smoke_generator
    )
    discriminator_fn = load(
        model_cfg.get("discriminator_builder", DEFAULT_DISCRIMINATOR_BUILDER),
        build_smoke_discriminator,
    )
    losses_module = model_cfg.get("losses_module", DEFAULT_LOSSES_MODULE)
    generator_loss_fn = load(
        f"{losses_module}:generator_loss", fallback=smoke_generator_loss if allow_fallback else None
    )
    discriminator_loss_fn = load(
        f"{losses_module}:discriminator_loss",
        fallback=smoke_discriminator_loss if allow_fallback else None,
    )

    generator = _call_builder(generator_fn, config)
    discriminator = _call_builder(discriminator_fn, config)
    return generator, discriminator, generator_loss_fn, discriminator_loss_fn


# ---------------------------------------------------------------------------
# Datasets y paso de entrenamiento
# ---------------------------------------------------------------------------
def _validate_pair(input_image: tf.Tensor, target_image: tf.Tensor) -> TensorBatch:
    """Verifica en grafo el contrato del dataset: float32, shape fija y rango [-1, 1].

    Si Isabella cambia el contrato, el entrenamiento falla aquí con un mensaje claro
    en vez de propagar NaNs silenciosos al generador.
    """
    tolerance = 1e-3
    checks = []
    for image, name in ((input_image, "input"), (target_image, "target")):
        checks.extend(
            [
                tf.debugging.assert_equal(tf.shape(image)[-1], 3, message=f"{name}: canales != 3"),
                tf.debugging.assert_greater_equal(
                    tf.reduce_min(image), -1.0 - tolerance, message=f"{name}: valor < -1"
                ),
                tf.debugging.assert_less_equal(
                    tf.reduce_max(image), 1.0 + tolerance, message=f"{name}: valor > 1"
                ),
            ]
        )
    with tf.control_dependencies(checks):
        return tf.identity(input_image), tf.identity(target_image)


def make_tf_dataset(
    split_dir: str | Path,
    image_size: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Envolvente legada sobre el generador numpy de Isabella, conservada como fallback.

    El bucle usa `make_dataset` (loader tf.data real de `src/gan/data/tf_dataset.py`); esta
    función queda para scripts antiguos que aún no tienen esa ruta disponible.
    """

    def pairs() -> Any:
        for input_image, target_image in build_dataset(split_dir, image_size):
            yield (
                tf.convert_to_tensor(input_image, dtype=tf.float32),
                tf.convert_to_tensor(target_image, dtype=tf.float32),
            )

    signature = (
        tf.TensorSpec((image_size, image_size, 3), tf.float32, name="input_image"),
        tf.TensorSpec((image_size, image_size, 3), tf.float32, name="target_image"),
    )
    dataset = tf.data.Dataset.from_generator(pairs, output_signature=signature).map(_validate_pair)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(8, batch_size * 4), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _make_split_dataset(config: Config, split_dir: str | Path, *, shuffle: bool) -> tf.data.Dataset:
    """Crea un split con el cargador tf.data real subido por Isabella.

    `make_dataset(split_dir, batch_size, shuffle, seed, size, num_parallel_calls)` ya garantiza
    pares `(input, target)` float32 en [-1,1] con `tf.ensure_shape`, así que aquí solo pasamos
    los valores del YAML sin reimplementar ni modificar su módulo.
    """
    return make_dataset(
        split_dir,
        batch_size=config.batch_size,
        shuffle=shuffle,
        seed=config.seed,
        size=config.image_size,
        num_parallel_calls=int(config.data.get("num_parallel_calls", tf.data.AUTOTUNE)),
    )


def _apply_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: list[tf.Tensor],
    variables: list[tf.Tensor],
    clip_norm: float,
) -> None:
    """Aplica clip de gradientes y el paso del optimizador indicado."""
    pairs = [
        (grad, var) for grad, var in zip(gradients, variables, strict=False) if grad is not None
    ]
    if not pairs:
        return
    grads, vars_ = zip(*pairs, strict=False)
    if clip_norm > 0:
        grads, _ = tf.clip_by_global_norm(list(grads), clip_norm)
    optimizer.apply_gradients(zip(grads, vars_, strict=False))


def make_train_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    generator_loss_fn: Callable[..., Any],
    discriminator_loss_fn: Callable[..., Any],
    *,
    lambda_l1: float = 100.0,
    label_smoothing: float = 0.0,
    clip_norm: float = 1.0,
    use_gan: bool = True,
    l1_loss_fn: Callable[..., Any] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor], dict[str, tf.Tensor]]:
    """Construye el `train_step` compilado con una firma de entrada fija.

    Orden estricto: (1) actualizar D con G congelado; (2) actualizar G con D congelado.
    La firma de entrada usa dimensiones `None` para no recompilar al cambiar batch/imagen.

    `generator_loss_fn` puede devolver el escalar total (contrato real de `src/gan/losses.py`)
    o la tupla `(total, adversarial, l1)` de los modelos mínimos de smoke. En el primer caso
    descomponemos el total con `l1_loss_fn` solo para las métricas; el gradiente siempre viene
    de la pérdida original, sin reimplementarla.
    """
    calculate_l1 = l1_loss_fn or _default_l1_loss

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None, None, 3), tf.float32, name="input_image"),
            tf.TensorSpec((None, None, None, 3), tf.float32, name="target_image"),
        ],
    )
    def train_step(input_image: tf.Tensor, target_image: tf.Tensor) -> dict[str, tf.Tensor]:
        loss_d = tf.constant(0.0, dtype=tf.float32)

        if use_gan:
            # 1) DISCRIMINADOR: G se evalúa fuera del tape (training=False), por eso no entrena.
            with tf.GradientTape() as disc_tape:
                fake_for_d = generator(input_image, training=False)
                disc_real = discriminator([input_image, target_image], training=True)
                disc_fake = discriminator(
                    [input_image, tf.stop_gradient(fake_for_d)], training=True
                )
                # La pérdida siempre en float32, aunque el policy sea mixed_float16.
                loss_d = _call_discriminator_loss(
                    discriminator_loss_fn,
                    tf.cast(disc_real, tf.float32),
                    tf.cast(disc_fake, tf.float32),
                    label_smoothing,
                )

            disc_grads = disc_tape.gradient(loss_d, discriminator.trainable_variables)
            _apply_gradients(
                disc_optimizer, disc_grads, discriminator.trainable_variables, clip_norm
            )

        # 2) GENERADOR: D se evalúa con training=False y no se aplican sus gradientes.
        with tf.GradientTape() as gen_tape:
            gen_output = generator(input_image, training=True)
            if use_gan:
                disc_fake_for_g = discriminator([input_image, gen_output], training=False)
                adversarial_part = tf.cast(disc_fake_for_g, tf.float32)
            else:
                adversarial_part = tf.zeros_like(tf.cast(gen_output, tf.float32))
            gen_f = tf.cast(gen_output, tf.float32)
            target_f = tf.cast(target_image, tf.float32)
            loss_result = generator_loss_fn(adversarial_part, gen_f, target_f, lambda_l1)
            if isinstance(loss_result, tuple | list):
                # Pérdida de los modelos mínimos: ya devuelve sus tres componentes.
                gen_total, gen_adv, gen_l1 = loss_result[:3]
            else:
                # Pérdida real: devuelve solo el escalar total. La L1 se calcula para
                # reportarla; la diferenciación sigue usando el escalar devuelto.
                gen_total = loss_result
                gen_l1 = calculate_l1(target_f, gen_f)
                gen_adv = gen_total - lambda_l1 * gen_l1
                if not use_gan:
                    # Ablación pura: sin término adversarial (el bundle real siempre lo
                    # incluye con logits falsos, pero ese término no aporta gradiente).
                    gen_l1 = calculate_l1(target_f, gen_f)
                    gen_total = lambda_l1 * gen_l1
                    gen_adv = tf.zeros_like(gen_l1)

        gen_grads = gen_tape.gradient(gen_total, generator.trainable_variables)
        _apply_gradients(gen_optimizer, gen_grads, generator.trainable_variables, clip_norm)

        return {
            "loss_g": tf.cast(gen_total, tf.float32),
            "loss_adv": tf.cast(gen_adv, tf.float32),
            "loss_l1": tf.cast(gen_l1, tf.float32),
            "loss_d": tf.cast(loss_d, tf.float32),
        }

    return train_step


def make_eval_step(
    generator: tf.keras.Model,
) -> Callable[[tf.Tensor, tf.Tensor], dict[str, tf.Tensor]]:
    """Compila L1, PSNR y SSIM de validación, sin gradientes.

    Las imágenes están en [-1, 1], por eso `max_val=2.0` en PSNR/SSIM.
    """

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None, None, 3), tf.float32, name="input_image"),
            tf.TensorSpec((None, None, None, 3), tf.float32, name="target_image"),
        ],
    )
    def eval_step(input_image: tf.Tensor, target_image: tf.Tensor) -> dict[str, tf.Tensor]:
        fake = tf.cast(generator(input_image, training=False), tf.float32)
        target = tf.cast(target_image, tf.float32)
        return {
            "l1": tf.reduce_mean(tf.abs(fake - target)),
            "psnr": tf.reduce_mean(tf.image.psnr(fake, target, max_val=2.0)),
            "ssim": tf.reduce_mean(tf.image.ssim(fake, target, max_val=2.0)),
        }

    return eval_step


# ---------------------------------------------------------------------------
# Checkpoints y escritura segura de artefactos
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    """Escribe texto en un temporal y lo renombra (guardado atómico)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _config_hash(config: Config) -> str:
    """Huella del YAML original: permite detectar cambios de hiperparámetros al reanudar."""
    return hashlib.sha256(config.source.read_bytes()).hexdigest()


def build_checkpoint(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    seed: int,
) -> tf.train.Checkpoint:
    """Checkpoint con G, D, optimizadores y todo el estado de la corrida.

    Además de época y semilla se guarda el estado de early stopping (`best_val` y `waited`)
    para que un `--resume` en Colab no reinicie la paciencia desde cero.
    """
    return tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        epoch=tf.Variable(0, dtype=tf.int64, name="epoch"),
        seed=tf.Variable(int(seed), dtype=tf.int64, name="seed"),
        best_val=tf.Variable(float("inf"), dtype=tf.float32, name="best_val"),
        waited=tf.Variable(0, dtype=tf.int64, name="waited"),
    )


def _initialize_variables(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    image_size: int,
) -> None:
    """Crea variables de modelos y optimizadores antes de restaurar un checkpoint."""
    dummy = tf.zeros((1, image_size, image_size, 3), tf.float32)
    generator(dummy, training=False)
    discriminator([dummy, dummy], training=False)
    for model, optimizer in ((generator, gen_optimizer), (discriminator, disc_optimizer)):
        zero_grads = [tf.zeros_like(variable) for variable in model.trainable_variables]
        optimizer.apply_gradients(zip(zero_grads, model.trainable_variables, strict=False))


def save_checkpoint(
    manager: tf.train.CheckpointManager,
    checkpoint: tf.train.Checkpoint,
    config: Config,
    epoch: int,
    *,
    best_val: float = float("inf"),
    waited: int = 0,
) -> str:
    """Guarda G/D/optimizadores y escribe config.yaml + metadata.json de forma atómica.

    El checkpoint se considera válido solo si TensorFlow escribió el índice y además pudimos
    escribir el metadata; así un corte de energía en Colab nunca deja un checkpoint "a medias"
    que `restore_checkpoint` pueda confundir con uno bueno.
    """
    checkpoint.epoch.assign(int(epoch))
    checkpoint.seed.assign(int(config.seed))
    checkpoint.best_val.assign(float(best_val))
    checkpoint.waited.assign(int(waited))
    path = manager.save(checkpoint_number=int(epoch))
    if not path or not Path(f"{path}.index").exists():
        raise RuntimeError(f"TensorFlow no escribió un checkpoint válido en {path!r}.")

    directory = Path(path).parent
    metadata = {
        "epoch": int(epoch),
        "seed": int(config.seed),
        "checkpoint": Path(path).name,
        "config_sha256": _config_hash(config),
        "best_val": float(best_val),
        "waited": int(waited),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    # Volcar config/metadata; si algo falla, el checkpoint queda marcado como incompleto.
    _atomic_write_text(directory / "config.yaml", config.source.read_text(encoding="utf-8"))
    _atomic_write_text(
        directory / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return path


def restore_checkpoint(
    manager: tf.train.CheckpointManager,
    checkpoint: tf.train.Checkpoint,
    config: Config,
    resume_from: str | None = None,
) -> tuple[int, float, int]:
    """Restaura el estado y devuelve `(época, best_val, waited)`.

    Un checkpoint solo es reanudable si tiene su archivo `.index` y un `metadata.json` legible.
    Si falta alguno, se sigue desde cero en vez de arriesgarse a restaurar un estado corrupto.
    """
    path = resume_from or manager.latest_checkpoint
    if not path:
        return 0, float("inf"), 0

    path = str(path)
    if not Path(f"{path}.index").exists():
        print(f"[resume] AVISO: falta el índice de {path}; se ignora este checkpoint.")
        return 0, float("inf"), 0

    status = checkpoint.restore(path)
    status.expect_partial()  # los slots del optimizador pueden no existir aún en el smoke

    directory = Path(path).parent
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        print("[resume] AVISO: falta metadata.json; se ignora el checkpoint.")
        return 0, float("inf"), 0
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[resume] AVISO: metadata.json ilegible ({exc}); se ignora el checkpoint.")
        return 0, float("inf"), 0

    start_epoch = int(metadata.get("epoch", -1)) + 1
    best_val = float(metadata.get("best_val", float("inf")))
    waited = int(metadata.get("waited", 0))
    saved_hash = metadata.get("config_sha256")
    if saved_hash not in (None, _config_hash(config)):
        print(
            "[resume] AVISO: el YAML cambió desde el checkpoint; se reanuda con la config actual."
        )
    print(f"[resume] Estado restaurado desde {path}; continuando en época {start_epoch}.")
    return max(start_epoch, 0), best_val, waited


def _write_metrics_rows(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    """Escribe filas de `metrics.csv`, evitando duplicados al reanudar.

    Si el CSV ya contiene una fila para `(epoch, split)`, se reemplaza en vez de añadirse.
    Así un `--resume` interrumpido a mitad de época no deja dos filas compitiendo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if append and path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    new_keys = {(str(row["epoch"]), str(row["split"])) for row in rows}
    merged = [row for row in existing if (row.get("epoch"), row.get("split")) not in new_keys]
    merged.extend(rows)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(merged)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------
def _mean_metrics(
    step_fn: Callable[..., Any], dataset: tf.data.Dataset, max_steps: int | None
) -> dict[str, float]:
    totals: dict[str, float] = {}
    seen = 0
    for input_image, target_image in dataset:
        result = step_fn(input_image, target_image)
        values = result if isinstance(result, dict) else {"l1": result}
        for name, value in values.items():
            # Las métricas pueden llegar como escalar o como vector (un valor por imagen).
            totals[name] = totals.get(name, 0.0) + float(tf.reduce_mean(tf.cast(value, tf.float32)))
        seen += 1
        if max_steps is not None and seen >= max_steps:
            break
    return {name: total / max(seen, 1) for name, total in totals.items()}


def train(
    config: Config,
    *,
    generator: tf.keras.Model | None = None,
    discriminator: tf.keras.Model | None = None,
    generator_loss_fn: Callable[..., Any] | None = None,
    discriminator_loss_fn: Callable[..., Any] | None = None,
    resume: bool = False,
    resume_from: str | None = None,
    max_steps: int | None = None,
) -> Path:
    """Entrena Pix2Pix y devuelve la ruta de `metrics.csv`.

    Los modelos y pérdidas son inyectables: los tests pasan mocks y la CLI usa los builders
    indicados en el YAML. Así este módulo no depende de PRs aún no mergeados.
    """
    # La política debe fijarse ANTES de construir los modelos.
    if config.train.get("mixed_precision", False) and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[train] Mixed precision activada (pérdidas en float32).")

    if (
        generator is None
        or discriminator is None
        or generator_loss_fn is None
        or discriminator_loss_fn is None
    ):
        resolved = resolve_components(config)
        generator = generator or resolved[0]
        discriminator = discriminator or resolved[1]
        generator_loss_fn = generator_loss_fn or resolved[2]
        discriminator_loss_fn = discriminator_loss_fn or resolved[3]

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.train["lr_g"],
        beta_1=config.train["beta_1"],
        beta_2=config.train["beta_2"],
    )
    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.train["lr_d"],
        beta_1=config.train["beta_1"],
        beta_2=config.train["beta_2"],
    )

    train_ds = _make_split_dataset(config, config.data["train_dir"], shuffle=True)
    val_ds = _make_split_dataset(config, config.data["val_dir"], shuffle=False)

    _initialize_variables(
        generator, discriminator, gen_optimizer, disc_optimizer, config.image_size
    )

    checkpoint = build_checkpoint(
        generator, discriminator, gen_optimizer, disc_optimizer, config.seed
    )
    manager = tf.train.CheckpointManager(
        checkpoint, str(config.checkpoint_dir), max_to_keep=int(config.train.get("max_to_keep", 3))
    )

    start_epoch, best_val, waited = 0, float("inf"), 0
    if resume or resume_from:
        start_epoch, best_val, waited = restore_checkpoint(manager, checkpoint, config, resume_from)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(config.output_dir / "config.yaml", config.source.read_text(encoding="utf-8"))

    clip_norm = float(config.train["clip_norm"])
    losses_module = config.model.get("losses_module", DEFAULT_LOSSES_MODULE)
    try:
        real_l1_loss = _import_object(f"{losses_module}:l1_loss")
    except (ImportError, AttributeError, ValueError):
        real_l1_loss = None  # las pérdidas smoke usan su propia descomposición
    train_step = make_train_step(
        generator,
        discriminator,
        gen_optimizer,
        disc_optimizer,
        generator_loss_fn,
        discriminator_loss_fn,
        lambda_l1=float(config.train["lambda_l1"]),
        label_smoothing=float(config.train.get("label_smoothing", 0.0)),
        clip_norm=clip_norm,
        use_gan=bool(config.train.get("use_gan", True)),
        l1_loss_fn=real_l1_loss,
    )
    eval_step = make_eval_step(generator)

    logger = TensorBoardLogger(config.log_dir)
    sample_interval = max(1, int(config.train.get("sample_interval", 1)))
    checkpoint_interval = max(1, int(config.train.get("checkpoint_interval", 1)))
    patience = int(config.train.get("early_stopping_patience", 0) or 0)
    steps_limit = max_steps if max_steps is not None else config.train.get("max_steps")
    steps_limit = int(steps_limit) if steps_limit is not None else None

    # Un batch fijo de validación para que las muestras sean comparables entre épocas.
    sample_batch = next(iter(val_ds), None)

    started = time.time()
    try:
        for epoch in range(start_epoch, int(config.train["epochs"])):
            train_metrics = _mean_metrics(train_step, train_ds, steps_limit)
            val_metrics = _mean_metrics(eval_step, val_ds, steps_limit)
            val_l1 = val_metrics["l1"]

            logger.log_scalars(
                epoch,
                loss_g=train_metrics.get("loss_g", 0.0),
                loss_d=train_metrics.get("loss_d", 0.0),
                loss_adv=train_metrics.get("loss_adv", 0.0),
                loss_l1=train_metrics.get("loss_l1", 0.0),
                val_l1=val_l1,
                val_psnr=val_metrics["psnr"],
                val_ssim=val_metrics["ssim"],
            )

            if (
                config.output.get("save_samples", True)
                and sample_batch is not None
                and (epoch % sample_interval == 0 or epoch == int(config.train["epochs"]) - 1)
            ):
                sample_input, sample_target = sample_batch
                sample_pred = generator(sample_input, training=False)
                logger.log_images(epoch, sample_input, sample_target, sample_pred)
                if config.samples_dir:
                    save_sample_grid(
                        sample_input,
                        sample_target,
                        sample_pred,
                        config.samples_dir / f"epoch_{epoch:04d}.png",
                    )

            rows = [
                {
                    "epoch": epoch,
                    "split": "train",
                    "loss_g": train_metrics.get("loss_g", ""),
                    "loss_d": train_metrics.get("loss_d", ""),
                    "l1": train_metrics.get("loss_l1", ""),
                    "psnr": "",
                    "ssim": "",
                },
                {
                    "epoch": epoch,
                    "split": "val",
                    "loss_g": "",
                    "loss_d": "",
                    "l1": val_l1,
                    "psnr": val_metrics["psnr"],
                    "ssim": val_metrics["ssim"],
                },
            ]
            _write_metrics_rows(
                config.metrics_path, rows, append=resume or bool(resume_from) or epoch > start_epoch
            )

            if epoch % checkpoint_interval == 0 or epoch == int(config.train["epochs"]) - 1:
                save_checkpoint(
                    manager, checkpoint, config, epoch, best_val=best_val, waited=waited
                )

            print(
                f"época {epoch:03d} | D {train_metrics.get('loss_d', 0.0):.4f} "
                f"| G {train_metrics.get('loss_g', 0.0):.4f} | L1 val {val_l1:.4f}"
            )

            if patience:
                if val_l1 < best_val - 1e-4:
                    best_val, waited = val_l1, 0
                else:
                    waited += 1
                    if waited >= patience:
                        print(f"[train] Early stopping en la época {epoch} (paciencia {patience}).")
                        break
    finally:
        logger.close()

    print(f"[train] Listo en {time.time() - started:.1f}s. Métricas: {config.metrics_path}")
    return config.metrics_path


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada CLI: `python -m src.gan.train --config ... --resume`."""
    parser = argparse.ArgumentParser(description="Entrenamiento Pix2Pix (Rol D).")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML de configuración.")
    parser.add_argument("--resume", action="store_true", help="Reanuda desde el último checkpoint.")
    parser.add_argument("--resume-from", default=None, help="Ruta explícita de checkpoint.")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Limitación de pasos (smoke/CI)."
    )
    args = parser.parse_args(argv)

    train(
        load_config(args.config),
        resume=args.resume,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
