"""Cargador de datos para TensorFlow: entrega las parejas (input, target) por lotes.

Contrato con el entrenamiento: cada elemento es `(input, target)`, ambos float32 de forma
(256, 256, 3) con valores en [-1, 1]. Con lotes, la forma es (B, 256, 256, 3). Es la misma
normalización que `load_image` (x / 127.5 - 1).

Lee `split_dir/input/*.png` y `split_dir/target/*.png`; cada input tiene un target con el
MISMO nombre. Esas carpetas las deja `scripts/build_dataset.py`.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.gan.data.dataset import IMAGE_SIZE


def list_pairs(split_dir: str | Path) -> tuple[list[str], list[str]]:
    """Rutas ordenadas de los inputs y de sus targets. Falla si falta alguno."""
    split_dir = Path(split_dir)
    inputs = sorted((split_dir / "input").glob("*.png"))
    if not inputs:
        raise FileNotFoundError(f"No hay imágenes en {split_dir / 'input'}")
    targets = []
    for p in inputs:
        target = split_dir / "target" / p.name
        if not target.exists():
            raise FileNotFoundError(f"Falta el target de {p.name}")
        targets.append(target)
    return [str(p) for p in inputs], [str(t) for t in targets]


def _decode(path: tf.Tensor, size: int) -> tf.Tensor:
    """Lee un PNG RGB y lo deja float32 en [-1, 1]. Falla si no mide size x size."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return tf.ensure_shape(img, [size, size, 3])


def make_dataset(
    split_dir: str | Path,
    batch_size: int = 2,
    shuffle: bool = False,
    seed: int = 42,
    size: int = IMAGE_SIZE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Crea un `tf.data.Dataset` de lotes `(input, target)` para un split.

    - `shuffle=True` (entrenamiento) mezcla las parejas en cada época; con la misma `seed`
      el orden es reproducible.
    - `shuffle=False` (validación y prueba) mantiene el orden alfabético.
    """
    inputs, targets = list_pairs(split_dir)
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        ds = ds.shuffle(len(inputs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda i, t: (_decode(i, size), _decode(t, size)),
        num_parallel_calls=num_parallel_calls,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_datasets(data_cfg: dict, seed: int = 42) -> tuple[tf.data.Dataset, ...]:
    """Crea (train, val, test) con la sección `data:` de `configs/base.yaml`."""
    common = {
        "batch_size": data_cfg.get("batch_size", 2),
        "size": data_cfg.get("image_size", IMAGE_SIZE),
        "num_parallel_calls": data_cfg.get("num_parallel_calls", tf.data.AUTOTUNE),
    }
    return (
        make_dataset(data_cfg["train_dir"], shuffle=True, seed=seed, **common),
        make_dataset(data_cfg["val_dir"], **common),
        make_dataset(data_cfg["test_dir"], **common),
    )