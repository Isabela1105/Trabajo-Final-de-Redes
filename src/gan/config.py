"""Propósito: cargar, validar y normalizar la configuración YAML del entrenamiento.

Autor: Cristian Fabian Muñoz (@CristianMP355)
Entradas: ruta a un archivo YAML con las secciones `seed`, `data`, `model`, `train` y `output`.
Salidas: objeto `Config` inmutable con rutas resueltas y semillas fijadas.
Referencia: docs/PLAN_COMMITS.md (PR-D1/D2) y configs/base.yaml.
"""

from __future__ import annotations

import random
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# config.py vive en <raiz>/src/gan/, así que parents[2] es la raíz del repositorio.
REPO_ROOT = Path(__file__).resolve().parents[2]

SECCIONES_OBLIGATORIAS = ("seed", "data", "model", "train", "output")
CLAVES_OBLIGATORIAS: dict[str, tuple[str, ...]] = {
    "data": ("train_dir", "val_dir", "image_size", "batch_size"),
    "model": ("ngf", "ndf", "n_downsample", "norm", "dropout"),
    "train": ("epochs", "lr_g", "lr_d", "beta_1", "beta_2", "lambda_l1", "clip_norm"),
    "output": ("dir",),
}
CLAVES_RUTA = ("train_dir", "val_dir", "test_dir")


def resolve_path(value: str | Path, root: Path = REPO_ROOT) -> Path:
    """Convierte una ruta del YAML en absoluta, relativa a la raíz del repo.

    Las rutas absolutas se dejan intactas; así el mismo config sirve en local y en Colab
    montando Drive en una ruta absoluta.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class Config:
    """Configuración validada y lista para usar desde el bucle de entrenamiento."""

    seed: int
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    output: dict[str, Any]
    source: Path

    @property
    def output_dir(self) -> Path:
        """Carpeta raíz de artefactos (checkpoints, logs, muestras y CSV)."""
        return resolve_path(self.output["dir"])

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def samples_dir(self) -> Path:
        return self.output_dir / "samples"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.csv"

    @property
    def image_size(self) -> int:
        return int(self.data["image_size"])

    @property
    def batch_size(self) -> int:
        return int(self.data["batch_size"])


def validate_config(raw: Any) -> None:
    """Lanza ValueError si el YAML no cumple el contrato mínimo del entrenamiento."""
    if not isinstance(raw, dict):
        raise ValueError("El YAML debe contener un diccionario en la raíz.")

    faltan = [k for k in SECCIONES_OBLIGATORIAS if k not in raw]
    if faltan:
        raise ValueError(f"Faltan secciones obligatorias en el YAML: {faltan}")

    for section, keys in CLAVES_OBLIGATORIAS.items():
        if not isinstance(raw[section], dict):
            raise ValueError(f"La sección `{section}` debe ser un diccionario.")
        faltan_keys = [k for k in keys if k not in raw[section]]
        if faltan_keys:
            raise ValueError(f"Faltan claves en `{section}`: {faltan_keys}")

    if not isinstance(raw["seed"], int) or raw["seed"] < 0:
        raise ValueError("`seed` debe ser un entero no negativo.")

    data = raw["data"]
    if int(data["image_size"]) <= 0:
        raise ValueError("`data.image_size` debe ser positivo.")
    if not 1 <= int(data["batch_size"]) <= 4:
        raise ValueError("`data.batch_size` debe estar entre 1 y 4 (límite de memoria en 256x256).")

    train = raw["train"]
    if int(train["epochs"]) <= 0:
        raise ValueError("`train.epochs` debe ser positivo.")
    for key in ("lr_g", "lr_d"):
        if float(train[key]) <= 0:
            raise ValueError(f"`train.{key}` debe ser positivo.")
    if float(train["lambda_l1"]) < 0:
        raise ValueError("`train.lambda_l1` no puede ser negativo.")
    if float(train["clip_norm"]) < 0:
        raise ValueError("`train.clip_norm` no puede ser negativo (0 desactiva el clip).")

    model = raw["model"]
    if str(model["norm"]).lower() not in {"instance", "batch"}:
        raise ValueError("`model.norm` debe ser 'instance' o 'batch'.")


def set_seed(seed: int) -> None:
    """Fija semillas de Python, NumPy y TensorFlow para carreras reproducibles."""
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # En CPU las operaciones deterministas evitan diferencias entre reanudaciones.
    with suppress(AttributeError, RuntimeError):
        tf.config.experimental.enable_op_determinism()


def load_config(path: str | Path, *, fixed_seed: bool = True) -> Config:
    """Lee un YAML, valida su contenido y devuelve un `Config` con rutas absolutas.

    Args:
        path: ruta al YAML (relativa a la raíz del repositorio si no es absoluta).
        fixed_seed: si es True, fija las semillas globales con el valor del YAML.
    """
    source = resolve_path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    validate_config(raw)

    # Resolver rutas de datos una sola vez: los módulos aguas abajo reciben Paths absolutos.
    data = dict(raw["data"])
    for key in CLAVES_RUTA:
        if key in data and data[key]:
            data[key] = resolve_path(data[key])

    config = Config(
        seed=int(raw["seed"]),
        data=data,
        model=dict(raw["model"]),
        train=dict(raw["train"]),
        output=dict(raw["output"]),
        source=source,
    )
    if fixed_seed:
        set_seed(config.seed)
    return config
