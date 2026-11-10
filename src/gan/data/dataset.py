"""Carga de imágenes, emparejamiento input/target y split por foto fuente.

Contrato con el entrenamiento: cada par es `(input, target)`, ambos arreglos float32
de forma (256, 256, 3) con valores en [-1, 1].
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 256
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def list_sources(raw_dir: str | Path) -> list[Path]:
    """Lista las fotos fuente de una carpeta, ordenadas por nombre (orden estable)."""
    raw_dir = Path(raw_dir)
    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def prepare_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Pasa a RGB y deja la imagen en size x size SIN deformarla.

    Se reduce hasta que el lado menor mida `size` y se recorta el centro.
    """
    img = img.convert("RGB")
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS)


def to_array(img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL RGB (0-255) en un arreglo float32 en [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32)
    return arr / 127.5 - 1.0


def load_image(path: str | Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Lee una imagen del disco y la devuelve como float32 (size, size, 3) en [-1, 1]."""
    with Image.open(path) as img:
        return to_array(prepare_image(img, size))


def load_pair(
    input_path: str | Path, target_path: str | Path, size: int = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Carga un par (input degradado, target limpio) listo para entrenar."""
    return load_image(input_path, size), load_image(target_path, size)


def build_dataset(
    split_dir: str | Path, size: int = IMAGE_SIZE
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Recorre `split_dir/input` y `split_dir/target` y entrega pares (input, target).

    Los archivos de input y target deben tener el mismo nombre.
    """
    split_dir = Path(split_dir)
    for input_path in sorted((split_dir / "input").iterdir()):
        target_path = split_dir / "target" / input_path.name
        if not target_path.exists():
            raise FileNotFoundError(f"Falta el target de {input_path.name}")
        yield load_pair(input_path, target_path, size)


def split_sources(
    source_ids: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Reparte las fotos FUENTE entre train/val/test (por foto, no por par).

    Una foto fuente y todas sus variantes caen en el mismo split. Con la misma semilla
    el reparto es siempre el mismo.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1, suman {sum(ratios)}")
    ids = sorted(set(source_ids))  # ordenar primero: el resultado no depende del orden de entrada
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    if n >= 3:  # con pocas fotos, garantizar al menos una en val y en test
        n_val, n_test = max(1, n_val), max(1, n_test)
    n_train = n - n_val - n_test
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def check_no_leakage(splits: dict[str, list[str]]) -> None:
    """Lanza ValueError si alguna foto fuente aparece en más de un split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(splits[a]) & set(splits[b])
            if shared:
                raise ValueError(f"Fuga de datos: {sorted(shared)} está en {a} y en {b}")