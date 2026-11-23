"""Fixtures compartidas de evaluación, inferencia y demo.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

`trained_checkpoint` entrena 2 pasos el generador y el discriminador REALES a 64×64 en CPU
y devuelve la carpeta de checkpoints. Se crea una sola vez por sesión de pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def write_pairs(directory: Path, *, size: int = 64, n_pairs: int = 4, seed: int = 42) -> None:
    """Pares degradado/limpio deterministas en `directory/{train,val}/{input,target}`."""
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for kind in ("input", "target"):
            (directory / split / kind).mkdir(parents=True, exist_ok=True)
        for index in range(n_pairs):
            target = (rng.random((size, size, 3)) * 255).astype(np.uint8)
            noisy = np.clip(
                target.astype(np.int16) + rng.integers(-20, 21, target.shape), 0, 255
            ).astype(np.uint8)
            Image.fromarray(noisy).save(directory / split / "input" / f"p{index:02d}.png")
            Image.fromarray(target).save(directory / split / "target" / f"p{index:02d}.png")


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.gan.config import load_config
    from src.gan.train import train

    root = tmp_path_factory.mktemp("eval")
    config = load_config("configs/smoke.yaml", fixed_seed=True)
    write_pairs(root / "data", size=config.image_size)
    config.data["train_dir"] = root / "data" / "train"
    config.data["val_dir"] = root / "data" / "val"
    config.data["test_dir"] = root / "data" / "val"
    config.output["dir"] = root / "outputs"
    config.train["allow_smoke_fallback"] = False
    config.train["epochs"] = 1
    config.train["max_steps"] = 2
    config.output["save_samples"] = False
    train(config, max_steps=2)
    return config.checkpoint_dir
