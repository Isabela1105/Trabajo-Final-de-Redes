"""Pruebas de la muestra sintética que usa el CI.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.make_synthetic_sample import main, make_sample
from src.gan.data.tf_dataset import list_pairs


def test_creates_pairs_for_every_split(tmp_path: Path) -> None:
    counts = make_sample(tmp_path, n=8, size=64)
    assert counts == {"train": 4, "val": 2, "test": 2}
    for split, count in counts.items():
        inputs, targets = list_pairs(tmp_path / split)
        assert len(inputs) == len(targets) == count
        with Image.open(inputs[0]) as image:
            assert image.size == (64, 64) and image.mode == "RGB"


def test_same_seed_same_pixels(tmp_path: Path) -> None:
    make_sample(tmp_path / "a", n=4, size=32, seed=7)
    make_sample(tmp_path / "b", n=4, size=32, seed=7)
    for rel in ("train/input/sample_00.png", "val/target/sample_00.png"):
        a = np.asarray(Image.open(tmp_path / "a" / rel))
        b = np.asarray(Image.open(tmp_path / "b" / rel))
        np.testing.assert_array_equal(a, b)


def test_input_differs_from_target(tmp_path: Path) -> None:
    make_sample(tmp_path, n=4, size=32)
    a = np.asarray(Image.open(tmp_path / "train/input/sample_00.png"), dtype=np.int16)
    b = np.asarray(Image.open(tmp_path / "train/target/sample_00.png"), dtype=np.int16)
    assert np.abs(a - b).mean() > 0


def test_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_sample(tmp_path, n=0)


def test_smoke_training_runs_on_the_sample(tmp_path: Path) -> None:
    """Lo mismo que hace el CI: muestra sintética + 2 pasos con los módulos reales."""
    from src.gan.config import load_config
    from src.gan.train import train

    main(["--out", str(tmp_path / "sample"), "--n", "8", "--size", "64"])
    config = load_config("configs/smoke.yaml")
    config.data["train_dir"] = tmp_path / "sample" / "train"
    config.data["val_dir"] = tmp_path / "sample" / "val"
    config.output["dir"] = tmp_path / "out"
    metrics = train(config, max_steps=2)
    assert metrics.exists()
