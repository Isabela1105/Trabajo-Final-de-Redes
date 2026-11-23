"""Pruebas de carga del checkpoint e inferencia sobre una imagen nueva.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.gan.infer import load_generator, main, resolve_checkpoint, restore_image


def test_resolve_checkpoint_accepts_folder_and_prefix(trained_checkpoint: Path) -> None:
    prefix = resolve_checkpoint(trained_checkpoint)
    assert Path(f"{prefix}.index").exists()
    assert resolve_checkpoint(prefix) == prefix


def test_resolve_checkpoint_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)


def test_generator_is_rebuilt_from_saved_config(trained_checkpoint: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    assert config.image_size == 64
    out = generator(tf.zeros((1, 64, 64, 3)), training=False)
    assert tuple(out.shape) == (1, 64, 64, 3)
    assert float(tf.reduce_max(tf.abs(out))) <= 1.0


def test_loading_twice_gives_identical_predictions(trained_checkpoint: Path) -> None:
    first, _ = load_generator(trained_checkpoint)
    second, _ = load_generator(trained_checkpoint)
    x = tf.random.stateless_uniform((1, 64, 64, 3), seed=(1, 2), minval=-1, maxval=1)
    np.testing.assert_allclose(
        first(x, training=False).numpy(), second(x, training=False).numpy(), atol=1e-6
    )


def test_restore_keeps_original_size(trained_checkpoint: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    image = Image.fromarray(np.full((50, 80, 3), 128, np.uint8))
    restored = restore_image(generator, image, config.image_size)
    assert restored.size == (80, 50)
    assert restored.mode == "RGB"


def test_cli_writes_png(trained_checkpoint: Path, tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(src)
    dst = tmp_path / "out" / "restaurada.png"
    main(["--checkpoint", str(trained_checkpoint), "--input", str(src), "--output", str(dst)])
    assert dst.exists()
