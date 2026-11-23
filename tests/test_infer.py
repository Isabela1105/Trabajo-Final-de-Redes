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


# --- Formatos difíciles (PR-E6) ---------------------------------------------------------
def test_large_photo_does_not_crash(trained_checkpoint: Path) -> None:
    from src.gan.infer import MAX_SIDE

    generator, config = load_generator(trained_checkpoint)
    big = Image.fromarray(np.full((3000, 4000, 3), 90, np.uint8))  # 12 MP
    assert restore_image(generator, big, config.image_size).size == (4000, 3000)
    huge = Image.new("RGB", (MAX_SIDE * 2, 100), (90, 90, 90))
    assert max(restore_image(generator, huge, config.image_size).size) == MAX_SIDE


def test_transparent_png_is_composited_on_white() -> None:
    from src.gan.infer import prepare_input

    transparent = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    tensor = prepare_input(transparent, 64).tensor.numpy()
    np.testing.assert_allclose(tensor, 1.0, atol=1e-6)  # blanco, no negro


def test_exif_rotation_is_applied(trained_checkpoint: Path, tmp_path: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    path = tmp_path / "vertical.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientación: rotar 90° (foto de celular "acostada")
    Image.new("RGB", (80, 40), (120, 60, 30)).save(path, exif=exif.tobytes())
    with Image.open(path) as image:
        assert restore_image(generator, image, config.image_size).size == (40, 80)


@pytest.mark.parametrize("mode", ["L", "P", "LA", "I;16", "CMYK"])
def test_unusual_modes(trained_checkpoint: Path, mode: str) -> None:
    generator, config = load_generator(trained_checkpoint)
    image = Image.new("RGB", (30, 70), (200, 100, 50)).convert(mode)
    restored = restore_image(generator, image, config.image_size)
    assert restored.mode == "RGB" and restored.size == (30, 70)


def test_non_square_is_padded_not_stretched() -> None:
    from src.gan.infer import prepare_input

    prepared = prepare_input(Image.new("RGB", (100, 25), (10, 20, 30)), 64)
    assert prepared.side == 100
    assert prepared.box == (0, 37, 100, 62)
    assert tuple(prepared.tensor.shape) == (1, 64, 64, 3)


def test_one_pixel_wide_image(trained_checkpoint: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    line = Image.new("RGB", (1, 50), (0, 0, 0))
    assert restore_image(generator, line, config.image_size).size == (1, 50)


def test_config_found_in_output_folder(trained_checkpoint: Path, tmp_path: Path) -> None:
    """Checkpoints de versiones anteriores: config.yaml solo en la carpeta de salida."""
    import shutil

    old = tmp_path / "run" / "checkpoints"
    shutil.copytree(trained_checkpoint, old)
    shutil.move(old / "config.yaml", old.parent / "config.yaml")
    _, config = load_generator(old)
    assert config.image_size == 64
