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


def test_restore_folder_script(trained_checkpoint: Path, tmp_path: Path) -> None:
    from scripts.restore_folder import restore_folder

    src = tmp_path / "real"
    src.mkdir()
    for i, size in enumerate([(40, 60), (70, 30), (50, 50)]):
        Image.new("RGB", size, (100 + i * 30, 90, 80)).save(src / f"real_{i:03d}.jpg")
    (src / "notas.txt").write_text("x", encoding="utf-8")
    grid = restore_folder(
        {"Pix2Pix": trained_checkpoint, "U-Net solo L1": trained_checkpoint}, src, tmp_path / "out"
    )
    assert grid.exists()
    restored = sorted((tmp_path / "out" / "Pix2Pix").glob("*.png"))
    assert [p.stem for p in restored] == ["real_000", "real_001", "real_002"]
    assert Image.open(restored[1]).size == (70, 30)
    assert len(list((tmp_path / "out" / "U-Net_solo_L1").glob("*.png"))) == 3


def test_cleaning_can_be_disabled(trained_checkpoint: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    image = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
    with_clean = restore_image(generator, image, config.image_size)
    without = restore_image(generator, image, config.image_size, quitar_puntos=False)
    assert with_clean.size == without.size == (64, 64)


def test_best_generator_is_saved_and_loadable(trained_checkpoint: Path) -> None:
    import json

    best = trained_checkpoint.parent / "best"
    info = json.loads((best / "best.json").read_text(encoding="utf-8"))
    assert info["epoch"] == 0 and info["val_l1"] > 0
    generator, config = load_generator(best)
    image = Image.fromarray(np.full((40, 30, 3), 90, np.uint8))
    assert restore_image(generator, image, config.image_size).size == (30, 40)


def test_tta_undoes_each_flip() -> None:
    from src.gan.infer import generate

    class Shift(tf.keras.Model):
        """Generador falso que depende de la posición: sin deshacer el volteo, falla."""

        def call(self, x, training=False):
            ramp = tf.reshape(tf.linspace(-0.2, 0.2, x.shape[2]), (1, 1, -1, 1))
            return x + ramp

    x = tf.random.uniform((2, 8, 8, 3), -0.5, 0.5, seed=1)
    out = generate(Shift(), x, tta=True).numpy()
    # el promedio de rampa y rampa invertida es 0: queda la entrada
    np.testing.assert_allclose(out, x.numpy(), atol=1e-5)
    np.testing.assert_allclose(generate(Shift(), x).numpy(), Shift()(x).numpy(), atol=1e-6)


def test_sharpen_is_noop_at_zero_and_boosts_edges() -> None:
    from src.gan.infer import sharpen

    x = np.zeros((1, 16, 16, 3), np.float32)
    x[:, :, 8:] = 0.5
    np.testing.assert_array_equal(sharpen(x, 0.0).numpy(), x)
    y = sharpen(x, 1.0).numpy()
    assert y[0, 5, 8, 0] > 0.5 and y[0, 5, 7, 0] < 0.0  # sobrepaso a ambos lados del borde
    assert y.min() >= -1.0 and y.max() <= 1.0


def test_restore_image_with_all_options(trained_checkpoint: Path) -> None:
    generator, config = load_generator(trained_checkpoint)
    image = Image.fromarray(np.full((48, 32, 3), 100, np.uint8))
    out = restore_image(
        generator, image, config.image_size, quitar_puntos=True, tta=True, nitidez=0.5
    )
    assert out.size == (32, 48)
