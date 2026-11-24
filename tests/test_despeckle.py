"""Pruebas de la limpieza de puntos de polvo.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.gan.data.degradations import dust
from src.gan.despeckle import detect_specks, remove_specks


def _scene(size: int = 96) -> Image.Image:
    """Degradado suave con un objeto oscuro grande y un cable fino: lo que NO se debe borrar."""
    y, x = np.mgrid[0:size, 0:size] / size
    base = np.stack([0.35 + 0.3 * x, 0.45 + 0.2 * y, 0.55 - 0.1 * x], axis=2)
    image = Image.fromarray((base * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.ellipse([10, 55, 40, 85], fill=(10, 10, 10))  # objeto oscuro grande
    draw.line([(0, 20), (size, 30)], fill=(5, 5, 5), width=1)  # cable negro fino
    return image


def test_clean_image_is_left_untouched() -> None:
    clean = np.asarray(_scene(), dtype=np.float32) / 255
    cleaned, changed = remove_specks(clean)
    assert changed == 0
    np.testing.assert_array_equal(cleaned, clean)


def test_dust_dots_are_removed_and_restore_the_background() -> None:
    clean = _scene()
    dirty = dust(clean, np.random.default_rng(3))
    clean_a = np.asarray(clean, dtype=np.float32) / 255
    dirty_a = np.asarray(dirty, dtype=np.float32) / 255
    cleaned, changed = remove_specks(dirty_a)
    assert changed > 0
    error_before = np.abs(dirty_a - clean_a).mean()
    error_after = np.abs(cleaned - clean_a).mean()
    assert error_after < 0.4 * error_before


def test_thin_lines_and_large_dark_objects_survive() -> None:
    clean = np.asarray(_scene(), dtype=np.float32) / 255
    mask = detect_specks(clean)
    assert not mask[20:31, :].any()  # el cable
    assert not mask[55:86, 10:41].any()  # el objeto oscuro


def test_isolated_white_and_black_specks_are_detected() -> None:
    image = np.full((40, 40, 3), 0.5, np.float32)
    image[10:13, 10:13] = 1.0
    image[25:28, 25:28] = 0.0
    mask = detect_specks(image)
    assert mask[11, 11] and mask[26, 26]
    assert mask.sum() < 60  # solo las dos manchas y su borde


def test_single_noise_pixel_is_not_dust() -> None:
    image = np.full((30, 30, 3), 0.5, np.float32)
    image[15, 15] = 0.0
    assert not detect_specks(image).any()


def test_colored_spot_is_not_dust() -> None:
    image = np.full((30, 30, 3), 0.5, np.float32)
    image[10:13, 10:13] = [0.9, 0.05, 0.05]  # rojo: no es polvo de escáner
    assert not detect_specks(image).any()
