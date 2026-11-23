"""Pruebas de PSNR, SSIM y L1 con valores conocidos.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gan.metrics import PSNR_MAXIMO, all_metrics, l1, psnr, ssim, to_unit_range


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((2, 64, 64, 3)).astype(np.float32)


def test_identical_image_gives_perfect_scores(image: np.ndarray) -> None:
    scores = all_metrics(image, image)
    np.testing.assert_allclose(scores["ssim"], 1.0, atol=1e-5)
    np.testing.assert_allclose(scores["l1"], 0.0, atol=1e-7)
    assert np.all(np.isfinite(scores["psnr"]))
    np.testing.assert_allclose(scores["psnr"], PSNR_MAXIMO, rtol=1e-4)


def test_psnr_matches_closed_form() -> None:
    target = np.zeros((1, 8, 8, 3), np.float32)
    pred = np.full_like(target, 0.1)  # MSE = 0.01 -> PSNR = 20 dB
    np.testing.assert_allclose(psnr(pred, target), 20.0, rtol=1e-5)


def test_l1_matches_closed_form() -> None:
    target = np.zeros((1, 8, 8, 3), np.float32)
    pred = np.full_like(target, 0.25)
    np.testing.assert_allclose(l1(pred, target), 0.25, rtol=1e-6)


def test_metrics_are_per_image(image: np.ndarray) -> None:
    noisy = np.clip(image + np.array([0.0, 0.2], np.float32)[:, None, None, None], 0, 1)
    scores = all_metrics(noisy, image)
    for values in scores.values():
        assert values.shape == (2,)
    assert scores["psnr"][0] > scores["psnr"][1]
    assert scores["ssim"][0] > scores["ssim"][1]
    assert scores["l1"][0] < scores["l1"][1]


def test_more_noise_means_worse_scores(image: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    small = np.clip(image + rng.normal(0, 0.02, image.shape), 0, 1).astype(np.float32)
    large = np.clip(image + rng.normal(0, 0.2, image.shape), 0, 1).astype(np.float32)
    assert psnr(small, image).mean() > psnr(large, image).mean()
    assert ssim(small, image).mean() > ssim(large, image).mean()


def test_unit_range_conversion() -> None:
    values = np.array([-1.5, -1.0, 0.0, 1.0, 1.5], np.float32)
    np.testing.assert_allclose(to_unit_range(values).numpy(), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_single_image_without_batch_axis(image: np.ndarray) -> None:
    assert psnr(image[0], image[0]).shape == (1,)


def test_shape_mismatch_fails(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        l1(image, image[:, :32])
