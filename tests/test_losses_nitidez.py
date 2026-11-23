"""Pruebas de las pérdidas del experimento de nitidez.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Se usan pesos aleatorios de VGG16 (`PIX2PIX_VGG_WEIGHTS=none`) para no descargar nada en CI;
las propiedades probadas no dependen de los pesos.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["PIX2PIX_VGG_WEIGHTS"] = "none"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.gan import losses, losses_nitidez  # noqa: E402
from src.gan.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def images() -> tuple[tf.Tensor, tf.Tensor]:
    rng = np.random.default_rng(0)
    target = tf.constant(rng.uniform(-1, 1, (2, 64, 64, 3)), tf.float32)
    blurred = tf.nn.avg_pool2d(target, 5, 1, "SAME")
    return target, blurred


def test_perceptual_is_zero_for_identical_images(images) -> None:
    target, _ = images
    assert float(losses_nitidez.perceptual_loss(target, target)) == pytest.approx(0.0, abs=1e-6)


def test_perceptual_penalizes_blur(images) -> None:
    target, blurred = images
    assert float(losses_nitidez.perceptual_loss(target, blurred)) > 0.05


def test_perceptual_gradient_reaches_the_generator_output(images) -> None:
    target, blurred = images
    variable = tf.Variable(blurred)
    with tf.GradientTape() as tape:
        loss = losses_nitidez.perceptual_loss(target, variable)
    grad = tape.gradient(loss, variable)
    assert grad is not None and float(tf.reduce_sum(tf.abs(grad))) > 0


def test_generator_loss_returns_parts_and_adds_perceptual(images) -> None:
    target, blurred = images
    logits = tf.zeros((2, 6, 6, 1))
    total, adv, l1 = losses_nitidez.generator_loss(logits, blurred, target, 50.0)
    base = losses.generator_loss(logits, blurred, target, 50.0)
    assert float(adv + 50.0 * l1) == pytest.approx(float(base), rel=1e-6)
    assert float(total) > float(base)  # la perceptual suma algo positivo


def test_label_smoothing_changes_d_loss_and_zero_matches_original() -> None:
    real = tf.fill((2, 30, 30, 1), 4.0)
    fake = tf.fill((2, 30, 30, 1), -4.0)
    original = float(losses.discriminator_loss(real, fake))
    assert float(losses_nitidez.discriminator_loss(real, fake, 0.0)) == pytest.approx(original)
    # Con D muy seguro, apuntar a 0.9 en vez de 1.0 sube la pérdida: D ya no "gana gratis".
    assert float(losses_nitidez.discriminator_loss(real, fake, 0.1)) > original


def test_extractor_stays_float32_under_mixed_precision() -> None:
    previous = tf.keras.mixed_precision.global_policy()
    losses_nitidez._extractor = None
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = losses_nitidez.get_extractor()
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
    assert model.layers[-1].compute_dtype == "float32"
    assert not model.trainable


def test_config_is_valid_and_differs_from_base_only_where_intended() -> None:
    base = load_config("configs/base.yaml", fixed_seed=False)
    sharp = load_config("configs/nitidez.yaml", fixed_seed=False)
    assert sharp.model["losses_module"] == "src.gan.losses_nitidez"
    assert sharp.train["lambda_l1"] == 50.0
    assert sharp.train["label_smoothing"] == 0.1
    assert sharp.train["lr_d"] == base.train["lr_d"] / 2
    assert Path(str(sharp.output["dir"])).name == "nitidez"
    for key in ("ngf", "ndf", "n_downsample", "norm", "dropout"):
        assert sharp.model[key] == base.model[key]


def test_two_real_training_steps_with_the_new_losses(tmp_path: Path) -> None:
    from src.gan.train import train
    from tests.conftest import write_pairs

    losses_nitidez._extractor = None
    config = load_config("configs/smoke.yaml", fixed_seed=True)
    write_pairs(tmp_path / "data", size=config.image_size)
    config.data["train_dir"] = tmp_path / "data" / "train"
    config.data["val_dir"] = tmp_path / "data" / "val"
    config.output["dir"] = tmp_path / "out"
    config.model["losses_module"] = "src.gan.losses_nitidez"
    config.train["allow_smoke_fallback"] = False
    config.train["label_smoothing"] = 0.1
    config.output["save_samples"] = False
    metrics = train(config, max_steps=2)
    rows = metrics.read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 3
    assert "nan" not in metrics.read_text(encoding="utf-8").lower()
