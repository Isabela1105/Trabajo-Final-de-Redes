"""Integración real con los módulos ya mergeados en `main` (PR-D4).

Comprueba que `train()` usa los builders, pérdidas y cargador tf.data reales sin haber
modificado ningún archivo de otro integrante. Las formas son pequeñas (64x64, ngf/ndf=8)
para que la prueba corra en CPU en menos de 2 minutos.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.gan.config import Config, load_config
from src.gan.losses import discriminator_loss
from src.gan.train import _call_discriminator_loss, resolve_components, train


def _write_synthetic_pairs(directory: Path, *, size: int = 64, n_pairs: int = 4) -> None:
    """Pares deterministas input/target con la estructura que espera `make_dataset`."""
    rng = np.random.default_rng(7)
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


@pytest.fixture
def integration_config(tmp_path: Path) -> Config:
    """Config de smoke apuntando a los builders reales y a datos aislados en `tmp_path`."""
    config = load_config("configs/smoke.yaml", fixed_seed=True)
    data_dir = tmp_path / "data"
    _write_synthetic_pairs(data_dir, size=config.image_size)
    config.data["train_dir"] = data_dir / "train"
    config.data["val_dir"] = data_dir / "val"
    config.data["test_dir"] = data_dir / "val"
    config.output["dir"] = tmp_path / "outputs"
    config.model["generator_builder"] = "src.gan.models.generator:build_generator"
    config.model["discriminator_builder"] = "src.gan.models.discriminator:build_discriminator"
    config.model["losses_module"] = "src.gan.losses"
    config.train["epochs"] = 1
    config.train["max_steps"] = 1
    return config


def test_real_modules_train_end_to_end(integration_config: Config) -> None:
    """G y D reales aprenden y el pipeline escribe todos los artefactos."""
    generator, discriminator, gen_loss_fn, disc_loss_fn = resolve_components(integration_config)
    before_g = [variable.numpy().copy() for variable in generator.trainable_variables]
    before_d = [variable.numpy().copy() for variable in discriminator.trainable_variables]

    metrics_path = train(
        integration_config,
        generator=generator,
        discriminator=discriminator,
        generator_loss_fn=gen_loss_fn,
        discriminator_loss_fn=disc_loss_fn,
        max_steps=1,
    )

    for name, before, model in (
        ("generador", before_g, generator),
        ("discriminador", before_d, discriminator),
    ):
        after = [variable.numpy() for variable in model.trainable_variables]
        assert len(before) == len(after)
        assert any(
            not np.allclose(old, new) for old, new in zip(before, after, strict=True)
        ), f"El {name} real no actualizó sus pesos."

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["train", "val"]
    assert (integration_config.checkpoint_dir / "checkpoint").exists()
    assert list(integration_config.samples_dir.glob("epoch_*.png"))
    assert list(integration_config.log_dir.glob("events.out.tfevents.*"))


def test_real_ablation_only_trains_generator(integration_config: Config) -> None:
    """Con los módulos reales, `use_gan: false` no debe mover ni un peso de D."""
    integration_config.train["use_gan"] = False
    generator, discriminator, gen_loss_fn, disc_loss_fn = resolve_components(integration_config)
    before_d = [variable.numpy().copy() for variable in discriminator.trainable_variables]

    train(
        integration_config,
        generator=generator,
        discriminator=discriminator,
        generator_loss_fn=gen_loss_fn,
        discriminator_loss_fn=disc_loss_fn,
        max_steps=1,
    )

    after_d = [variable.numpy() for variable in discriminator.trainable_variables]
    assert all(np.allclose(old, new) for old, new in zip(before_d, after_d, strict=True))


def test_real_discriminator_loss_ignores_label_smoothing() -> None:
    """La pérdida real no acepta `label_smoothing`; el adaptador no debe romper la llamada."""
    logits = tf.zeros((2, 6, 6, 1), dtype=tf.float32)
    result = _call_discriminator_loss(discriminator_loss, logits, logits, 0.2)
    assert float(result) == pytest.approx(math.log(2.0), rel=1e-5)
