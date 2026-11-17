"""Propósito: smoke test de 2 pasos en CPU que verifica que G y D sí aprenden.

Autor: Cristian Fabian Muñoz (@CristianMP355)
Entradas: `configs/smoke.yaml` + pares sintéticos creados en `tmp_path`.
Salidas: falla si G o D no actualizan pesos, o si faltan checkpoints/muestras/métricas.
Referencia: docs/PLAN_COMMITS.md (PR-D3) y .github/workflows/ci.yml.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.gan.config import Config, load_config
from src.gan.train import (
    build_smoke_discriminator,
    build_smoke_generator,
    make_train_step,
    smoke_discriminator_loss,
    smoke_generator_loss,
    train,
)


def _write_synthetic_pairs(directory: Path, *, size: int = 64, n_pairs: int = 4) -> None:
    """Crea pares degradado/limpio deterministas, sin depender del PR-F3."""
    rng = np.random.default_rng(42)
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
def smoke_config(tmp_path: Path) -> Config:
    """Config de smoke con rutas aisladas en `tmp_path` (no ensucia el repo)."""
    config = load_config("configs/smoke.yaml", fixed_seed=True)
    data_dir = tmp_path / "data"
    _write_synthetic_pairs(data_dir, size=config.image_size)
    config.data["train_dir"] = data_dir / "train"
    config.data["val_dir"] = data_dir / "val"
    config.data["test_dir"] = data_dir / "val"
    config.output["dir"] = tmp_path / "outputs"
    config.train["epochs"] = 1
    config.train["max_steps"] = 2
    return config


def _weights_changed(before: list[np.ndarray], model: object) -> bool:
    """True si al menos una variable entrenable cambió respecto a `before`."""
    after = [variable.numpy() for variable in model.trainable_variables]
    return len(before) == len(after) and any(
        not np.allclose(old, new) for old, new in zip(before, after, strict=True)
    )


def test_smoke_train_updates_both_networks_and_writes_artifacts(smoke_config: Config) -> None:
    """Dos pasos en CPU deben mover los pesos de G y D y dejar los artefactos esperados."""
    generator = build_smoke_generator(image_size=smoke_config.image_size)
    discriminator = build_smoke_discriminator()
    before_g = [variable.numpy().copy() for variable in generator.trainable_variables]
    before_d = [variable.numpy().copy() for variable in discriminator.trainable_variables]

    metrics_path = train(
        smoke_config,
        generator=generator,
        discriminator=discriminator,
        generator_loss_fn=smoke_generator_loss,
        discriminator_loss_fn=smoke_discriminator_loss,
        max_steps=2,
    )

    assert _weights_changed(before_g, generator), "El generador no actualizó sus pesos."
    assert _weights_changed(before_d, discriminator), "El discriminador no actualizó sus pesos."

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["split"] for row in rows] == ["train", "val"]
    assert all(row["epoch"] == "0" for row in rows)
    assert (smoke_config.checkpoint_dir / "checkpoint").exists()
    assert (smoke_config.checkpoint_dir / "metadata.json").exists()
    assert (smoke_config.checkpoint_dir / "config.yaml").exists()
    assert list(smoke_config.samples_dir.glob("epoch_*.png")), "Falta la muestra visual por época."
    assert list(
        smoke_config.log_dir.glob("events.out.tfevents.*")
    ), "Faltan eventos de TensorBoard."


def test_smoke_train_resume_continues_from_last_checkpoint(smoke_config: Config) -> None:
    """--resume debe continuar en la época siguiente sin reescribir el CSV."""
    smoke_config.train["epochs"] = 2
    smoke_config.train["max_steps"] = 1
    train(
        smoke_config,
        generator=build_smoke_generator(image_size=smoke_config.image_size),
        discriminator=build_smoke_discriminator(),
        generator_loss_fn=smoke_generator_loss,
        discriminator_loss_fn=smoke_discriminator_loss,
    )

    smoke_config.train["epochs"] = 3
    train(
        smoke_config,
        generator=build_smoke_generator(image_size=smoke_config.image_size),
        discriminator=build_smoke_discriminator(),
        generator_loss_fn=smoke_generator_loss,
        discriminator_loss_fn=smoke_discriminator_loss,
        resume=True,
    )

    with smoke_config.metrics_path.open(newline="", encoding="utf-8") as handle:
        epochs = {int(row["epoch"]) for row in csv.DictReader(handle)}
    assert epochs == {0, 1, 2}, f"Épocas esperadas {{0,1,2}}, se obtuvieron {sorted(epochs)}."
    with (smoke_config.checkpoint_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["epoch"] == 2
    assert "best_val" in metadata and "waited" in metadata


def test_resume_does_not_duplicate_metrics_rows(smoke_config: Config) -> None:
    """Reanudar dos veces desde el mismo checkpoint no debe duplicar filas del CSV."""
    kwargs = {
        "generator": build_smoke_generator(image_size=smoke_config.image_size),
        "discriminator": build_smoke_discriminator(),
        "generator_loss_fn": smoke_generator_loss,
        "discriminator_loss_fn": smoke_discriminator_loss,
        "max_steps": 1,
    }
    smoke_config.train["epochs"] = 2
    train(smoke_config, **kwargs)
    # Reanudar desde el último checkpoint (época 1) sin avanzar más épocas.
    kwargs["resume_from"] = str(next(smoke_config.checkpoint_dir.glob("ckpt-1*")))
    smoke_config.train["epochs"] = 2
    train(smoke_config, **kwargs)

    with smoke_config.metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4, f"El CSV tiene filas duplicadas: {len(rows)}."
    assert len({(row["epoch"], row["split"]) for row in rows}) == 4


def test_early_stopping_when_val_l1_plateaus(
    smoke_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con L1 de validación plana y paciencia 1, el bucle debe detenerse en la época 1."""
    monkeypatch.setattr(
        "src.gan.train.make_eval_step",
        lambda generator: lambda input_image, target_image: {
            "l1": tf.constant(0.5),
            "psnr": tf.constant(10.0),
            "ssim": tf.constant(0.5),
        },
    )
    smoke_config.train["epochs"] = 5
    smoke_config.train["early_stopping_patience"] = 1
    smoke_config.train["max_steps"] = 1

    train(
        smoke_config,
        generator=build_smoke_generator(image_size=smoke_config.image_size),
        discriminator=build_smoke_discriminator(),
        generator_loss_fn=smoke_generator_loss,
        discriminator_loss_fn=smoke_discriminator_loss,
    )

    with smoke_config.metrics_path.open(newline="", encoding="utf-8") as handle:
        epochs = {int(row["epoch"]) for row in csv.DictReader(handle)}
    assert epochs == {0, 1}, f"Early stopping no se activó a tiempo: {sorted(epochs)}."


def test_train_step_updates_discriminator_before_generator() -> None:
    """El orden de `apply_gradients` debe ser D y después G, con dos optimizadores."""
    call_order: list[str] = []

    class RecordingOptimizer:
        """Envuelve un Adam real y registra cada llamada sin alterar el gradiente."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        def apply_gradients(self, gradients_and_variables: object) -> object:
            call_order.append(self.name)
            return self.optimizer.apply_gradients(list(gradients_and_variables))

    generator = build_smoke_generator(image_size=64)
    discriminator = build_smoke_discriminator()
    step = make_train_step(
        generator,
        discriminator,
        RecordingOptimizer("G"),
        RecordingOptimizer("D"),
        smoke_generator_loss,
        smoke_discriminator_loss,
    )
    input_image = tf.random.uniform((2, 64, 64, 3), minval=-1.0, maxval=1.0, seed=1)
    target_image = tf.random.uniform((2, 64, 64, 3), minval=-1.0, maxval=1.0, seed=2)

    step(input_image, target_image)

    # `tf.function` puede trazar más de una vez; en cada traza el orden debe ser D→G.
    assert call_order, "No se llamó a apply_gradients."
    assert len(call_order) % 2 == 0, f"Número impar de actualizaciones: {call_order}."
    assert call_order[::2] == ["D"] * (
        len(call_order) // 2
    ), f"El discriminador no se actualizó primero: {call_order}."
    assert call_order[1::2] == ["G"] * (
        len(call_order) // 2
    ), f"El generador no se actualizó después: {call_order}."


def test_ablation_without_gan_only_trains_generator(smoke_config: Config) -> None:
    """Con `use_gan: false` el discriminador no debe recibir actualizaciones."""
    smoke_config.train["use_gan"] = False
    generator = build_smoke_generator(image_size=smoke_config.image_size)
    discriminator = build_smoke_discriminator()
    before_d = [variable.numpy().copy() for variable in discriminator.trainable_variables]

    train(
        smoke_config,
        generator=generator,
        discriminator=discriminator,
        generator_loss_fn=smoke_generator_loss,
        discriminator_loss_fn=smoke_discriminator_loss,
        max_steps=2,
    )

    assert not _weights_changed(before_d, discriminator), "La ablación sin GAN actualizó D."
