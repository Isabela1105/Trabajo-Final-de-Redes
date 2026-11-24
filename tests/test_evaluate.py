"""Pruebas de la evaluación del test set: CSVs, baselines y figuras.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.gan.evaluate import (
    INPUT_BASELINE,
    MAIN_MODEL,
    MEDIAN_BASELINE,
    by_degradation,
    evaluate_predictors,
    identity_baseline,
    main,
    median_filter,
    summarize,
    worst_cases,
)
from tests.conftest import write_pairs


def test_median_filter_removes_isolated_spike() -> None:
    batch = np.zeros((1, 5, 5, 3), np.float32)
    batch[0, 2, 2] = 1.0
    assert median_filter(batch)[0, 2, 2].max() == 0.0
    assert median_filter(batch).shape == batch.shape


def test_median_filter_rejects_even_size() -> None:
    with pytest.raises(ValueError):
        median_filter(np.zeros((1, 4, 4, 3), np.float32), size=2)


def test_identity_is_perfect_when_input_equals_target(tmp_path: Path) -> None:
    write_pairs(tmp_path, size=32, n_pairs=3)
    for target in (tmp_path / "val" / "target").glob("*.png"):
        shutil.copy(target, tmp_path / "val" / "input" / target.name)

    result = evaluate_predictors(
        {INPUT_BASELINE: identity_baseline, MEDIAN_BASELINE: median_filter}, tmp_path / "val", 32
    )

    summary = {row["model"]: row for row in result.summary}
    assert summary[INPUT_BASELINE]["n"] == 3
    assert summary[INPUT_BASELINE]["ssim_mean"] == pytest.approx(1.0, abs=1e-5)
    assert summary[INPUT_BASELINE]["l1_mean"] == pytest.approx(0.0, abs=1e-7)
    assert summary[INPUT_BASELINE]["psnr_mean"] > summary[MEDIAN_BASELINE]["psnr_mean"]
    assert len(result.rows) == 6  # 3 imágenes × 2 modelos


def test_summary_and_breakdowns() -> None:
    rows = [
        {
            "image": "a",
            "model": "M",
            "degradations": "blur;noise",
            "psnr": 20,
            "ssim": 0.5,
            "l1": 0.1,
        },
        {"image": "b", "model": "M", "degradations": "blur", "psnr": 10, "ssim": 0.3, "l1": 0.3},
        {
            "image": "a",
            "model": "N",
            "degradations": "blur;noise",
            "psnr": 5,
            "ssim": 0.1,
            "l1": 0.5,
        },
    ]
    summary = summarize(rows)
    assert [s["model"] for s in summary] == ["M", "N"]
    assert summary[0]["psnr_mean"] == pytest.approx(15)
    assert summary[0]["psnr_std"] == pytest.approx(5)
    breakdown = {b["degradation"]: b for b in by_degradation(rows, "M")}
    assert breakdown["blur"]["n"] == 2 and breakdown["noise"]["n"] == 1
    assert worst_cases(rows, "M", k=1) == ["b"]


def test_cli_end_to_end(trained_checkpoint: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    write_pairs(data, size=64, n_pairs=3, seed=7)
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["data"].update(
        {
            "train_dir": str(data / "train"),
            "val_dir": str(data / "val"),
            "test_dir": str(data / "val"),
        }
    )
    raw["output"]["dir"] = str(tmp_path / "out")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    main(["--config", str(config_path), "--checkpoint", str(trained_checkpoint), "--manifest", ""])

    out = tmp_path / "out" / "eval"
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        models = [row["model"] for row in csv.DictReader(handle)]
    assert models == [INPUT_BASELINE, MEDIAN_BASELINE, MAIN_MODEL]
    with (out / "per_image.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 9  # una fila por imagen y modelo
    assert (out / "results_grid.png").stat().st_size > 0
    assert (out / "failures.png").stat().st_size > 0


def test_main_model_name_is_not_shadowed_by_extra_models(
    trained_checkpoint: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    write_pairs(data, size=64, n_pairs=2, seed=3)
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
    raw["data"].update({"train_dir": str(data / "train"), "test_dir": str(data / "val")})
    raw["data"]["val_dir"] = str(data / "val")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "eval"
    main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(trained_checkpoint),
            "--model",
            f"Otro={trained_checkpoint}",
            "--name",
            "Principal",
            "--manifest",
            "",
            "--no-baselines",
            "--out",
            str(out),
        ]
    )
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        assert [row["model"] for row in csv.DictReader(handle)] == ["Otro", "Principal"]
