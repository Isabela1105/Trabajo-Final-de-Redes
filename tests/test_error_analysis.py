"""Pruebas de scripts/error_analysis.py.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.error_analysis import INPUT, by_damage, by_degradation, correlation, gains, load


def _csv(path: Path) -> Path:
    rows = [
        ("a", INPUT, "blur", 17.0, 0.40),
        ("a", "M", "blur", 20.0, 0.50),
        ("b", INPUT, "dust;jpeg", 25.0, 0.80),
        ("b", "M", "dust;jpeg", 23.0, 0.75),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "model", "degradations", "psnr", "ssim", "l1"])
        for image, model, deg, psnr, ssim in rows:
            writer.writerow([image, model, deg, psnr, ssim, 0.1])
    return path


def test_gains_and_tables(tmp_path: Path) -> None:
    data = gains(load(_csv(tmp_path / "p.csv")), "M")
    assert sorted(round(d[1], 2) for d in data) == [-2.0, 3.0]
    assert correlation(data) < 0
    damage = by_damage(data)
    assert "< 18 dB | 1 | +3.00" in damage and "≥ 22 dB | 1 | -2.00" in damage
    assert "| dust | 1 | -2.00 | 0 de 1 |" in by_degradation(data)
