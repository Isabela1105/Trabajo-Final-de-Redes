"""Pruebas del generador de la tabla de resultados.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)
"""

from __future__ import annotations

from pathlib import Path

from scripts.results_table import END, START, main, render_table, update_file
from src.gan.evaluate import write_csv

ROWS = [
    {"model": "Entrada", "n": 2, "psnr_mean": 18.0, "psnr_std": 1.0, "ssim_mean": 0.5,
     "ssim_std": 0.1, "l1_mean": 0.1, "l1_std": 0.01},
    {"model": "Pix2Pix", "n": 2, "psnr_mean": 21.5, "psnr_std": 0.5, "ssim_mean": 0.6,
     "ssim_std": 0.05, "l1_mean": 0.07, "l1_std": 0.02},
]  # fmt: skip


def test_table_marks_best_model(tmp_path: Path) -> None:
    summary = write_csv(tmp_path / "summary.csv", ROWS)
    table = render_table(summary)
    assert "| **Pix2Pix** | 2 | 21.50 ± 0.50 | 0.6000 ± 0.0500 | 0.0700 ± 0.0200 |" in table
    assert "| Entrada |" in table


def test_update_only_touches_marked_block(tmp_path: Path) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(f"antes\n{START}\nviejo\n{END}\ndespués\n", encoding="utf-8")
    assert update_file(doc, "NUEVA")
    assert doc.read_text(encoding="utf-8") == f"antes\n{START}\nNUEVA\n{END}\ndespués\n"
    plain = tmp_path / "otro.md"
    plain.write_text("sin marcadores", encoding="utf-8")
    assert not update_file(plain, "NUEVA")


def test_cli(tmp_path: Path) -> None:
    summary = write_csv(tmp_path / "summary.csv", ROWS)
    doc = tmp_path / "informe.md"
    doc.write_text(f"{START}\n{END}\n", encoding="utf-8")
    main(["--summary", str(summary), "--update", str(doc)])
    assert "Pix2Pix" in doc.read_text(encoding="utf-8")
