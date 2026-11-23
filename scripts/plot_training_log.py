"""Dibuja las curvas de entrenamiento a partir de un CSV.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Acepta dos formatos:
- `metrics.csv` de `src/gan/train.py` (columnas `epoch, split, loss_g, loss_d, l1, psnr, ssim`).
- El registro transcrito de Colab en `docs/resultados/colab_base_20ep_log.csv`
  (columnas `epoch, loss_d_train, loss_g_train, l1_val_m11`).

Uso:
    python scripts/plot_training_log.py docs/resultados/colab_base_20ep_log.csv \
        docs/img/curvas_entrenamiento.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _series(rows: list[dict[str, str]]) -> dict[str, tuple[list[int], list[float]]]:
    """Normaliza ambos formatos a {nombre: (épocas, valores)}."""

    def pick(column: str, split: str | None = None) -> tuple[list[int], list[float]]:
        selected = [
            r
            for r in rows
            if r.get(column) not in (None, "") and (split is None or r["split"] == split)
        ]
        return [int(float(r["epoch"])) for r in selected], [float(r[column]) for r in selected]

    if "split" in rows[0]:
        return {
            "Pérdida del generador (train)": pick("loss_g", "train"),
            "Pérdida del discriminador (train)": pick("loss_d", "train"),
            "L1 de validación (rango [-1, 1])": pick("l1", "val"),
        }
    return {
        "Pérdida del generador (train)": pick("loss_g_train"),
        "Pérdida del discriminador (train)": pick("loss_d_train"),
        "L1 de validación (rango [-1, 1])": pick("l1_val_m11"),
    }


def plot(csv_path: str | Path, out_path: str | Path, title: str) -> Path:
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    series = _series(rows)
    colors = ("#8a4b16", "#3b6ea5", "#1f7a4d")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, color, (name, (epochs, values)) in zip(axes, colors, series.items(), strict=True):
        ax.plot(epochs, values, marker="o", ms=3.5, lw=1.6, color=color)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if epochs:
            step = max(1, len(epochs) // 10)
            ax.set_xticks(range(min(epochs), max(epochs) + 1, step))
    epochs, values = series["L1 de validación (rango [-1, 1])"]
    if values:
        i = values.index(min(values))
        axes[2].annotate(
            f"mín. {values[i]:.4f} (época {epochs[i]})",
            (epochs[i], values[i]),
            xytext=(0.05, 0.9),
            textcoords="axes fraction",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Curvas de entrenamiento desde un CSV.")
    parser.add_argument("csv")
    parser.add_argument("out")
    parser.add_argument(
        "--title", default="Corrida base en Colab (GPU, 20 épocas, configs/base.yaml)"
    )
    args = parser.parse_args(argv)
    print(plot(args.csv, args.out, args.title))


if __name__ == "__main__":
    main()
