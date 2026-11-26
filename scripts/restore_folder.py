"""Restaura todas las imágenes de una carpeta y arma una cuadrícula antes/después.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Pensado para las 12 fotos antiguas reales (`real.zip`), que no tienen versión limpia: no
hay métricas posibles, solo inspección visual. Esas fotos tienen licencia sin verificar, así
que la salida va a `outputs/` (ignorada por git) y NO se sube al repositorio.

Uso:
    python scripts/restore_folder.py --checkpoint outputs/base/checkpoints \
        --input data/real --out outputs/real_restauradas
    # con la ablación al lado:
    python scripts/restore_folder.py --checkpoint outputs/base/checkpoints \
        --compare "U-Net solo L1=outputs/ablation/checkpoints" \
        --input data/real --out outputs/real_restauradas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan.data.dataset import to_rgb  # noqa: E402
from src.gan.infer import load_generator, restore_image  # noqa: E402

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def restore_folder(
    models: dict[str, str | Path],
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    columns: int = 2,
    quitar_puntos: bool = True,
    blanco_y_negro: bool = False,
) -> Path:
    """Restaura cada imagen con cada modelo y guarda PNGs y `comparacion.png`."""
    files = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No hay imágenes en {input_dir}")
    out = Path(out_dir)
    loaded = {name: load_generator(path) for name, path in models.items()}

    rows: list[list[Image.Image]] = []
    for path in files:
        with Image.open(path) as image:
            original = to_rgb(image)
        row = [original]
        for name, (generator, config) in loaded.items():
            restored = restore_image(
                generator,
                original,
                config.image_size,
                quitar_puntos=quitar_puntos,
                blanco_y_negro=blanco_y_negro,
            )
            target = out / name.replace(" ", "_").replace("/", "-")
            target.mkdir(parents=True, exist_ok=True)
            restored.save(target / f"{path.stem}.png")
            row.append(restored)
        rows.append(row)

    titles = ["Original", *loaded]
    per_row = len(titles)
    grid_rows = -(-len(rows) // columns)
    fig, axes = plt.subplots(
        grid_rows,
        per_row * columns,
        figsize=(2.4 * per_row * columns, 2.9 * grid_rows),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for index, (path, row) in enumerate(zip(files, rows, strict=True)):
        r, c0 = divmod(index, columns)
        for k, image in enumerate(row):
            ax = axes[r][c0 * per_row + k]
            ax.imshow(image)
            ax.set_title(f"{path.stem} · {titles[k]}" if k == 0 else titles[k], fontsize=7)
    fig.tight_layout()
    grid = out / "comparacion.png"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return grid


def _parse_model(spec: str) -> tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("Use 'Nombre=ruta'")
    return name.strip(), path.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restaura una carpeta de fotos.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint de Pix2Pix.")
    parser.add_argument("--compare", action="append", type=_parse_model, default=[])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="outputs/real_restauradas")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--sin-limpieza", action="store_true", help="No quitar el polvo.")
    parser.add_argument(
        "--blanco-y-negro", action="store_true", help="Fotos sin color: no inventar color."
    )
    args = parser.parse_args(argv)
    models = {"Pix2Pix": args.checkpoint, **dict(args.compare)}
    print(
        restore_folder(
            models,
            args.input,
            args.out,
            columns=args.columns,
            quitar_puntos=not args.sin_limpieza,
            blanco_y_negro=args.blanco_y_negro,
        )
    )


if __name__ == "__main__":
    main()
