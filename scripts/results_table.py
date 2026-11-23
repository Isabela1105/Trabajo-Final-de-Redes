"""Genera la tabla de resultados desde `summary.csv` y la inserta en README e informe.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Regla del informe: ninguna cifra se escribe a mano. Este script lee el `summary.csv` que
deja `python -m src.gan.evaluate` y reemplaza el bloque entre los marcadores

    <!-- RESULTADOS:INICIO -->
    <!-- RESULTADOS:FIN -->

de cada archivo indicado con `--update`.

Uso:
    python scripts/results_table.py --summary outputs/base/eval/summary.csv \
        --update README.md report/informe.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

START = "<!-- RESULTADOS:INICIO -->"
END = "<!-- RESULTADOS:FIN -->"


def _fmt(row: dict[str, str], metric: str, digits: int) -> str:
    return f"{float(row[f'{metric}_mean']):.{digits}f} ± {float(row[f'{metric}_std']):.{digits}f}"


def render_table(summary_csv: str | Path, *, source: str | None = None) -> str:
    """Tabla Markdown con media ± desviación estándar por modelo."""
    with Path(summary_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{summary_csv} está vacío")
    has_fid = all(row.get("fid") for row in rows)
    header = "| Modelo | n | PSNR (dB) ↑ | SSIM ↑ | L1 ↓ |" + (" FID ↓ |" if has_fid else "")
    sep = "|---|---|---|---|---|" + ("---|" if has_fid else "")
    lines = [header, sep]
    best_psnr = max(float(r["psnr_mean"]) for r in rows)
    for row in rows:
        name = row["model"]
        if float(row["psnr_mean"]) == best_psnr:
            name = f"**{name}**"
        line = (
            f"| {name} | {row['n']} | {_fmt(row, 'psnr', 2)} | {_fmt(row, 'ssim', 4)} "
            f"| {_fmt(row, 'l1', 4)} |"
        )
        if has_fid:
            line += f" {float(row['fid']):.1f} |"
        lines.append(line)
    note = source or Path(summary_csv).as_posix()
    lines.append("")
    lines.append(
        f"_Split `test`, métricas en [0, 1] por imagen completa (media ± std). Fuente: `{note}`._"
    )
    return "\n".join(lines)


def update_file(path: str | Path, table: str) -> bool:
    """Reemplaza el bloque marcado. Devuelve False si el archivo no tiene marcadores."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        return False
    new = pattern.sub(lambda _: f"{START}\n{table}\n{END}", text)
    path.write_text(new, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", default="outputs/base/eval/summary.csv")
    parser.add_argument("--update", nargs="*", default=[], help="Archivos con marcadores.")
    parser.add_argument("--source", default=None, help="Texto de la fuente en la nota.")
    args = parser.parse_args(argv)

    table = render_table(args.summary, source=args.source)
    print(table)
    for target in args.update:
        status = "actualizado" if update_file(target, table) else "SIN marcadores, no se tocó"
        print(f"[results] {target}: {status}")


if __name__ == "__main__":
    main()
