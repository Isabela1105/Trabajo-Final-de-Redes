"""Genera las estadísticas (Markdown) y la galería de ejemplos del dataset.

Uso (desde la raíz del repositorio):
    python scripts/dataset_report.py

Salidas por defecto:
    docs/dataset_stats.md       tablas con las cifras del manifiesto
    docs/img/dataset_grid.png   pares input/target de ejemplo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar `python scripts/dataset_report.py` desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan.data.stats import load_manifest, make_grid, make_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--sources", type=Path, default=Path("data/sources.csv"))
    parser.add_argument("--grid", type=Path, default=Path("docs/img/dataset_grid.png"))
    parser.add_argument("--stats", type=Path, default=Path("docs/dataset_stats.md"))
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    report = make_report(args.manifest, args.sources)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(report, encoding="utf-8")
    print(report)

    shown = make_grid(load_manifest(args.manifest), args.grid, n=args.examples)
    print(f"Galería: {args.grid} (pares: {', '.join(shown)})")
    print(f"Estadísticas: {args.stats}")


if __name__ == "__main__":
    main()
