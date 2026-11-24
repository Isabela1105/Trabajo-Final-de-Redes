"""Análisis de errores desde `per_image.csv`: dónde ayuda el modelo y dónde empeora.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Agrupa las imágenes de prueba por el PSNR de la entrada (cuánto daño tenía) y por tipo de
degradación, y reporta la ganancia media del modelo frente a la entrada y en cuántas imágenes
mejora. Imprime tablas en Markdown listas para pegar en `docs/RESULTADOS.md`.

    python scripts/error_analysis.py --per-image outputs/base/eval/per_image.csv \
        --model "Pix2Pix (λ = 100)"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

INPUT = "Entrada (sin restaurar)"
BINS = ((0.0, 18.0), (18.0, 20.0), (20.0, 22.0), (22.0, 99.0))


def load(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_image: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_image[row["image"]][row["model"]] = row
    return by_image


def gains(by_image: dict, model: str) -> list[tuple[float, float, float, list[str]]]:
    """(PSNR entrada, ganancia PSNR, ganancia SSIM, degradaciones) por imagen."""
    out = []
    for rows in by_image.values():
        if INPUT not in rows or model not in rows:
            continue
        base, pred = rows[INPUT], rows[model]
        out.append(
            (
                float(base["psnr"]),
                float(pred["psnr"]) - float(base["psnr"]),
                float(pred["ssim"]) - float(base["ssim"]),
                [d for d in base["degradations"].split(";") if d],
            )
        )
    return out


def by_damage(data: list[tuple[float, float, float, list[str]]]) -> str:
    lines = [
        "| PSNR de la entrada | n | Ganancia PSNR (dB) | Ganancia SSIM | Mejora en |",
        "|---|---|---|---|---|",
    ]
    for low, high in BINS:
        group = [d for d in data if low <= d[0] < high]
        if not group:
            continue
        label = (
            f"< {high:.0f} dB"
            if low == 0
            else (f"≥ {low:.0f} dB" if high > 90 else f"{low:.0f}–{high:.0f} dB")
        )
        better = sum(d[1] > 0 for d in group)
        lines.append(
            f"| {label} | {len(group)} | {mean(d[1] for d in group):+.2f} | "
            f"{mean(d[2] for d in group):+.3f} | {better} de {len(group)} |"
        )
    return "\n".join(lines)


def by_degradation(data: list[tuple[float, float, float, list[str]]]) -> str:
    groups: dict[str, list] = defaultdict(list)
    for item in data:
        for name in item[3]:
            groups[name].append(item)
    lines = ["| Degradación | n | Ganancia PSNR (dB) | Mejora en |", "|---|---|---|---|"]
    for name, group in sorted(groups.items(), key=lambda kv: -mean(d[1] for d in kv[1])):
        better = sum(d[1] > 0 for d in group)
        lines.append(
            f"| {name} | {len(group)} | {mean(d[1] for d in group):+.2f} | {better} de {len(group)} |"
        )
    return "\n".join(lines)


def correlation(data: list[tuple[float, float, float, list[str]]]) -> float:
    xs, ys = [d[0] for d in data], [d[1] for d in data]
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--model", default="Pix2Pix (λ = 100)")
    args = parser.parse_args(argv)
    data = gains(load(args.per_image), args.model)
    if not data:
        raise SystemExit(f"No hay filas de '{args.model}' en {args.per_image}")
    print(f"## {args.model}: n = {len(data)}, mejora en {sum(d[1] > 0 for d in data)}")
    print(f"Correlación PSNR de la entrada vs ganancia: {correlation(data):+.2f}\n")
    print(by_damage(data) + "\n")
    print(by_degradation(data))


if __name__ == "__main__":
    main()
