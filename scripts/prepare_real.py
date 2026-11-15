"""Prepara las fotos antiguas REALES para evaluación cualitativa (no se usan para métricas).

Lee las fotos descargadas de `data/raw/real_originales/` (esa carpeta no se sube a git) y deja
copias listas en `data/real/real_001.jpg`, `real_002.jpg`... (RGB, lado mayor <= 1024 px).
También crea/actualiza `data/real_sources.csv`, donde se anota el autor, la licencia y el
enlace de cada foto. Se puede correr varias veces: las fotos ya registradas no se repiten.

Uso (desde la raíz del repositorio):
    python scripts/prepare_real.py
    python scripts/prepare_real.py --licencia "Dominio público (Library of Congress)"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image

# Permite ejecutar `python scripts/prepare_real.py` desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan.data.dataset import IMAGE_EXTENSIONS, to_rgb  # noqa: E402

COLUMNS = ["archivo", "original", "descripcion", "anio", "autor", "licencia", "url"]
PENDIENTE = "POR COMPLETAR"
MAX_SIDE = 1024


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(src: Path, out: Path, sources_path: Path, licencia: str = PENDIENTE) -> dict[str, int]:
    """Convierte las fotos nuevas de `src` y las registra. Devuelve un resumen con conteos."""
    rows = read_rows(sources_path)
    known = {r["original"] for r in rows}
    numbers = [
        int(m.group(1)) for r in rows if (m := re.fullmatch(r"real_(\d+)\.jpg", r["archivo"]))
    ]
    next_n = max(numbers, default=0) + 1

    out.mkdir(parents=True, exist_ok=True)
    new = 0
    originals = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for p in originals:
        if p.name in known:
            continue
        with Image.open(p) as img:
            rgb = to_rgb(img)  # arregla rotación EXIF y transparencia
        rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)  # solo reduce, nunca amplía
        name = f"real_{next_n:03d}.jpg"
        rgb.save(out / name, format="JPEG", quality=90)
        rows.append(
            {
                "archivo": name,
                "original": p.name,
                "descripcion": PENDIENTE,
                "anio": PENDIENTE,
                "autor": PENDIENTE,
                "licencia": licencia,
                "url": PENDIENTE,
            }
        )
        next_n += 1
        new += 1

    write_rows(sources_path, rows)
    incomplete = sum(any(v.strip() in ("", PENDIENTE) for v in r.values()) for r in rows)
    return {"nuevas": new, "total": len(rows), "incompletas": incomplete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/raw/real_originales"))
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    parser.add_argument("--sources", type=Path, default=Path("data/real_sources.csv"))
    parser.add_argument("--licencia", default=PENDIENTE, help="licencia para las fotos nuevas")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"No existe la carpeta {args.src}. Créala y copia ahí las fotos.")
    res = prepare(args.src, args.out, args.sources, args.licencia)
    print(f"Fotos nuevas: {res['nuevas']}. Total registradas: {res['total']}.")
    print(f"Filas con campos por completar: {res['incompletas']} (edita {args.sources}).")


if __name__ == "__main__":
    main()
