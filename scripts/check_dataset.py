"""Valida el dataset construido: pares completos, sin fuga, RGB 256x256 y sin archivos corruptos.

Uso (desde la raíz del repositorio):
    python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv

Termina con código 0 si todo está bien y con código 1 si encuentra algún problema.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

# Permite ejecutar `python scripts/check_dataset.py` desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan.data.dataset import IMAGE_SIZE, SPLITS  # noqa: E402

REQUIRED_COLUMNS = {"id", "source_id", "split", "input_path", "target_path", "seed"}


def _check_image(path: Path) -> str | None:
    """Devuelve un texto con el problema de la imagen, o None si está bien."""
    if not path.exists():
        return f"no existe: {path}"
    try:
        with Image.open(path) as img:
            img.load()  # fuerza a leer todo el archivo: detecta archivos corruptos
            if img.mode != "RGB":
                return f"no es RGB ({img.mode}): {path}"
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                return f"tamaño {img.size} en vez de {IMAGE_SIZE}x{IMAGE_SIZE}: {path}"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es un archivo corrupto
        return f"archivo corrupto ({type(exc).__name__}): {path}"
    return None


def run_checks(root: Path, manifest_path: Path) -> list[str]:
    """Revisa el dataset y devuelve la lista de problemas (vacía si todo está bien)."""
    if not manifest_path.exists():
        return [f"no existe el manifiesto: {manifest_path}"]
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            return [f"al manifiesto le faltan columnas: {sorted(missing)}"]
        rows = list(reader)
    if not rows:
        return ["el manifiesto está vacío"]

    errors: list[str] = []

    # 1) Ninguna foto fuente en más de un split (se compara por ID de fuente, no por archivo).
    splits_of: dict[str, set[str]] = {}
    for r in rows:
        splits_of.setdefault(r["source_id"], set()).add(r["split"])
    for sid, where in sorted(splits_of.items()):
        if len(where) > 1:
            errors.append(f"FUGA: la fuente {sid} está en {sorted(where)}")

    # 2) Cada fila: split válido, imágenes existentes, legibles, RGB y 256x256.
    for r in rows:
        if r["split"] not in SPLITS:
            errors.append(f"split desconocido '{r['split']}' en {r['id']}")
        for key in ("input_path", "target_path"):
            problem = _check_image(Path(r[key]))
            if problem:
                errors.append(problem)

    # 3) Pares completos: mismos nombres en input/ y target/, y nada fuera del manifiesto.
    in_manifest = {Path(r["input_path"]).as_posix() for r in rows}
    for split in SPLITS:
        inputs = {p.name for p in (root / split / "input").glob("*.png")}
        targets = {p.name for p in (root / split / "target").glob("*.png")}
        for name in sorted(inputs - targets):
            errors.append(f"{split}: {name} tiene input pero no target")
        for name in sorted(targets - inputs):
            errors.append(f"{split}: {name} tiene target pero no input")
        for p in (root / split / "input").glob("*.png"):
            if p.as_posix() not in in_manifest:
                errors.append(f"{split}: {p.name} no está en el manifiesto")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()

    errors = run_checks(args.root, args.manifest)
    if errors:
        print(f"Se encontraron {len(errors)} problemas:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... y {len(errors) - 20} más")
        sys.exit(1)
    print("Dataset OK: pares completos, sin fuga, RGB 256x256 y sin archivos corruptos.")


if __name__ == "__main__":
    main()
