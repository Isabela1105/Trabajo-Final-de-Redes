"""Construye el dataset de pares (degradada -> limpia) desde las fotos de data/raw/.

Uso (desde la raíz del repositorio):
    python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42

Resultado:
    data/processed/{train,val,test}/{input,target}/<foto>_v<N>.png
    data/manifest.csv  (una fila por par, con split, degradaciones y semilla)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Permite ejecutar `python scripts/build_dataset.py` desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gan.data.dataset import (  # noqa: E402
    IMAGE_SIZE,
    SPLITS,
    check_no_leakage,
    list_sources,
    prepare_image,
    split_sources,
)
from src.gan.data.degradations import apply_random_degradations  # noqa: E402

MANIFEST_COLUMNS = [
    "id",
    "source_id",
    "source_path",
    "split",
    "variant",
    "input_path",
    "target_path",
    "degradations",
    "seed",
]


def pair_seed(base_seed: int, source_index: int, variant: int) -> int:
    """Semilla única de cada par: la misma foto y variante siempre dan el mismo daño."""
    return base_seed * 1_000_000 + source_index * 1_000 + variant


def build(
    raw_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    variants: int = 3,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, int]:
    """Construye train/val/test y el manifiesto. Devuelve el número de pares por split."""
    if not 1 <= variants <= 999:
        raise ValueError("--variants debe estar entre 1 y 999")
    sources = list_sources(raw_dir)
    if not sources:
        raise FileNotFoundError(f"No hay imágenes en {raw_dir}")
    ids = [p.stem for p in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay fotos con el mismo nombre y distinta extensión en data/raw")
    index_of = {sid: i for i, sid in enumerate(ids)}  # ids ya vienen ordenados
    path_of = dict(zip(ids, sources, strict=True))

    # 1) Dividir PRIMERO por foto fuente; 2) generar las variantes DESPUÉS.
    splits = split_sources(ids, ratios=ratios, seed=seed)
    check_no_leakage(splits)

    # Borrar lo que haya quedado de una corrida anterior (solo train/val/test).
    for split in SPLITS:
        shutil.rmtree(out_dir / split, ignore_errors=True)
        (out_dir / split / "input").mkdir(parents=True)
        (out_dir / split / "target").mkdir(parents=True)

    rows = []
    for split in SPLITS:
        for sid in tqdm(splits[split], desc=split):
            with Image.open(path_of[sid]) as img:
                target = prepare_image(img, IMAGE_SIZE)  # recorte centrado 256x256
            for v in range(variants):
                s = pair_seed(seed, index_of[sid], v)
                degraded, names = apply_random_degradations(target, np.random.default_rng(s))
                name = f"{sid}_v{v}.png"
                input_path = out_dir / split / "input" / name
                target_path = out_dir / split / "target" / name
                degraded.save(input_path)
                target.save(target_path)
                rows.append(
                    {
                        "id": f"{sid}_v{v}",
                        "source_id": sid,
                        "source_path": path_of[sid].as_posix(),
                        "split": split,
                        "variant": v,
                        "input_path": input_path.as_posix(),
                        "target_path": target_path.as_posix(),
                        "degradations": ";".join(names),
                        "seed": s,
                    }
                )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {split: sum(r["split"] == split for r in rows) for split in SPLITS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = build(args.raw, args.out, args.manifest, args.variants, args.seed)
    print(f"Pares por split: {counts} (total {sum(counts.values())})")
    print(f"Manifiesto: {args.manifest}")


if __name__ == "__main__":
    main()
