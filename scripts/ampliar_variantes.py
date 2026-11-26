"""Genera más variantes dañadas de las fotos de entrenamiento, sin fotos nuevas.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Repite exactamente el proceso de `scripts/build_dataset.py`: toma el objetivo limpio de
256×256 de cada foto fuente de `train` y le aplica `apply_random_degradations` con la semilla
`pair_seed(seed, índice_de_la_fuente, variante)`. Las variantes que ya existían salen idénticas
byte a byte (se comprueba y se reporta), así que las nuevas siguen el mismo proceso. Validación
y prueba no se tocan.

    python scripts/ampliar_variantes.py --variants 10 --out data/train_x10/train

Experimento documentado en `docs/RESULTADOS.md` §7.5: con 10 variantes por foto (1 220 pares)
el modelo no mejoró en validación y no se adoptó.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_dataset import pair_seed
from src.gan.data.degradations import apply_random_degradations


def _source_indices(rows: list[dict], seed: int) -> dict[str, int]:
    """Recupera el índice de cada fuente a partir de la semilla guardada en el manifiesto."""
    index_of: dict[str, int] = {}
    for row in rows:
        index = (int(row["seed"]) - seed * 1_000_000) // 1_000
        if pair_seed(seed, index, int(row["variant"])) != int(row["seed"]):
            raise ValueError(f"La semilla de {row['id']} no sigue pair_seed con seed={seed}")
        index_of.setdefault(row["source_id"], index)
    return index_of


def ampliar(manifest: Path, out_dir: Path, variants: int, seed: int = 42) -> dict[str, int]:
    """Escribe `variants` pares por foto de entrenamiento en `out_dir` y su manifiesto."""
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index_of = _source_indices(rows, seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    others = {row["source_id"] for row in rows if row["split"] != "train"}
    sources = sorted({row["source_id"] for row in train_rows})
    if set(sources) & others:
        raise ValueError("Hay fuentes de train también en val/test: fuga de datos")
    existing = {(row["source_id"], int(row["variant"])): row for row in train_rows}

    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "target").mkdir(parents=True, exist_ok=True)
    written, identical, checked = [], 0, 0
    for sid in sources:
        first = existing[(sid, 0)]
        target = Image.open(first["target_path"]).convert("RGB")
        for variant in range(variants):
            s = pair_seed(seed, index_of[sid], variant)
            degraded, names = apply_random_degradations(target, np.random.default_rng(s))
            name = f"{sid}_v{variant}.png"
            if (sid, variant) in existing:
                original = Image.open(existing[(sid, variant)]["input_path"]).convert("RGB")
                checked += 1
                identical += int(np.array_equal(np.asarray(original), np.asarray(degraded)))
            degraded.save(out_dir / "input" / name)
            target.save(out_dir / "target" / name)
            written.append(
                {
                    "id": name[:-4],
                    "source_id": sid,
                    "variant": variant,
                    "degradations": ";".join(names),
                    "seed": s,
                }
            )
    with (out_dir.parent / f"manifest_{out_dir.name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(written[0]))
        writer.writeheader()
        writer.writerows(written)
    return {
        "fuentes": len(sources),
        "pares": len(written),
        "comprobadas": checked,
        "identicas": identical,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out", type=Path, required=True, help="Carpeta train nueva.")
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    counts = ampliar(args.manifest, args.out, args.variants, args.seed)
    print(
        f"{counts['fuentes']} fuentes → {counts['pares']} pares; variantes existentes "
        f"idénticas: {counts['identicas']} de {counts['comprobadas']}"
    )
    if counts["identicas"] != counts["comprobadas"]:
        raise SystemExit("Las variantes regeneradas no coinciden con el dataset original")


if __name__ == "__main__":
    main()
