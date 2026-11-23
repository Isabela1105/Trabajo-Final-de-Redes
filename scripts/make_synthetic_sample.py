"""Genera una muestra sintética reproducible para CI y para probar el pipeline sin datos reales.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01) — cubre el PR-F3 del plan, que seguía
pendiente y hacía que el CI se saltara el smoke training.

Crea `out/{train,val,test}/{input,target}/sample_XX.png`. El target es una imagen limpia con
formas y degradado de color; el input es la misma imagen con las degradaciones del proyecto
(`src/gan/data/degradations.py`). Misma semilla = mismos píxeles.

Uso:
    python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Permite `python scripts/make_synthetic_sample.py` desde la raíz sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLITS = (("train", 0.5), ("val", 0.25), ("test", 0.25))


def clean_image(rng: np.random.Generator, size: int) -> Image.Image:
    """Imagen limpia: degradado de color + círculos y rectángulos al azar."""
    y, x = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = (x * rng.random() + y * (1 - rng.random()))[..., None] / 2
    base = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(2, 6))):
        x0, y0 = rng.integers(0, size, 2)
        w, h = rng.integers(size // 8, size // 2, 2)
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x0 + w, y0 + h], fill=color)
        else:
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
    return image


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Aplica un subconjunto aleatorio de las degradaciones reales del dataset."""
    from src.gan.data.degradations import apply_random_degradations

    degraded, _ = apply_random_degradations(image, rng)
    return degraded


def make_sample(out: str | Path, n: int = 8, size: int = 64, seed: int = 42) -> dict[str, int]:
    """Escribe `n` pares por split (mínimo 1) y devuelve el conteo por split."""
    if n < 1 or size < 8:
        raise ValueError("n debe ser >= 1 y size >= 8")
    out = Path(out)
    rng = np.random.default_rng(seed)
    counts = {}
    for split, share in SPLITS:
        count = max(1, round(n * share)) if split != "train" else max(2, round(n * share))
        for kind in ("input", "target"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            target = clean_image(rng, size)
            source = degrade(target, rng)
            if source.size != target.size:
                source = source.resize(target.size)
            name = f"sample_{index:02d}.png"
            target.save(out / split / "target" / name)
            source.convert("RGB").save(out / split / "input" / name)
        counts[split] = count
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Muestra sintética para CI.")
    parser.add_argument("--out", default="data/sample")
    parser.add_argument("--n", type=int, default=8, help="Pares totales aproximados.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    counts = make_sample(args.out, args.n, args.size, args.seed)
    print(f"[sample] {counts} pares en {args.out}")


if __name__ == "__main__":
    main()
