"""Evaluación del split de prueba: métricas por imagen, resumen, baselines y figuras.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Qué hace, en orden:
1. Lee los pares `test/input` y `test/target` (mismo nombre de archivo).
2. Calcula las predicciones de cada "modelo": las líneas base sin aprendizaje
   (`Entrada`, `Mediana 3x3`) y uno o más checkpoints del generador.
3. Mide PSNR, SSIM y L1 en [0, 1] por imagen completa (`src/gan/metrics.py`).
4. Escribe:
   - `per_image.csv`: una fila por imagen y modelo (+ degradaciones del manifiesto).
   - `summary.csv`: media ± desviación estándar por modelo.
   - `by_degradation.csv`: medias del modelo principal por tipo de degradación.
   - `results_grid.png` y `failures.png` (peores casos por PSNR, sin elegir a mano).

Uso:
    python -m src.gan.evaluate --config configs/base.yaml \
        --checkpoint outputs/base/checkpoints --split test
    # con la ablación:
    python -m src.gan.evaluate --config configs/base.yaml \
        --checkpoint outputs/base/checkpoints \
        --model "U-Net solo L1=outputs/ablation/checkpoints"
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.gan.config import REPO_ROOT, load_config, resolve_path
from src.gan.data.tf_dataset import list_pairs
from src.gan.metrics import all_metrics, to_unit_range

INPUT_BASELINE = "Entrada (sin restaurar)"
MEDIAN_BASELINE = "Filtro de mediana 3x3"
MAIN_MODEL = "Pix2Pix (λ = 100)"
METRICS = ("psnr", "ssim", "l1")

# Un "predictor" recibe un lote en [-1, 1] y devuelve otro lote en [-1, 1].
Predictor = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Líneas base sin aprendizaje
# ---------------------------------------------------------------------------
def identity_baseline(batch: np.ndarray) -> np.ndarray:
    """La foto degradada tal cual: el piso que cualquier modelo debe superar."""
    return batch


def median_filter(batch: np.ndarray, size: int = 3) -> np.ndarray:
    """Filtro de mediana por canal con borde reflejado (sin SciPy).

    Se apilan los `size × size` desplazamientos de cada píxel y se toma la mediana.
    """
    if size % 2 == 0 or size < 1:
        raise ValueError("size debe ser impar y positivo")
    pad = size // 2
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = batch.shape[1:3]
    windows = [padded[:, dy : dy + h, dx : dx + w, :] for dy in range(size) for dx in range(size)]
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.float32)


def generator_predictor(generator: tf.keras.Model) -> Predictor:
    def predict(batch: np.ndarray) -> np.ndarray:
        return tf.cast(generator(batch, training=False), tf.float32).numpy()

    return predict


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    names: list[str]
    inputs: np.ndarray  # (B, H, W, 3) en [-1, 1]
    targets: np.ndarray


def _load_png(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (size, size):
            raise ValueError(f"{path} mide {rgb.size}, se esperaba {size}x{size}")
        return np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0


def iter_batches(split_dir: str | Path, size: int, batch_size: int = 4) -> Iterator[Batch]:
    """Recorre el split en orden alfabético (reproducible) por lotes."""
    inputs, targets = list_pairs(split_dir)
    for start in range(0, len(inputs), batch_size):
        chunk = slice(start, start + batch_size)
        yield Batch(
            names=[Path(p).stem for p in inputs[chunk]],
            inputs=np.stack([_load_png(p, size) for p in inputs[chunk]]),
            targets=np.stack([_load_png(p, size) for p in targets[chunk]]),
        )


def load_degradations(manifest: str | Path | None) -> dict[str, str]:
    """`id -> "blur;noise;..."` desde `data/manifest.csv` (vacío si no existe)."""
    if not manifest or not Path(manifest).is_file():
        return {}
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row.get("degradations", "") for row in csv.DictReader(handle)}


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    rows: list[dict[str, object]]
    summary: list[dict[str, object]]
    examples: dict[str, dict[str, np.ndarray]]  # nombre -> {modelo/input/target: imagen}


def evaluate_predictors(
    predictors: dict[str, Predictor],
    split_dir: str | Path,
    size: int,
    *,
    batch_size: int = 4,
    degradations: dict[str, str] | None = None,
) -> EvaluationResult:
    """Mide cada predictor sobre todas las imágenes del split."""
    if not predictors:
        raise ValueError("No hay modelos que evaluar")
    degradations = degradations or {}
    rows: list[dict[str, object]] = []
    examples: dict[str, dict[str, np.ndarray]] = {}

    for batch in iter_batches(split_dir, size, batch_size):
        target_unit = to_unit_range(batch.targets).numpy()
        for name, image in zip(batch.names, target_unit, strict=True):
            examples.setdefault(name, {})["Objetivo"] = image
        for name, image in zip(batch.names, to_unit_range(batch.inputs).numpy(), strict=True):
            examples[name]["Entrada"] = image

        for model, predict in predictors.items():
            pred_unit = to_unit_range(predict(batch.inputs)).numpy()
            scores = all_metrics(pred_unit, target_unit)
            for i, name in enumerate(batch.names):
                examples[name][model] = pred_unit[i]
                rows.append(
                    {
                        "image": name,
                        "model": model,
                        "degradations": degradations.get(name, ""),
                        **{m: float(scores[m][i]) for m in METRICS},
                    }
                )

    return EvaluationResult(rows=rows, summary=summarize(rows), examples=examples)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Media y desviación estándar (poblacional) por modelo, en el orden de aparición."""
    order: list[str] = []
    for row in rows:
        if row["model"] not in order:
            order.append(str(row["model"]))
    summary = []
    for model in order:
        selected = [r for r in rows if r["model"] == model]
        entry: dict[str, object] = {"model": model, "n": len(selected)}
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in selected])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std())
        summary.append(entry)
    return summary


def by_degradation(rows: list[dict[str, object]], model: str) -> list[dict[str, object]]:
    """Medias del modelo por tipo de degradación (un par cuenta en cada tipo que tiene)."""
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["model"] != model:
            continue
        for kind in str(row["degradations"]).split(";"):
            if kind:
                groups.setdefault(kind, []).append(row)
    result = []
    for kind in sorted(groups):
        entry: dict[str, object] = {"degradation": kind, "n": len(groups[kind])}
        for metric in METRICS:
            entry[f"{metric}_mean"] = float(np.mean([float(r[metric]) for r in groups[kind]]))
        result.append(entry)
    return result


def add_fid(result: EvaluationResult) -> None:
    """Agrega la columna `fid` al resumen (descarga InceptionV3 la primera vez)."""
    from src.gan.metrics import fid_from_features, inception_features

    names = sorted(result.examples)
    real = inception_features(np.stack([result.examples[n]["Objetivo"] for n in names]))
    for entry in result.summary:
        model = str(entry["model"])
        column = "Entrada" if model == INPUT_BASELINE else model
        fake = inception_features(np.stack([result.examples[n][column] for n in names]))
        entry["fid"] = fid_from_features(real, fake)


def worst_cases(rows: list[dict[str, object]], model: str, k: int = 4) -> list[str]:
    """Nombres de las k imágenes con menor PSNR para ese modelo."""
    selected = sorted((r for r in rows if r["model"] == model), key=lambda r: float(r["psnr"]))
    return [str(r["image"]) for r in selected[:k]]


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_model(spec: str) -> tuple[str, str]:
    name, sep, path = spec.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"Use 'Nombre=ruta', no {spec!r}")
    return name.strip(), path.strip()


def run(
    config_path: str | Path,
    checkpoint: str | Path | None,
    *,
    split: str = "test",
    out_dir: str | Path | None = None,
    extra_models: list[tuple[str, str]] | None = None,
    batch_size: int = 4,
    manifest: str | Path | None = REPO_ROOT / "data" / "manifest.csv",
    with_baselines: bool = True,
    grid_examples: int = 4,
    fid: bool = False,
) -> dict[str, Path]:
    """Evalúa y escribe CSVs y figuras. Devuelve las rutas escritas."""
    from src.gan.infer import load_generator
    from src.gan.utils.viz import save_comparison_grid

    config = load_config(config_path, fixed_seed=False)
    split_key = f"{split}_dir"
    if split_key not in config.data:
        raise ValueError(f"El YAML no define data.{split_key}")
    if split != "test":
        print(f"[eval] AVISO: se evalúa el split {split!r}; la tabla final usa SIEMPRE 'test'.")

    predictors: dict[str, Predictor] = {}
    if with_baselines:
        predictors[INPUT_BASELINE] = identity_baseline
        predictors[MEDIAN_BASELINE] = median_filter
    for name, path in extra_models or []:
        generator, _ = load_generator(path)
        predictors[name] = generator_predictor(generator)
    main_model = None
    if checkpoint is not None:
        try:  # primero el config.yaml guardado junto al checkpoint (es el que entrenó)
            generator, _ = load_generator(checkpoint)
        except FileNotFoundError:
            generator, _ = load_generator(checkpoint, config_path)
        predictors[MAIN_MODEL] = generator_predictor(generator)
        main_model = MAIN_MODEL
    elif extra_models:
        main_model = extra_models[-1][0]

    result = evaluate_predictors(
        predictors,
        config.data[split_key],
        config.image_size,
        batch_size=batch_size,
        degradations=load_degradations(manifest),
    )

    if fid:
        add_fid(result)

    out = resolve_path(out_dir) if out_dir else config.output_dir / "eval"
    written = {
        "per_image": write_csv(out / "per_image.csv", result.rows),
        "summary": write_csv(out / "summary.csv", result.summary),
    }
    columns = ["Entrada", *(m for m in predictors if m != INPUT_BASELINE), "Objetivo"]
    names = sorted(result.examples)[:grid_examples]
    written["results_grid"] = save_comparison_grid(
        result.examples, names, columns, out / "results_grid.png"
    )
    if main_model is not None:
        written["by_degradation"] = write_csv(
            out / "by_degradation.csv", by_degradation(result.rows, main_model)
        )
        worst = worst_cases(result.rows, main_model, k=grid_examples)
        psnr_of = {r["image"]: r["psnr"] for r in result.rows if r["model"] == main_model}
        labels = {n: f"{n} · {float(psnr_of[n]):.1f} dB" for n in worst}
        written["failures"] = save_comparison_grid(
            result.examples,
            worst,
            ["Entrada", main_model, "Objetivo"],
            out / "failures.png",
            row_labels=labels,
        )

    for entry in result.summary:
        fid_text = f" | FID {entry['fid']:.1f}" if "fid" in entry else ""
        print(
            f"[eval] {entry['model']:<28} n={entry['n']:<3} "
            f"PSNR {entry['psnr_mean']:.2f}±{entry['psnr_std']:.2f} dB | "
            f"SSIM {entry['ssim_mean']:.4f}±{entry['ssim_std']:.4f} | "
            f"L1 {entry['l1_mean']:.4f}±{entry['l1_std']:.4f}{fid_text}"
        )
    print(f"[eval] Resultados en {out}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evalúa Pix2Pix en el split de prueba.")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML de entrenamiento.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint del modelo principal.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--out", default=None, help="Carpeta de salida (por defecto <output>/eval)."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        default=[],
        help="Modelo adicional 'Nombre=ruta/checkpoints' (se puede repetir).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifest.csv"))
    parser.add_argument("--no-baselines", action="store_true", help="Omite Entrada y Mediana.")
    parser.add_argument("--fid", action="store_true", help="Calcula FID (descarga InceptionV3).")
    args = parser.parse_args(argv)

    run(
        args.config,
        args.checkpoint,
        split=args.split,
        out_dir=args.out,
        extra_models=args.model,
        batch_size=args.batch_size,
        manifest=args.manifest,
        with_baselines=not args.no_baselines,
        fid=args.fid,
    )


if __name__ == "__main__":
    main()
