"""Estadísticas y galería de ejemplos del dataset, calculadas desde el manifiesto.

Todas las cifras salen de `data/manifest.csv` y `data/sources.csv`; nada se escribe a mano.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.gan.data.dataset import SPLITS
from src.gan.data.degradations import PIPELINE


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Lee el manifiesto como tabla."""
    return pd.read_csv(path)


def pairs_per_split(df: pd.DataFrame) -> pd.DataFrame:
    """Número de fuentes y de pares por split, más una fila de total."""
    rows = []
    for split in SPLITS:
        part = df[df["split"] == split]
        rows.append([split, part["source_id"].nunique(), len(part)])
    rows.append(["total", df["source_id"].nunique(), len(df)])
    return pd.DataFrame(rows, columns=["split", "fuentes", "pares"])


def degradation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cuántos pares tienen cada degradación, por split, y su porcentaje frente a lo configurado."""
    lists = df["degradations"].str.split(";")
    rows = []
    for name, _, prob in PIPELINE:
        has = lists.apply(lambda items, n=name: n in items)
        per_split = [int((has & (df["split"] == s)).sum()) for s in SPLITS]
        total = int(has.sum())
        rows.append([name, *per_split, total, round(100 * total / len(df), 1), round(100 * prob)])
    cols = ["degradacion", *SPLITS, "total", "% de pares", "% configurado"]
    return pd.DataFrame(rows, columns=cols)


def origin_table(df: pd.DataFrame, sources_path: str | Path) -> pd.DataFrame:
    """Fuentes por origen (propias, Pexels, DIV2K) y por split."""
    src = pd.read_csv(sources_path)
    src["source_id"] = src["archivo"].str.rsplit(".", n=1).str[0]

    def origin(licencia: str) -> str:
        if licencia == "propia":
            return "Propias"
        if "Pexels" in licencia:
            return "Pexels"
        return "DIV2K"

    src["origen"] = src["licencia"].apply(origin)
    first = df.drop_duplicates("source_id")[["source_id", "split"]]
    merged = first.merge(src[["source_id", "origen"]], on="source_id", how="left")
    merged["origen"] = merged["origen"].fillna("sin registrar")
    table = merged.pivot_table(
        index="origen", columns="split", values="source_id", aggfunc="count", fill_value=0
    ).reindex(columns=list(SPLITS), fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def original_size_stats(df: pd.DataFrame) -> pd.DataFrame | None:
    """Ancho y alto originales de las fuentes (mín, media, máx). None si no está data/raw."""
    paths = [Path(p) for p in df.drop_duplicates("source_id")["source_path"]]
    if not paths or not all(p.exists() for p in paths):
        return None
    sizes = []
    for p in paths:
        with Image.open(p) as img:
            sizes.append(img.size)
    arr = np.array(sizes)
    rows = [
        [label, int(arr[:, i].min()), round(float(arr[:, i].mean())), int(arr[:, i].max())]
        for i, label in enumerate(["ancho (px)", "alto (px)"])
    ]
    return pd.DataFrame(rows, columns=["medida", "mínimo", "media", "máximo"])


def to_markdown(df: pd.DataFrame) -> str:
    """Convierte una tabla en Markdown (sin depender de librerías extra)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def make_report(manifest_path: str | Path, sources_path: str | Path) -> str:
    """Arma el texto Markdown con todas las estadísticas del dataset."""
    df = load_manifest(manifest_path)
    seeds = sorted(df["seed"].astype(int).tolist())
    parts = [
        "# Estadísticas del dataset",
        "",
        "Generado con `python scripts/dataset_report.py` a partir de `data/manifest.csv`.",
        "",
        "## Fuentes y pares por split",
        "",
        to_markdown(pairs_per_split(df)),
        "",
        f"Variantes por fuente: {df['variant'].nunique()}. Semillas de par: de {seeds[0]} a {seeds[-1]}.",
        "",
        "## Degradaciones aplicadas (número de pares)",
        "",
        to_markdown(degradation_table(df)),
        "",
        "Cada par recibe un subconjunto aleatorio de degradaciones; por eso los porcentajes",
        "reales se parecen a los configurados pero no coinciden exactamente.",
        "",
        "## Fuentes por origen y split",
        "",
        to_markdown(origin_table(df, sources_path)),
    ]
    sizes = original_size_stats(df)
    if sizes is not None:
        parts += ["", "## Tamaño original de las fuentes", "", to_markdown(sizes)]
    return "\n".join(parts) + "\n"


def make_grid(
    df: pd.DataFrame, out_path: str | Path, n: int = 6, seed: int = 0, split: str = "train"
) -> list[str]:
    """Dibuja n pares: fila de arriba = input (degradada), fila de abajo = target (limpia).

    Devuelve los ids de los pares mostrados.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = df[(df["split"] == split) & (df["variant"] == 0)].sort_values("source_id")
    rng = np.random.default_rng(seed)
    chosen = part.iloc[sorted(rng.choice(len(part), size=min(n, len(part)), replace=False))]

    fig, axes = plt.subplots(2, len(chosen), figsize=(2.6 * len(chosen), 5.8), squeeze=False)
    for col, (_, row) in enumerate(chosen.iterrows()):
        axes[0][col].imshow(Image.open(row["input_path"]))
        axes[1][col].imshow(Image.open(row["target_path"]))
        names = row["degradations"].replace(";", " + ")
        axes[0][col].set_title("\n".join(textwrap.wrap(names, 24)), fontsize=7)
        axes[1][col].set_xlabel(row["id"], fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].set_ylabel("input (degradada)", fontsize=9)
    axes[1][0].set_ylabel("target (limpia)", fontsize=9)
    fig.suptitle(f"Ejemplos del dataset (split {split})", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return chosen["id"].tolist()