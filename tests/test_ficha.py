"""La ficha del dataset debe coincidir con el manifiesto y estar en UTF-8."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "data" / "README.md"
MANIFEST = ROOT / "data" / "manifest.csv"


@pytest.fixture(scope="module")
def ficha() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def _numero(ficha: str, campo: str) -> int:
    m = re.search(rf"\|\s*{campo}\s*\|\s*(\d+)\s*\|", ficha)
    assert m, f"No encontré la fila '{campo}' en data/README.md"
    return int(m.group(1))


@pytest.mark.parametrize(
    "ruta",
    ["data/README.md", "docs/DATASET.md", "data/sources.csv", "data/manifest.csv"],
)
def test_documentos_del_dataset_estan_en_utf8(ruta):
    (ROOT / ruta).read_bytes().decode("utf-8")  # lanza error si hay bytes inválidos


def test_ficha_fuentes_coincide_con_el_manifiesto(ficha, manifest):
    assert _numero(ficha, "Fuentes") == manifest["source_id"].nunique()


def test_ficha_pares_coincide_con_el_manifiesto(ficha, manifest):
    assert _numero(ficha, "Pares") == len(manifest)


def test_ficha_pares_por_split_coincide_con_el_manifiesto(ficha, manifest):
    m = re.search(r"\|\s*Pares por split\s*\|\s*train (\d+) / val (\d+) / test (\d+)\s*\|", ficha)
    assert m, "No encontré la fila 'Pares por split' en data/README.md"
    esperado = manifest["split"].value_counts()
    assert tuple(int(x) for x in m.groups()) == (
        esperado["train"],
        esperado["val"],
        esperado["test"],
    )


def test_ficha_semilla_base_coincide_con_el_manifiesto(ficha, manifest):
    assert _numero(ficha, "Semilla base") == int(manifest["seed"].min()) // 1_000_000
