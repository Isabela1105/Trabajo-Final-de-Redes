"""La tabla de fotos antiguas reales (evaluación cualitativa) debe estar completa y separada.

Las fotos no se suben a git (están en data/raw/real/); en el repositorio solo está la tabla
data/real_sources.csv. Las comprobaciones sobre los archivos se hacen solo si las fotos existen
en el computador que corre las pruebas.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "raw" / "real"
REAL_CSV = ROOT / "data" / "real_sources.csv"
MANIFEST = ROOT / "data" / "manifest.csv"
CAMPOS = ["descripcion", "anio", "autor", "licencia", "url"]
MIN_FOTOS = 10
MAX_BYTES = 5 * 1024 * 1024


@pytest.fixture(scope="module")
def filas() -> list[dict[str, str]]:
    with open(REAL_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def archivos() -> list[Path]:
    if not REAL_DIR.exists():
        pytest.skip("las fotos reales no están en este computador (no se suben a git)")
    return sorted(REAL_DIR.glob("real_*.jpg"))


def test_hay_suficientes_fotos_reales_registradas(filas):
    assert len(filas) >= MIN_FOTOS


def test_cada_foto_real_tiene_autor_licencia_y_fuente(filas):
    for fila in filas:
        for campo in CAMPOS:
            valor = fila[campo].strip()
            assert valor, f"{fila['archivo']}: falta {campo}"
            assert valor != "POR COMPLETAR", f"{fila['archivo']}: {campo} sin completar"


def test_no_hay_fotos_repetidas_en_la_tabla(filas):
    assert len({f["archivo"] for f in filas}) == len(filas)
    assert len({f["original"] for f in filas}) == len(filas)


def test_las_fotos_reales_no_se_mezclan_con_train_val_test():
    manifest = pd.read_csv(MANIFEST)
    assert not manifest["source_id"].str.startswith("real_").any()
    for col in ("source_path", "input_path", "target_path"):
        assert not manifest[col].str.contains("raw/real").any()


def test_real_sources_esta_en_utf8():
    REAL_CSV.read_bytes().decode("utf-8")  # lanza error si hay bytes inválidos


def test_cada_foto_tiene_su_fila_y_cada_fila_su_foto(filas, archivos):
    assert {f["archivo"] for f in filas} == {a.name for a in archivos}


def test_ninguna_foto_real_pesa_mas_de_5_mb(archivos):
    for a in archivos:
        assert a.stat().st_size <= MAX_BYTES, f"{a.name} pesa más de 5 MB"


def test_las_fotos_reales_son_legibles_y_en_rgb(archivos):
    for a in archivos:
        with Image.open(a) as img:
            img.load()
            assert img.mode == "RGB", f"{a.name} no es RGB"
