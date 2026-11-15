"""Pruebas del script que prepara las fotos antiguas reales (evaluación cualitativa)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_real import COLUMNS, MAX_SIDE, PENDIENTE, prepare, read_rows, write_rows


def _carpetas(tmp_path: Path) -> tuple[Path, Path, Path]:
    src = tmp_path / "originales"
    src.mkdir()
    return src, tmp_path / "real", tmp_path / "real_sources.csv"


def _foto(path: Path, size=(1600, 1200)) -> None:
    arr = np.random.default_rng(0).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_crea_copias_rgb_con_lado_mayor_limitado(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "grande.jpg", size=(3000, 2000))
    prepare(src, out, csv_path)
    with Image.open(out / "real_001.jpg") as img:
        assert img.mode == "RGB"
        assert max(img.size) == MAX_SIDE
        assert img.size == (1024, 683)  # conserva la proporción


def test_no_amplia_las_fotos_pequenas(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "chica.png", size=(400, 300))
    prepare(src, out, csv_path)
    with Image.open(out / "real_001.jpg") as img:
        assert img.size == (400, 300)


def test_registra_cada_foto_con_los_campos_por_completar(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "a.jpg")
    _foto(src / "b.png")
    resumen = prepare(src, out, csv_path, licencia="Dominio público")
    filas = read_rows(csv_path)
    assert resumen == {"nuevas": 2, "total": 2, "incompletas": 2}
    assert list(filas[0].keys()) == COLUMNS
    assert [f["archivo"] for f in filas] == ["real_001.jpg", "real_002.jpg"]
    assert [f["original"] for f in filas] == ["a.jpg", "b.png"]
    assert filas[0]["licencia"] == "Dominio público"
    assert filas[0]["autor"] == PENDIENTE


def test_correrlo_dos_veces_no_repite_fotos(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "a.jpg")
    prepare(src, out, csv_path)
    resumen = prepare(src, out, csv_path)
    assert resumen["nuevas"] == 0 and resumen["total"] == 1
    assert len(list(out.glob("*.jpg"))) == 1


def test_agrega_solo_las_nuevas_y_conserva_lo_escrito_a_mano(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "a.jpg")
    prepare(src, out, csv_path)
    filas = read_rows(csv_path)
    filas[0]["autor"] = "Autora Real"
    write_rows(csv_path, filas)

    _foto(src / "b.jpg")
    resumen = prepare(src, out, csv_path)
    filas = read_rows(csv_path)
    assert resumen["nuevas"] == 1
    assert filas[0]["autor"] == "Autora Real"
    assert filas[1]["archivo"] == "real_002.jpg"


def test_aplica_la_rotacion_exif(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    exif = Image.Exif()
    exif[274] = 6  # hay que girarla 90°
    Image.new("RGB", (600, 300), (10, 20, 30)).save(src / "acostada.jpg", exif=exif)
    prepare(src, out, csv_path)
    with Image.open(out / "real_001.jpg") as img:
        assert img.size == (300, 600)


def test_ignora_archivos_que_no_son_imagenes(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    (src / "notas.txt").write_text("hola", encoding="utf-8")
    _foto(src / "a.jpg")
    assert prepare(src, out, csv_path)["total"] == 1


def test_una_foto_completa_no_cuenta_como_incompleta(tmp_path):
    src, out, csv_path = _carpetas(tmp_path)
    _foto(src / "a.jpg")
    prepare(src, out, csv_path)
    fila = read_rows(csv_path)[0]
    fila.update(descripcion="Familia", anio="1920", autor="Anónimo", licencia="PD", url="http://x")
    write_rows(csv_path, [fila])
    assert prepare(src, out, csv_path)["incompletas"] == 0
