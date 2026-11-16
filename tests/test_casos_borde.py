"""Casos borde: imágenes de 16 bits, formas extremas, scripts de línea de comandos y tamaños."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import build_dataset as build_script
from scripts import check_dataset as check_script
from scripts import dataset_report as report_script
from scripts.build_dataset import MANIFEST_COLUMNS, build
from scripts.check_dataset import run_checks
from src.gan.data import degradations
from src.gan.data.dataset import IMAGE_SIZE, load_image, split_sources
from src.gan.data.stats import load_manifest, make_report, original_size_stats

ROOT = Path(__file__).resolve().parents[1]
N_SOURCES = 6


def _make_raw(raw: Path, n: int = N_SOURCES) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.random.default_rng(i).integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw / f"foto_{i:03d}.png")


def _make_sources(path: Path, n: int = N_SOURCES) -> None:
    licencias = ["propia", "propia", "Pexels License"] + ["Solo uso académico (DIV2K)"] * n
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["archivo", "autor", "licencia", "url"])
        for i in range(n):
            writer.writerow([f"foto_{i:03d}.png", "autor", licencias[i], ""])


@pytest.fixture
def built(tmp_path):
    raw, out, manifest = tmp_path / "raw", tmp_path / "processed", tmp_path / "manifest.csv"
    _make_raw(raw)
    build(raw, out, manifest, variants=3, seed=42)
    return raw, out, manifest


# ---------- imágenes de 16 bits y formas extremas ----------


def test_gris_de_16_bits_no_se_va_a_blanco(tmp_path):
    degradado = np.linspace(0, 65535, 300 * 300).reshape(300, 300).astype("uint16")
    ruta = tmp_path / "gris16.png"
    Image.fromarray(degradado).save(ruta)
    assert Image.open(ruta).mode == "I;16"
    arr = load_image(ruta)
    assert arr.min() < -0.9 and arr.max() > 0.9
    assert abs(float(arr.mean())) < 0.1  # un degradado lineal queda centrado
    assert np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2])


def test_gris_de_16_bits_a_media_luz_queda_gris(tmp_path):
    ruta = tmp_path / "medio16.png"
    Image.fromarray(np.full((300, 300), 32768, dtype="uint16")).save(ruta)
    assert abs(float(load_image(ruta).mean())) < 0.02


@pytest.mark.parametrize(
    ("valor", "esperado"), [(100, 100 / 127.5 - 1), (40000, 40000 / 257 / 127.5 - 1)]
)
def test_gris_de_32_bits_se_escala_bien(tmp_path, valor, esperado):
    ruta = tmp_path / "gris32.tif"
    Image.new("I", (300, 300), valor).save(ruta)
    assert Image.open(ruta).mode == "I"
    assert float(load_image(ruta).mean()) == pytest.approx(esperado, abs=0.02)


@pytest.mark.parametrize("tamano", [(10000, 100), (100, 10000), (1, 1), (257, 255)])
def test_formas_extremas_dan_256x256(tmp_path, tamano):
    ruta = tmp_path / "extrema.png"
    Image.new("RGB", tamano, (200, 0, 0)).save(ruta)
    arr = load_image(ruta)
    assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.allclose(arr[..., 0], 200 / 127.5 - 1, atol=0.01)  # no se deforma ni se mezcla


# ---------- split con pocas fuentes ----------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_split_con_muy_pocas_fuentes_no_pierde_ni_repite(n):
    ids = [f"f{i}" for i in range(n)]
    splits = split_sources(ids)
    todos = splits["train"] + splits["val"] + splits["test"]
    assert sorted(todos) == sorted(ids)


# ---------- degradaciones ----------


def test_si_el_sorteo_no_elige_nada_se_fuerza_una_degradacion(monkeypatch):
    sin_azar = [(nombre, func, 0.0) for nombre, func, _ in degradations.PIPELINE]
    monkeypatch.setattr(degradations, "PIPELINE", sin_azar)
    img = Image.new("RGB", (256, 256), (120, 80, 40))
    for seed in range(10):
        _, nombres = degradations.apply_random_degradations(img, np.random.default_rng(seed))
        assert len(nombres) == 1


# ---------- build con una sola variante ----------


def test_build_con_una_variante_por_fuente(tmp_path):
    raw, out, manifest = tmp_path / "raw", tmp_path / "out", tmp_path / "m.csv"
    _make_raw(raw)
    conteo = build(raw, out, manifest, variants=1, seed=1)
    assert sum(conteo.values()) == N_SOURCES


# ---------- validador: casos que faltaban ----------


def _escribir_manifiesto(path: Path, filas: list[dict], columnas=MANIFEST_COLUMNS) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)


def _leer_manifiesto(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_check_detecta_manifiesto_sin_columnas_obligatorias(built):
    _, out, manifest = built
    _escribir_manifiesto(manifest, [], columnas=["id", "split"])
    errores = run_checks(out, manifest)
    assert len(errores) == 1 and "faltan columnas" in errores[0]


def test_check_detecta_manifiesto_vacio(built):
    _, out, manifest = built
    _escribir_manifiesto(manifest, [])
    assert run_checks(out, manifest) == ["el manifiesto está vacío"]


def test_check_detecta_split_desconocido(built):
    _, out, manifest = built
    filas = _leer_manifiesto(manifest)
    filas[0]["split"] = "examen"
    _escribir_manifiesto(manifest, filas)
    assert any("split desconocido" in e for e in run_checks(out, manifest))


def test_check_detecta_target_sin_input(built):
    _, out, manifest = built
    next((out / "val" / "input").glob("*.png")).unlink()
    assert any("tiene target pero no input" in e for e in run_checks(out, manifest))


# ---------- scripts de línea de comandos ----------


def test_build_dataset_por_linea_de_comandos(tmp_path, monkeypatch, capsys):
    raw, out, manifest = tmp_path / "raw", tmp_path / "out", tmp_path / "m.csv"
    _make_raw(raw)
    argv = ["build_dataset.py", "--raw", str(raw), "--out", str(out), "--manifest", str(manifest)]
    monkeypatch.setattr(sys, "argv", argv)
    build_script.main()
    assert "Pares por split" in capsys.readouterr().out
    assert manifest.exists()


def test_check_dataset_por_linea_de_comandos_ok(built, monkeypatch, capsys):
    _, out, manifest = built
    monkeypatch.setattr(
        sys, "argv", ["check_dataset.py", "--root", str(out), "--manifest", str(manifest)]
    )
    check_script.main()
    assert "Dataset OK" in capsys.readouterr().out


def test_check_dataset_por_linea_de_comandos_con_error_sale_con_codigo_1(
    built, monkeypatch, capsys
):
    _, out, manifest = built
    next((out / "train" / "target").glob("*.png")).unlink()
    monkeypatch.setattr(
        sys, "argv", ["check_dataset.py", "--root", str(out), "--manifest", str(manifest)]
    )
    with pytest.raises(SystemExit) as salida:
        check_script.main()
    assert salida.value.code == 1
    assert "problemas" in capsys.readouterr().out


def test_dataset_report_genera_estadisticas_y_galeria(built, tmp_path, monkeypatch, capsys):
    _, _, manifest = built
    sources = tmp_path / "sources.csv"
    _make_sources(sources)
    grid, stats = tmp_path / "galeria.png", tmp_path / "stats.md"
    argv = ["dataset_report.py", "--manifest", str(manifest), "--sources", str(sources)]
    argv += ["--grid", str(grid), "--stats", str(stats), "--examples", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    report_script.main()
    assert grid.exists() and stats.exists()
    assert "Estadísticas del dataset" in stats.read_text(encoding="utf-8")


def test_el_informe_de_estadisticas_tiene_todas_las_secciones(built, tmp_path):
    raw, _, manifest = built
    sources = tmp_path / "sources.csv"
    _make_sources(sources)
    texto = make_report(manifest, sources)
    for seccion in ("Fuentes y pares por split", "Degradaciones aplicadas", "origen y split"):
        assert seccion in texto
    assert "Tamaño original" in texto


def test_tamano_original_es_none_si_faltan_las_fotos_fuente(built):
    raw, _, manifest = built
    df = load_manifest(manifest)
    assert original_size_stats(df) is not None
    for foto in raw.glob("*.png"):
        foto.unlink()
    assert original_size_stats(df) is None


# ---------- tamaño de los archivos del repositorio ----------


def test_ningun_archivo_del_repositorio_pesa_mas_de_5_mb():
    try:
        salida = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no se pudo consultar git en este entorno")
    limite = 5 * 1024**2
    pesados = [
        f
        for f in salida.splitlines()
        if (ROOT / f).is_file() and (ROOT / f).stat().st_size > limite
    ]
    assert pesados == [], f"archivos de más de 5 MB: {pesados}"
