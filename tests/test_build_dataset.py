"""Pruebas del script de construcción, del validador y de las estadísticas del dataset."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.build_dataset import MANIFEST_COLUMNS, build
from scripts.check_dataset import run_checks
from src.gan.data.stats import (
    degradation_table,
    load_manifest,
    make_grid,
    pairs_per_split,
)

N_SOURCES = 6
VARIANTS = 3


def _make_raw(raw: Path, n: int = N_SOURCES) -> None:
    """Crea n fotos fuente pequeñas, no cuadradas, con contenido aleatorio."""
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw / f"foto_{i:03d}.png")


@pytest.fixture
def built(tmp_path):
    """Dataset pequeño ya construido: devuelve (raw, out, manifest)."""
    raw, out, manifest = tmp_path / "raw", tmp_path / "processed", tmp_path / "manifest.csv"
    _make_raw(raw)
    build(raw, out, manifest, variants=VARIANTS, seed=42)
    return raw, out, manifest


def _read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hash_pngs(out: Path) -> dict[str, str]:
    return {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*.png"))
    }


# ---------- construcción ----------


def test_manifiesto_tiene_las_columnas_esperadas(built):
    _, _, manifest = built
    assert list(_read_manifest(manifest)[0].keys()) == MANIFEST_COLUMNS


def test_tres_variantes_por_fuente_y_conteo_total(built):
    _, _, manifest = built
    rows = _read_manifest(manifest)
    assert len(rows) == N_SOURCES * VARIANTS
    por_fuente: dict[str, int] = {}
    for r in rows:
        por_fuente[r["source_id"]] = por_fuente.get(r["source_id"], 0) + 1
    assert set(por_fuente.values()) == {VARIANTS}


def test_manifiesto_no_tiene_fuga_entre_splits(built):
    _, _, manifest = built
    splits_de: dict[str, set[str]] = {}
    for r in _read_manifest(manifest):
        splits_de.setdefault(r["source_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits_de.values())


def test_input_y_target_existen_para_cada_fila(built):
    _, _, manifest = built
    for r in _read_manifest(manifest):
        assert Path(r["input_path"]).exists()
        assert Path(r["target_path"]).exists()


def test_el_input_es_distinto_del_target(built):
    _, _, manifest = built
    r = _read_manifest(manifest)[0]
    x = np.asarray(Image.open(r["input_path"]))
    y = np.asarray(Image.open(r["target_path"]))
    assert x.shape == y.shape == (256, 256, 3)
    assert not np.array_equal(x, y)


# ---------- reproducibilidad ----------


def test_misma_semilla_mismo_manifiesto_y_mismas_imagenes(built):
    raw, out, manifest = built
    manifiesto_1 = manifest.read_bytes()
    imagenes_1 = _hash_pngs(out)
    build(raw, out, manifest, variants=VARIANTS, seed=42)  # segunda corrida
    assert manifest.read_bytes() == manifiesto_1
    assert _hash_pngs(out) == imagenes_1


def test_semilla_distinta_cambia_el_resultado(built):
    raw, out, manifest = built
    manifiesto_1 = manifest.read_bytes()
    build(raw, out, manifest, variants=VARIANTS, seed=7)
    assert manifest.read_bytes() != manifiesto_1


def test_reconstruir_borra_pares_viejos(built):
    raw, out, manifest = built
    viejo = out / "train" / "input" / "huerfano_v0.png"
    Image.new("RGB", (256, 256)).save(viejo)
    build(raw, out, manifest, variants=VARIANTS, seed=42)
    assert not viejo.exists()


# ---------- casos borde de build ----------


def test_build_falla_si_no_hay_imagenes(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "raw", tmp_path / "out", tmp_path / "m.csv")


def test_build_falla_con_nombres_repetidos(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (300, 300)).save(raw / "a.png")
    Image.new("RGB", (300, 300)).save(raw / "a.jpg")
    with pytest.raises(ValueError):
        build(raw, tmp_path / "out", tmp_path / "m.csv")


def test_build_rechaza_numero_de_variantes_invalido(tmp_path):
    _make_raw(tmp_path / "raw", n=3)
    with pytest.raises(ValueError):
        build(tmp_path / "raw", tmp_path / "out", tmp_path / "m.csv", variants=0)


# ---------- validador (check_dataset) ----------


def test_check_acepta_un_dataset_valido(built):
    _, out, manifest = built
    assert run_checks(out, manifest) == []


def test_check_detecta_fuga(built):
    _, out, manifest = built
    rows = _read_manifest(manifest)
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    test["source_id"] = train["source_id"]  # la misma fuente en train y en test
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert any("FUGA" in e for e in run_checks(out, manifest))


def test_check_detecta_archivo_corrupto(built):
    _, out, manifest = built
    victima = next((out / "train" / "input").glob("*.png"))
    victima.write_bytes(victima.read_bytes()[:100])  # PNG truncado
    assert any("corrupto" in e for e in run_checks(out, manifest))


def test_check_detecta_par_incompleto(built):
    _, out, manifest = built
    next((out / "val" / "target").glob("*.png")).unlink()
    errores = run_checks(out, manifest)
    assert any("no existe" in e for e in errores)
    assert any("tiene input pero no target" in e for e in errores)


def test_check_detecta_tamano_incorrecto(built):
    _, out, manifest = built
    victima = next((out / "train" / "input").glob("*.png"))
    Image.new("RGB", (100, 100)).save(victima)
    assert any("tamaño" in e for e in run_checks(out, manifest))


def test_check_detecta_imagen_que_no_es_rgb(built):
    _, out, manifest = built
    victima = next((out / "train" / "target").glob("*.png"))
    Image.new("L", (256, 256)).save(victima)
    assert any("no es RGB" in e for e in run_checks(out, manifest))


def test_check_detecta_archivo_fuera_del_manifiesto(built):
    _, out, manifest = built
    origen = next((out / "train" / "input").glob("*.png"))
    shutil.copy(origen, origen.with_name("extra_v0.png"))
    shutil.copy(out / "train" / "target" / origen.name, out / "train" / "target" / "extra_v0.png")
    assert any("no está en el manifiesto" in e for e in run_checks(out, manifest))


def test_check_avisa_si_falta_el_manifiesto(tmp_path):
    errores = run_checks(tmp_path, tmp_path / "no_existe.csv")
    assert len(errores) == 1
    assert "no existe el manifiesto" in errores[0]


# ---------- estadísticas y galería ----------


def test_estadisticas_coinciden_con_el_manifiesto(built):
    _, _, manifest = built
    df = load_manifest(manifest)
    tabla = pairs_per_split(df)
    assert tabla["pares"].iloc[-1] == len(df)
    assert tabla["pares"].iloc[:-1].sum() == len(df)
    assert tabla["fuentes"].iloc[-1] == N_SOURCES


def test_tabla_de_degradaciones_es_consistente(built):
    _, _, manifest = built
    df = load_manifest(manifest)
    tabla = degradation_table(df)
    assert (tabla["total"] <= len(df)).all()
    assert (tabla[["train", "val", "test"]].sum(axis=1) == tabla["total"]).all()


def test_galeria_se_genera(built, tmp_path):
    _, _, manifest = built
    salida = tmp_path / "galeria.png"
    ids = make_grid(load_manifest(manifest), salida, n=3)
    assert salida.exists() and salida.stat().st_size > 0
    assert len(ids) == 3
