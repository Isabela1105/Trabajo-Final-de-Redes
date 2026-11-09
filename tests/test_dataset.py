"""Pruebas del dataset: formas, rangos, reproducibilidad y ausencia de fuga entre splits."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.gan.data.dataset import (
    IMAGE_SIZE,
    build_dataset,
    check_no_leakage,
    load_image,
    load_pair,
    split_sources,
)
from src.gan.data.degradations import PIPELINE, apply_random_degradations


def _make_image(path, size=(600, 300), color=(200, 100, 50)):
    """Crea una imagen de prueba (por defecto NO cuadrada)."""
    Image.new("RGB", size, color).save(path)
    return path


# ---------- formas, tipo y rango ----------


@pytest.mark.parametrize("size", [(600, 300), (300, 600), (256, 256), (1, 1), (64, 64)])
def test_load_image_forma_tipo_y_rango(tmp_path, size):
    path = _make_image(tmp_path / "foto.png", size=size)
    arr = load_image(path)
    assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= -1.0
    assert arr.max() <= 1.0


def test_load_image_blanco_y_negro_dan_los_extremos(tmp_path):
    blanco = load_image(_make_image(tmp_path / "b.png", color=(255, 255, 255)))
    negro = load_image(_make_image(tmp_path / "n.png", color=(0, 0, 0)))
    assert np.allclose(negro, -1.0)
    assert np.allclose(blanco, 1.0, atol=0.01)


def test_load_image_convierte_gris_a_rgb(tmp_path):
    path = tmp_path / "gris.png"
    Image.new("L", (100, 100), 128).save(path)
    assert load_image(path).shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_load_pair_devuelve_input_y_target_con_la_misma_forma(tmp_path):
    a = _make_image(tmp_path / "a.png")
    b = _make_image(tmp_path / "b.png", color=(10, 20, 30))
    x, y = load_pair(a, b)
    assert x.shape == y.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert not np.array_equal(x, y)


def test_build_dataset_entrega_pares(tmp_path):
    for sub in ("input", "target"):
        (tmp_path / sub).mkdir()
        _make_image(tmp_path / sub / "0001.png")
    pares = list(build_dataset(tmp_path))
    assert len(pares) == 1
    x, y = pares[0]
    assert x.shape == y.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_build_dataset_falla_si_falta_el_target(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "target").mkdir()
    _make_image(tmp_path / "input" / "0001.png")
    with pytest.raises(FileNotFoundError):
        list(build_dataset(tmp_path))


# ---------- split por fuente y no fuga ----------


def test_split_proporciones_80_10_10():
    splits = split_sources([f"foto{i}" for i in range(150)])
    assert {k: len(v) for k, v in splits.items()} == {"train": 120, "val": 15, "test": 15}


def test_no_fuga_entre_splits():
    splits = split_sources([f"foto{i}" for i in range(150)])
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    check_no_leakage(splits)  # no debe lanzar error


def test_cada_fuente_cae_en_un_solo_split():
    ids = [f"foto{i}" for i in range(150)]
    splits = split_sources(ids)
    todos = splits["train"] + splits["val"] + splits["test"]
    assert sorted(todos) == sorted(ids)  # ninguna se pierde ni se repite


def test_check_no_leakage_detecta_fuga():
    with pytest.raises(ValueError, match="Fuga de datos"):
        check_no_leakage({"train": ["a", "b"], "val": ["c"], "test": ["a"]})


def test_split_misma_semilla_mismo_resultado_sin_importar_el_orden():
    ids = [f"foto{i}" for i in range(50)]
    assert split_sources(ids, seed=7) == split_sources(list(reversed(ids)), seed=7)


def test_split_semillas_distintas_dan_repartos_distintos():
    ids = [f"foto{i}" for i in range(50)]
    assert split_sources(ids, seed=1) != split_sources(ids, seed=2)


def test_split_con_pocas_fotos_deja_al_menos_una_en_val_y_test():
    splits = split_sources(["a", "b", "c"])
    assert all(len(v) == 1 for v in splits.values())


def test_split_rechaza_proporciones_que_no_suman_uno():
    with pytest.raises(ValueError):
        split_sources(["a", "b", "c"], ratios=(0.5, 0.5, 0.5))


# ---------- degradaciones ----------


def test_degradaciones_conservan_tamano_y_modo():
    img = Image.new("RGB", (256, 256), (120, 80, 40))
    for seed in range(30):
        out, nombres = apply_random_degradations(img, np.random.default_rng(seed))
        assert out.size == (256, 256)
        assert out.mode == "RGB"
        assert len(nombres) >= 1
        assert set(nombres) <= {name for name, _, _ in PIPELINE}


def test_degradaciones_reproducibles_con_la_misma_semilla():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    a, nombres_a = apply_random_degradations(img, np.random.default_rng(5))
    b, nombres_b = apply_random_degradations(img, np.random.default_rng(5))
    assert nombres_a == nombres_b
    assert np.array_equal(np.asarray(a), np.asarray(b))
