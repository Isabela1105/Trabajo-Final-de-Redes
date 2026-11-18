"""Pruebas del cargador de datos de TensorFlow (se omiten si TensorFlow no está instalado)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from scripts.build_dataset import build  # noqa: E402
from src.gan.data.dataset import IMAGE_SIZE, load_pair  # noqa: E402
from src.gan.data.tf_dataset import list_pairs, make_dataset, make_datasets  # noqa: E402

N_SOURCES = 10
VARIANTS = 3


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """Dataset pequeño ya construido (10 fotos x 3 variantes) con la misma ruta que el real."""
    base = tmp_path_factory.mktemp("tf")
    raw = base / "raw"
    raw.mkdir()
    for i in range(N_SOURCES):
        arr = np.random.default_rng(i).integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(raw / f"foto_{i:03d}.png")
    out = base / "processed"
    build(raw, out, base / "manifest.csv", variants=VARIANTS, seed=42)
    return out


def _todos_los_lotes(ds):
    return [(x.numpy(), y.numpy()) for x, y in ds]


def test_los_lotes_tienen_forma_tipo_y_rango_del_contrato(dataset_dir):
    x, y = next(iter(make_dataset(dataset_dir / "train", batch_size=2)))
    for lote in (x, y):
        assert lote.shape == (2, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert lote.dtype == tf.float32
        assert float(tf.reduce_min(lote)) >= -1.0
        assert float(tf.reduce_max(lote)) <= 1.0


def test_da_los_mismos_valores_que_load_pair(dataset_dir):
    inputs, targets = list_pairs(dataset_dir / "val")
    x, y = next(iter(make_dataset(dataset_dir / "val", batch_size=1)))
    esperado_x, esperado_y = load_pair(inputs[0], targets[0])
    assert np.allclose(x.numpy()[0], esperado_x, atol=1e-6)
    assert np.allclose(y.numpy()[0], esperado_y, atol=1e-6)


def test_recorre_todos_los_pares_una_sola_vez(dataset_dir):
    inputs, _ = list_pairs(dataset_dir / "train")
    lotes = _todos_los_lotes(make_dataset(dataset_dir / "train", batch_size=4, shuffle=True))
    assert sum(len(x) for x, _ in lotes) == len(inputs)


def test_un_lote_mas_grande_que_el_split_da_un_solo_lote(dataset_dir):
    lotes = _todos_los_lotes(make_dataset(dataset_dir / "test", batch_size=1000))
    assert len(lotes) == 1


def test_sin_mezclar_el_orden_es_siempre_el_mismo(dataset_dir):
    a = _todos_los_lotes(make_dataset(dataset_dir / "train", batch_size=3))
    b = _todos_los_lotes(make_dataset(dataset_dir / "train", batch_size=3))
    assert all(np.array_equal(x1, x2) for (x1, _), (x2, _) in zip(a, b, strict=True))


def test_mezclar_con_la_misma_semilla_es_reproducible(dataset_dir):
    def orden(seed):
        ds = make_dataset(dataset_dir / "train", batch_size=1, shuffle=True, seed=seed)
        return [float(tf.reduce_sum(x)) for x, _ in ds]

    assert orden(1) == orden(1)
    assert orden(1) != orden(2)


def test_el_input_y_el_target_de_cada_pareja_van_alineados(dataset_dir):
    """Con la mezcla activada, el input y el target siguen siendo de la misma foto."""
    inputs, targets = list_pairs(dataset_dir / "train")
    target_de = {}
    for ruta_input, ruta_target in zip(inputs, targets, strict=True):
        x, y = load_pair(ruta_input, ruta_target)
        target_de[round(float(x.sum()), 1)] = y  # "huella" del input -> su target
    for x, y in make_dataset(dataset_dir / "train", batch_size=1, shuffle=True):
        huella = round(float(x.numpy().sum()), 1)
        assert np.allclose(y.numpy()[0], target_de[huella], atol=1e-6)


def test_falla_si_falta_un_target(dataset_dir, tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "target").mkdir()
    Image.new("RGB", (256, 256)).save(tmp_path / "input" / "a.png")
    with pytest.raises(FileNotFoundError, match="Falta el target"):
        make_dataset(tmp_path)


def test_falla_si_la_carpeta_esta_vacia(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError, match="No hay imágenes"):
        make_dataset(tmp_path)


def test_falla_al_leer_una_imagen_que_no_mide_256(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "target").mkdir()
    for carpeta in ("input", "target"):
        Image.new("RGB", (100, 100)).save(tmp_path / carpeta / "a.png")
    with pytest.raises(tf.errors.InvalidArgumentError):
        _todos_los_lotes(make_dataset(tmp_path))


def test_make_datasets_arma_train_val_y_test_desde_la_configuracion(dataset_dir):
    cfg = {
        "train_dir": str(dataset_dir / "train"),
        "val_dir": str(dataset_dir / "val"),
        "test_dir": str(dataset_dir / "test"),
        "image_size": 256,
        "batch_size": 2,
        "num_parallel_calls": 4,
    }
    train, val, test = make_datasets(cfg, seed=7)
    for ds in (train, val, test):
        x, y = next(iter(ds))
        assert x.shape[1:] == (256, 256, 3) and y.shape[1:] == (256, 256, 3)
