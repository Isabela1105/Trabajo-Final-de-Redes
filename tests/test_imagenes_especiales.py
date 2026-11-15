"""Imágenes con transparencia (alfa), rotación EXIF y otros modos de color."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.gan.data.dataset import IMAGE_SIZE, load_image, prepare_image, to_array, to_rgb


def test_png_totalmente_transparente_queda_blanco_y_no_negro(tmp_path):
    ruta = tmp_path / "transparente.png"
    Image.new("RGBA", (300, 300), (255, 0, 0, 0)).save(ruta)
    arr = load_image(ruta)
    assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.allclose(arr, 1.0, atol=0.01)  # blanco = 1.0 (negro sería -1.0)


def test_png_semitransparente_se_mezcla_con_blanco(tmp_path):
    ruta = tmp_path / "semi.png"
    Image.new("RGBA", (300, 300), (255, 0, 0, 128)).save(ruta)
    arr = load_image(ruta)
    assert np.allclose(arr[..., 0], 1.0, atol=0.02)  # rojo se mantiene
    assert np.allclose(arr[..., 1], 0.0, atol=0.05)  # verde y azul quedan a media luz
    assert np.allclose(arr[..., 2], 0.0, atol=0.05)


def test_imagen_con_paleta_y_transparencia_queda_blanca(tmp_path):
    ruta = tmp_path / "paleta.png"
    Image.new("P", (300, 300), 0).save(ruta, transparency=0)
    assert np.allclose(load_image(ruta), 1.0, atol=0.01)


def test_imagen_gris_con_alfa_pasa_a_rgb(tmp_path):
    ruta = tmp_path / "gris_alfa.png"
    Image.new("LA", (300, 300), (100, 255)).save(ruta)
    assert load_image(ruta).shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_exif_rotacion_se_aplica(tmp_path):
    """Foto 400x200 'acostada' (izquierda roja, derecha azul) con EXIF orientación 6."""
    img = Image.new("RGB", (400, 200))
    img.paste((255, 0, 0), (0, 0, 200, 200))
    img.paste((0, 0, 255), (200, 0, 400, 200))
    exif = Image.Exif()
    exif[274] = 6  # hay que rotarla 90° para verla derecha
    ruta = tmp_path / "rotada.jpg"
    img.save(ruta, exif=exif, quality=100)

    arr = load_image(ruta)
    arriba, abajo = arr[10, 128], arr[245, 128]
    assert arriba[0] > 0.5 and arriba[2] < -0.5  # arriba: rojo
    assert abajo[2] > 0.5 and abajo[0] < -0.5  # abajo: azul


def test_sin_exif_la_imagen_no_se_rota(tmp_path):
    img = Image.new("RGB", (400, 200))
    img.paste((255, 0, 0), (0, 0, 200, 200))
    img.paste((0, 0, 255), (200, 0, 400, 200))
    ruta = tmp_path / "normal.png"
    img.save(ruta)
    arr = load_image(ruta)
    izquierda, derecha = arr[128, 10], arr[128, 245]
    assert izquierda[0] > 0.5 and derecha[2] > 0.5


@pytest.mark.parametrize("modo", ["L", "CMYK", "RGB", "P"])
def test_otros_modos_de_color_dan_rgb_256(tmp_path, modo):
    ruta = tmp_path / ("cmyk.jpg" if modo == "CMYK" else f"{modo}.png")  # CMYK no existe en PNG
    Image.new(modo, (300, 200)).save(ruta)
    assert load_image(ruta).shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_imagen_rgb_normal_no_cambia_al_convertir():
    original = Image.new("RGB", (50, 50), (10, 200, 30))
    assert list(to_rgb(original).getdata()) == list(original.getdata())


def test_prepare_image_devuelve_rgb_cuadrada():
    out = prepare_image(Image.new("RGBA", (500, 300), (0, 0, 0, 255)))
    assert out.mode == "RGB" and out.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert to_array(out).dtype == np.float32
