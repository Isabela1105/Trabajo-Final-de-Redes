"""Degradaciones sintéticas para simular fotos antiguas o dañadas.

Cada función recibe una imagen PIL en RGB y un generador aleatorio `rng`
(numpy.random.Generator) y devuelve una imagen nueva. Usar siempre el mismo `rng`
(creado con una semilla) hace que el resultado sea reproducible.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def blur(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Desenfoque gaussiano. Simula una foto movida o un lente sucio."""
    sigma = float(rng.uniform(0.5, 2.0))  # sigma entre 0.5 y 2.0 (ver docs/DATASET.md)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def noise(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Ruido gaussiano. Simula el grano de la película o del sensor."""
    sigma = float(rng.uniform(5, 25))  # escala 0-255
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def jpeg(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Compresión JPEG fuerte. Simula un escaneo viejo o una foto recomprimida."""
    quality = int(rng.integers(20, 61))  # calidad entre 20 y 60
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def downscale(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Baja resolución: reduce la foto y la vuelve a ampliar. Simula una foto pequeña ampliada."""
    factor = int(rng.integers(2, 5))  # factor 2, 3 o 4
    w, h = img.size
    small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def fade(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Desvanecido de color: mezcla con sepia y quita saturación. Simula papel amarillento."""
    amount = float(rng.uniform(0.2, 0.6))  # cuánto sepia se mezcla
    saturation = float(rng.uniform(0.3, 0.8))  # 1.0 = color original, 0 = blanco y negro
    sepia_matrix = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float32,
    )
    arr = np.asarray(img, dtype=np.float32)
    sepia = np.clip(arr @ sepia_matrix.T, 0, 255)
    mixed = (1 - amount) * arr + amount * sepia
    out = Image.fromarray(mixed.astype(np.uint8))
    return ImageEnhance.Color(out).enhance(saturation)


def contrast(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Cambia contraste y brillo. Simula un revelado deficiente."""
    out = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.7, 1.3)))
    return ImageEnhance.Brightness(out).enhance(float(rng.uniform(0.7, 1.3)))


def scratches(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Rayones: líneas finas blancas o negras. Simula daño físico."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for _ in range(int(rng.integers(1, 6))):  # de 1 a 5 rayones
        x0, x1 = (int(v) for v in rng.integers(0, w, size=2))
        y0, y1 = (int(v) for v in rng.integers(0, h, size=2))
        color = (255, 255, 255) if rng.random() < 0.5 else (0, 0, 0)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=int(rng.integers(1, 4)))
    return out


def dust(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Polvo: puntitos blancos y negros. Simula suciedad del escáner."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for _ in range(int(rng.integers(10, 51))):  # de 10 a 50 puntos
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        color = (255, 255, 255) if rng.random() < 0.5 else (0, 0, 0)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return out


# Probabilidad de que cada degradación se aplique a una foto (ver docs/DATASET.md).
# El orden de esta lista es el orden en que se aplican; JPEG va al final.
PIPELINE = [
    ("downscale", downscale, 0.4),
    ("blur", blur, 0.7),
    ("noise", noise, 0.6),
    ("fade", fade, 0.5),
    ("contrast", contrast, 0.3),
    ("scratches", scratches, 0.4),
    ("dust", dust, 0.4),
    ("jpeg", jpeg, 0.5),
]


def apply_random_degradations(
    img: Image.Image, rng: np.random.Generator
) -> tuple[Image.Image, list[str]]:
    """Aplica un subconjunto aleatorio de degradaciones y dice cuáles se aplicaron.

    Devuelve (imagen degradada, lista de nombres). Si el sorteo no elige ninguna,
    se fuerza una para que el par input/target nunca sea idéntico.
    """
    img = img.convert("RGB")
    chosen = [name for name, _, prob in PIPELINE if rng.random() < prob]
    if not chosen:
        chosen = [PIPELINE[int(rng.integers(0, len(PIPELINE)))][0]]
    out = img
    for name, func, _ in PIPELINE:
        if name in chosen:
            out = func(out, rng)
    return out, chosen