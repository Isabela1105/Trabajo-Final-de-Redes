"""Limpieza de puntos de polvo: detecta manchitas aisladas y las rellena con su entorno.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Por qué existe: el generador aprendió a corregir color, contraste y ruido, pero deja pasar
la mayoría de los puntos de polvo (en el test, la degradación "polvo" es de las que menos
mejora). Un punto de polvo es un problema local y bien definido, así que se resuelve mejor
con un paso clásico y controlado que reentrenando la red:

1. **Detectar**: un píxel es sospechoso si es mucho más oscuro (o más claro) que la mediana
   de su vecindario 7×7 **y** además es casi negro (o casi blanco), como el polvo del escáner.
2. **Quedarse solo con manchas compactas**: componentes conectadas pequeñas (≤ `MAX_AREA`
   píxeles y ≤ `MAX_SIDE` de lado). Así no se borran líneas reales (cables, bordes, pasto)
   ni objetos oscuros grandes (ojos, sombras).
3. **Rellenar**: cada píxel marcado toma la mediana de sus vecinos NO marcados.

Se aplica sobre la imagen de 256×256 que entra al generador y, otra vez, sobre su salida
(el generador a veces conserva o suaviza un punto en vez de quitarlo).
"""

from __future__ import annotations

import warnings

import numpy as np

VENTANA = 7  # vecindario para la mediana local
DIFERENCIA = 0.25  # cuánto debe separarse de la mediana local (en [0, 1])
OSCURO = 0.15  # el polvo negro está por debajo de esta luminancia
CLARO = 0.90  # el polvo blanco está por encima
MAX_AREA = 40  # píxeles; un punto de radio 2 ocupa ~13, con el borde suavizado ~25
MAX_SIDE = 8  # lado máximo de la caja que encierra la mancha
MIN_AREA = 3  # un píxel suelto es ruido, no polvo (el polvo de radio 1 ya ocupa ~5)
CROMA = 0.15  # el polvo es negro o blanco puro: los tres canales casi iguales
LLENADO = 0.45  # fracción mínima de la caja ocupada (manchas redondas, no trazos)
ANILLO_MAX = 0.12  # en el anillo alrededor, fracción máxima de píxeles del mismo tono
#                    (un punto está aislado; un cable o un borde sigue más allá de la mancha)


def _shifts(array: np.ndarray, size: int) -> np.ndarray:
    """Pila (size², H, W, ...) con todos los desplazamientos del vecindario (borde reflejado)."""
    pad = size // 2
    widths = [(pad, pad), (pad, pad)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, widths, mode="reflect")
    h, w = array.shape[:2]
    return np.stack([padded[dy : dy + h, dx : dx + w] for dy in range(size) for dx in range(size)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Componentes conectadas (8-vecindad) de una máscara dispersa, sin SciPy."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    found = []
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        stack, pixels = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        found.append(np.array(pixels))
    return found


def _isolated(lum: np.ndarray, pixels: np.ndarray) -> bool:
    """True si alrededor de la mancha (anillo de 2 a 4 px) casi nada tiene su mismo tono."""
    h, w = lum.shape
    y0, x0 = np.maximum(pixels.min(axis=0) - 4, 0)
    y1, x1 = np.minimum(pixels.max(axis=0) + 5, (h, w))
    blob = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    blob[pixels[:, 0] - y0, pixels[:, 1] - x0] = True
    near = _shifts(blob, 3).any(axis=0)
    far = _shifts(blob, 9).any(axis=0)
    ring = far & ~near
    if not ring.any():
        return False
    tone = lum[pixels[:, 0], pixels[:, 1]].mean()
    patch = lum[y0:y1, x0:x1]
    return float(np.mean(np.abs(patch[ring] - tone) < 0.15)) <= ANILLO_MAX


def detect_specks(image: np.ndarray) -> np.ndarray:
    """Máscara booleana (H, W) de puntos de polvo. `image` en [0, 1], (H, W, 3)."""
    lum = image.mean(axis=2)
    gris = (image.max(axis=2) - image.min(axis=2)) < CROMA
    local = np.median(_shifts(lum, VENTANA), axis=0)
    candidates = gris & (
        ((lum < local - DIFERENCIA) & (lum < OSCURO)) | ((lum > local + DIFERENCIA) & (lum > CLARO))
    )
    mask = np.zeros_like(candidates)
    for pixels in _components(candidates):
        height, width = np.ptp(pixels, axis=0) + 1
        area = len(pixels)
        if (
            MIN_AREA <= area <= MAX_AREA
            and height <= MAX_SIDE
            and width <= MAX_SIDE
            and area >= LLENADO * height * width
            and _isolated(lum, pixels)
        ):
            mask[pixels[:, 0], pixels[:, 1]] = True
    # Un píxel más de margen: el borde suavizado del punto también se rellena.
    return _shifts(mask, 3).any(axis=0)


def fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rellena los píxeles marcados con la mediana de sus vecinos no marcados."""
    if not mask.any():
        return image
    out = image.copy()
    size = VENTANA
    for _ in range(3):  # manchas grandes: se rellenan de afuera hacia adentro
        values = _shifts(np.where(mask[..., None], np.nan, out), size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # vecindarios todo marcados
            median = np.nanmedian(values, axis=0)
        filled = mask & ~np.isnan(median).any(axis=2)
        out[filled] = median[filled]
        mask = mask & ~filled
        if not mask.any():
            break
        size += 4
    return out


def remove_specks(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Imagen en [0, 1] (H, W, 3) → (imagen limpia, número de píxeles corregidos)."""
    image = np.asarray(image, dtype=np.float32)
    mask = detect_specks(image)
    return fill(image, mask), int(mask.sum())
