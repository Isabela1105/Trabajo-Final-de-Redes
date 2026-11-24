"""Inferencia: carga el generador desde un checkpoint y restaura una imagen nueva.

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Contrato con el entrenamiento (Cristian): `train.py` guarda en `checkpoints/` los archivos
`ckpt-N.index` / `ckpt-N.data-*` (G, D y optimizadores) y, al lado, el `config.yaml` con
el que se entrenó. Aquí el generador se reconstruye con ESE config, usando el mismo
`generator_builder` del YAML; nunca se escribe `ngf` ni `niveles` a mano.

Uso:
    python -m src.gan.infer --checkpoint outputs/base/checkpoints \
        --input foto.jpg --output restaurada.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.gan.config import Config, load_config
from src.gan.data.dataset import to_rgb


def resolve_checkpoint(path: str | Path) -> str:
    """Devuelve el prefijo `.../ckpt-N` a partir de una carpeta o de un prefijo.

    - Carpeta: se usa el último checkpoint que registró el `CheckpointManager`.
    - Prefijo (`.../ckpt-19`): se valida que exista su `.index`.
    """
    path = Path(path)
    if path.is_dir():
        latest = tf.train.latest_checkpoint(str(path))
        if latest is None:
            raise FileNotFoundError(f"No hay checkpoints en {path}")
        return latest
    if Path(f"{path}.index").exists():
        return str(path)
    raise FileNotFoundError(f"No existe el checkpoint {path} (falta {path}.index)")


def _config_for(checkpoint_prefix: str, config_path: str | Path | None) -> Config:
    """Config explícito si se pasa; si no, el `config.yaml` guardado junto al checkpoint."""
    if config_path is not None:
        return load_config(config_path, fixed_seed=False)
    # `train.py` copia el YAML en checkpoints/ y en la carpeta de salida; corridas de versiones
    # anteriores solo lo tienen en la carpeta de salida.
    folder = Path(checkpoint_prefix).parent
    for saved in (folder / "config.yaml", folder.parent / "config.yaml"):
        if saved.exists():
            return load_config(saved, fixed_seed=False)
    raise FileNotFoundError(
        f"No se encontró config.yaml en {folder} ni en {folder.parent}. "
        "Pase --config con el YAML usado para entrenar."
    )


def build_generator_from_config(config: Config) -> tf.keras.Model:
    """Construye el generador exactamente como lo hace `train.py`.

    Se reutilizan `_import_object` y `_call_builder` del entrenamiento para no duplicar la
    traducción `n_downsample -> niveles` ni la firma del builder.
    """
    from src.gan.train import DEFAULT_GENERATOR_BUILDER, _call_builder, _import_object

    spec = config.model.get("generator_builder", DEFAULT_GENERATOR_BUILDER)
    return _call_builder(_import_object(spec), config)


def load_generator(
    checkpoint: str | Path, config_path: str | Path | None = None
) -> tuple[tf.keras.Model, Config]:
    """Restaura SOLO el generador de un checkpoint de entrenamiento.

    `expect_partial()` es necesario porque el checkpoint también trae D y los optimizadores,
    que en inferencia no se usan. `assert_existing_objects_matched()` sí obliga a que todas
    las variables del generador se hayan cargado: si la arquitectura no coincide, falla aquí
    y no produce imágenes basura en silencio.
    """
    prefix = resolve_checkpoint(checkpoint)
    config = _config_for(prefix, config_path)
    generator = build_generator_from_config(config)
    status = tf.train.Checkpoint(generator=generator).restore(prefix)
    status.expect_partial().assert_existing_objects_matched()
    return generator, config


# Fotos de celular de 12 MP (4000×3000) o más se reducen antes de procesar: el modelo trabaja a
# 256×256, así que conservar más píxeles solo gasta memoria y tiempo.
MAX_SIDE = 4096


@dataclass(frozen=True)
class Prepared:
    """Imagen lista para el generador y lo necesario para deshacer el relleno."""

    tensor: tf.Tensor  # (1, size, size, 3) en [-1, 1]
    size: tuple[int, int]  # (ancho, alto) tras EXIF y reducción
    box: tuple[int, int, int, int]  # región útil dentro del cuadrado relleno
    side: int  # lado del cuadrado relleno


def _pad_to_square(array: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rellena por reflexión hasta un cuadrado; devuelve el arreglo y la caja original."""
    h, w = array.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    pads = ((top, side - h - top), (left, side - w - left), (0, 0))
    mode = "reflect" if min(h, w) > 1 else "edge"
    return np.pad(array, pads, mode=mode), (left, top, left + w, top + h)


def prepare_input(image: Image.Image, size: int) -> Prepared:
    """EXIF, alfa, modos raros, tamaño máximo y relleno cuadrado → tensor en [-1, 1].

    - `to_rgb` (del módulo de datos) aplica la rotación EXIF y compone el alfa sobre BLANCO;
      sin esto, un PNG transparente sale negro y una foto vertical sale acostada.
    - El relleno por reflexión evita deformar fotos no cuadradas (se recorta después).
    """
    rgb = to_rgb(image)
    if max(rgb.size) > MAX_SIDE:
        rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    square, box = _pad_to_square(np.asarray(rgb, dtype=np.uint8))
    small = Image.fromarray(square).resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(small, dtype=np.float32) / 127.5 - 1.0
    return Prepared(
        tensor=tf.convert_to_tensor(array[np.newaxis]),
        size=rgb.size,
        box=box,
        side=square.shape[0],
    )


def image_to_tensor(image: Image.Image, size: int) -> tf.Tensor:
    """PIL → tensor (1, size, size, 3) float32 en [-1, 1] (con EXIF, alfa y relleno)."""
    return prepare_input(image, size).tensor


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """Tensor (1, H, W, 3) o (H, W, 3) en [-1, 1] → PIL RGB uint8."""
    array = tf.convert_to_tensor(tensor, dtype=tf.float32)
    if array.shape.rank == 4:
        array = array[0]
    array = np.clip((array.numpy() + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def generate(generator: tf.keras.Model, tensor: tf.Tensor, *, tta: bool = False) -> tf.Tensor:
    """Generador sobre un lote en [-1, 1]. `tta=True` promedia las 4 versiones volteadas.

    Test-time augmentation: se restaura la imagen original, volteada en horizontal, en vertical
    y en ambos ejes; cada salida se devuelve a su orientación y se promedian. Los errores de cada
    pasada no coinciden, así que el promedio tiene menos ruido y artefactos de cuadrícula. Se
    eligió con el split de validación (+0.05 dB, +0.014 SSIM) y no con el de prueba.
    """
    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    if not tta:
        return tf.cast(generator(tensor, training=False), tf.float32)
    total = tf.zeros_like(tensor)
    for axes in ((), (2,), (1,), (1, 2)):
        flipped = tf.reverse(tensor, axis=list(axes)) if axes else tensor
        out = tf.cast(generator(flipped, training=False), tf.float32)
        total += tf.reverse(out, axis=list(axes)) if axes else out
    return total / 4.0


def sharpen(tensor: tf.Tensor, amount: float, sigma: float = 1.0) -> tf.Tensor:
    """Máscara de enfoque (unsharp mask) sobre un lote en [-1, 1]: x + amount·(x − blur(x)).

    Solo mejora la nitidez **a la vista**: en validación y prueba baja el PSNR y el SSIM con
    cualquier intensidad, porque también realza el ruido. Por eso es opcional (demo) y la tabla
    de resultados no la usa.
    """
    if amount <= 0:
        return tf.convert_to_tensor(tensor, dtype=tf.float32)
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = (np.outer(g, g)[:, :, None, None] * np.eye(3, dtype=np.float32)[None, None]).astype(
        np.float32
    )
    t = tf.convert_to_tensor(tensor, dtype=tf.float32)
    padded = tf.pad(t, [[0, 0], [radius, radius], [radius, radius], [0, 0]], mode="REFLECT")
    blurred = tf.nn.conv2d(padded, kernel, strides=1, padding="VALID")
    return tf.clip_by_value(t + amount * (t - blurred), -1.0, 1.0)


def predict(
    generator: tf.keras.Model,
    tensor: tf.Tensor,
    *,
    quitar_puntos: bool = True,
    tta: bool = True,
    nitidez: float = 0.0,
    mezcla: float = 1.0,
) -> tf.Tensor:
    """Generador sobre un lote en [-1, 1] con los pasos opcionales de la demo.

    1. Limpieza de polvo antes del generador (`src/gan/despeckle.py`), para que no lo copie.
    2. Generador, con TTA de 4 volteos si `tta`.
    3. Limpieza de polvo otra vez, por si alguno sobrevivió.
    4. Mezcla con la entrada si `mezcla < 1`: `mezcla·salida + (1 − mezcla)·entrada`.
       Frena la sobre-corrección en fotos poco dañadas (ver `docs/RESULTADOS.md` §7).
    5. Máscara de enfoque si `nitidez > 0` (solo visual).
    """
    if quitar_puntos:
        from src.gan.despeckle import remove_specks

        def limpiar(batch: np.ndarray) -> np.ndarray:
            unit = (np.asarray(batch, dtype=np.float32) + 1.0) / 2.0
            return np.stack([remove_specks(img)[0] for img in unit]) * 2.0 - 1.0

        tensor = tf.convert_to_tensor(limpiar(tensor), dtype=tf.float32)
    pred = generate(generator, tensor, tta=tta)
    if quitar_puntos:
        pred = tf.convert_to_tensor(limpiar(np.clip(pred.numpy(), -1.0, 1.0)), dtype=tf.float32)
    if mezcla < 1.0:
        pred = mezcla * pred + (1.0 - mezcla) * tf.cast(tensor, tf.float32)
    return sharpen(pred, nitidez) if nitidez > 0 else pred


def restore_image(
    generator: tf.keras.Model,
    image: Image.Image,
    size: int,
    *,
    quitar_puntos: bool = True,
    tta: bool = True,
    nitidez: float = 0.0,
    mezcla: float = 1.0,
) -> Image.Image:
    """Restaura una imagen PIL de cualquier tamaño y formato.

    La salida tiene el tamaño de la entrada ya orientada (y acotada a `MAX_SIDE`).
    `quitar_puntos=False, tta=False` usa solo el generador, una pasada.
    """
    prepared = prepare_input(image, size)
    pred = predict(
        generator,
        prepared.tensor,
        quitar_puntos=quitar_puntos,
        tta=tta,
        nitidez=nitidez,
        mezcla=mezcla,
    )
    square = tensor_to_image(pred).resize((prepared.side, prepared.side), Image.Resampling.BICUBIC)
    return square.crop(prepared.box)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restaura una imagen con el generador Pix2Pix.")
    parser.add_argument(
        "--checkpoint", required=True, help="Carpeta checkpoints/ o prefijo ckpt-N."
    )
    parser.add_argument("--config", default=None, help="YAML de entrenamiento (opcional).")
    parser.add_argument("--input", required=True, help="Imagen a restaurar.")
    parser.add_argument("--output", required=True, help="Ruta del PNG restaurado.")
    parser.add_argument(
        "--sin-limpieza", action="store_true", help="No quitar los puntos de polvo."
    )
    parser.add_argument("--sin-tta", action="store_true", help="Una sola pasada, sin volteos.")
    parser.add_argument(
        "--nitidez", type=float, default=0.0, help="Máscara de enfoque (0 = no; 0.5 = media)."
    )
    args = parser.parse_args(argv)

    generator, config = load_generator(args.checkpoint, args.config)
    with Image.open(args.input) as image:
        restored = restore_image(
            generator,
            image,
            config.image_size,
            quitar_puntos=not args.sin_limpieza,
            tta=not args.sin_tta,
            nitidez=args.nitidez,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    restored.save(output)
    print(f"[infer] Imagen restaurada en {output}")


if __name__ == "__main__":
    main()
