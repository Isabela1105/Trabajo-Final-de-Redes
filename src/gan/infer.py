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
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.gan.config import Config, load_config


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
    saved = Path(checkpoint_prefix).parent / "config.yaml"
    if not saved.exists():
        raise FileNotFoundError(
            f"No se encontró {saved}. Pase --config con el YAML usado para entrenar."
        )
    return load_config(saved, fixed_seed=False)


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


def image_to_tensor(image: Image.Image, size: int) -> tf.Tensor:
    """PIL → tensor (1, size, size, 3) float32 en [-1, 1]."""
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return tf.convert_to_tensor(array[np.newaxis])


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """Tensor (1, H, W, 3) o (H, W, 3) en [-1, 1] → PIL RGB uint8."""
    array = tf.convert_to_tensor(tensor, dtype=tf.float32)
    if array.shape.rank == 4:
        array = array[0]
    array = np.clip((array.numpy() + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def restore_image(generator: tf.keras.Model, image: Image.Image, size: int) -> Image.Image:
    """Restaura una imagen PIL y la devuelve con su tamaño original."""
    original_size = image.size
    pred = generator(image_to_tensor(image, size), training=False)
    return tensor_to_image(pred).resize(original_size, Image.Resampling.BICUBIC)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Restaura una imagen con el generador Pix2Pix.")
    parser.add_argument(
        "--checkpoint", required=True, help="Carpeta checkpoints/ o prefijo ckpt-N."
    )
    parser.add_argument("--config", default=None, help="YAML de entrenamiento (opcional).")
    parser.add_argument("--input", required=True, help="Imagen a restaurar.")
    parser.add_argument("--output", required=True, help="Ruta del PNG restaurado.")
    args = parser.parse_args(argv)

    generator, config = load_generator(args.checkpoint, args.config)
    with Image.open(args.input) as image:
        restored = restore_image(generator, image, config.image_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    restored.save(output)
    print(f"[infer] Imagen restaurada en {output}")


if __name__ == "__main__":
    main()
