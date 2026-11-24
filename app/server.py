"""Demo web de restauración: FastAPI + `web/` (HTML, CSS y JS).

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

Flujo de punta a punta:
    navegador --(POST /api/restore, multipart "file")--> servidor
    servidor: abre la imagen -> normaliza a [-1, 1] -> G(x) -> desnormaliza -> PNG
    servidor --(image/png)--> navegador

No necesita internet: el modelo se carga desde una carpeta local de checkpoints y la
página no usa CDNs. Gradio queda como respaldo en `app/gradio_app.py`.

Uso:
    python -m app.server --checkpoint outputs/base/checkpoints
    # o con variable de entorno
    PIX2PIX_CHECKPOINT=outputs/base/checkpoints uvicorn app.server:app --port 8000
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.gan.data.dataset import to_rgb
from src.gan.infer import MAX_SIDE, load_generator, restore_image
from src.gan.metrics import psnr, ssim

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
# Ejemplos del selector: primero las entradas degradadas del split de prueba (se generan con
# scripts/build_dataset.py y no se suben a git); si no existen, web/examples/.
DEFAULT_EXAMPLE_DIRS = (REPO_ROOT / "data" / "processed" / "test" / "input", WEB_DIR / "examples")
EXAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_EXAMPLES = 60  # alcanza para los 45 pares de test


def find_examples(directories: tuple[Path, ...] | list[Path]) -> dict[str, Path]:
    """Primer directorio con imágenes -> {nombre: ruta}, ordenado y acotado."""
    for directory in directories:
        directory = Path(directory)
        if directory.is_dir():
            files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXAMPLE_SUFFIXES)
            if files:
                return {p.name: p for p in files[:MAX_EXAMPLES]}
    return {}


def find_originals(examples: dict[str, Path]) -> dict[str, Path]:
    """Para cada ejemplo `.../input/x.png`, su original `.../target/x.png` si existe."""
    originals = {}
    for name, path in examples.items():
        target = path.parent.parent / "target" / name
        if path.parent.name == "input" and target.is_file():
            originals[name] = target
    return originals


def compare_to_original(image: Image.Image, original: Image.Image) -> dict[str, float]:
    """PSNR y SSIM en [0, 1] contra la original (llevada al tamaño de `image`)."""
    a = np.asarray(to_rgb(image), dtype=np.float32) / 255.0
    ref = to_rgb(original)
    if ref.size != image.size:
        ref = ref.resize(image.size, Image.Resampling.BICUBIC)
    b = np.asarray(ref, dtype=np.float32) / 255.0
    return {"psnr": round(float(psnr(a, b)[0]), 2), "ssim": round(float(ssim(a, b)[0]), 4)}


class ModelHolder:
    """Carga perezosa del generador: el servidor arranca aunque aún no haya checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        generator: tf.keras.Model | None = None,
        image_size: int = 256,
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.generator = generator
        self.image_size = image_size
        self.error: str | None = None

    def get(self) -> tf.keras.Model:
        if self.generator is None:
            if not self.checkpoint:
                raise HTTPException(
                    status_code=503,
                    detail="No hay modelo cargado. Inicie con --checkpoint o PIX2PIX_CHECKPOINT.",
                )
            try:
                self.generator, config = load_generator(self.checkpoint)
                self.image_size = config.image_size
            except (FileNotFoundError, AssertionError, ValueError) as error:
                self.error = str(error)
                raise HTTPException(
                    status_code=503, detail=f"No se pudo cargar: {error}"
                ) from error
        return self.generator


def create_app(
    checkpoint: str | Path | None = None,
    *,
    generator: tf.keras.Model | None = None,
    image_size: int = 256,
    examples_dirs: tuple[Path, ...] | list[Path] = DEFAULT_EXAMPLE_DIRS,
) -> FastAPI:
    """Fábrica de la app. Los tests inyectan un generador pequeño sin checkpoint."""
    holder = ModelHolder(checkpoint, generator=generator, image_size=image_size)
    examples = find_examples(examples_dirs)
    originals = find_originals(examples)
    api = FastAPI(title="Restauración de fotografías con Pix2Pix", version="1.0.0")

    @api.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model_loaded": holder.generator is not None,
                "checkpoint": holder.checkpoint,
                "image_size": holder.image_size,
                "error": holder.error,
            }
        )

    @api.get("/api/examples")
    def list_examples() -> JSONResponse:
        return JSONResponse(
            [
                {
                    "name": name,
                    "url": f"/api/examples/{name}",
                    "original_url": f"/api/examples/{name}/original" if name in originals else None,
                }
                for name in sorted(examples)
            ]
        )

    @api.get("/api/examples/{name}")
    def get_example(name: str) -> FileResponse:
        # Solo se sirven archivos del índice: evita rutas como ../../secretos.
        if name not in examples:
            raise HTTPException(status_code=404, detail="Ejemplo no encontrado.")
        return FileResponse(examples[name])

    @api.get("/api/examples/{name}/original")
    def get_original(name: str) -> FileResponse:
        if name not in originals:
            raise HTTPException(status_code=404, detail="Este ejemplo no tiene original.")
        return FileResponse(originals[name])

    @api.post("/api/restore")
    async def restore(
        file: UploadFile = File(...),  # noqa: B008
        example: str | None = Form(None),  # noqa: B008
    ) -> Response:
        """Devuelve el PNG restaurado.

        Si `example` es un ejemplo del test con original, agrega la cabecera `X-Metricas`
        (JSON con PSNR y SSIM de la entrada y de la restaurada frente a la original).
        """
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="El archivo está vacío.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="La imagen supera 20 MB.")
        try:
            image = Image.open(io.BytesIO(data))
            # JPEG grandes: decodificar directamente a una escala reducida (mucho más rápido).
            image.draft("RGB", (MAX_SIDE, MAX_SIDE))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(status_code=415, detail="No es una imagen válida.") from error

        generator = holder.get()
        restored = restore_image(generator, image, holder.image_size)
        buffer = io.BytesIO()
        restored.save(buffer, format="PNG")
        headers = {}
        if example and example in originals:
            with Image.open(originals[example]) as original:
                original.load()
                metrics = {
                    "entrada": compare_to_original(image, original),
                    "restaurada": compare_to_original(restored, original),
                }
            headers["X-Metricas"] = json.dumps(metrics)
        return Response(content=buffer.getvalue(), media_type="image/png", headers=headers)

    @api.middleware("http")
    async def no_cache_web(request, call_next):  # noqa: ANN001, ANN202
        """La página y sus archivos se revalidan siempre: tras un `git pull` el navegador
        nunca mezcla un HTML nuevo con un JavaScript viejo guardado en caché."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    if WEB_DIR.is_dir():
        api.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @api.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return api


# Instancia para `uvicorn app.server:app`.
app = create_app(os.environ.get("PIX2PIX_CHECKPOINT"))


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Demo web de Pix2Pix.")
    parser.add_argument("--checkpoint", default=os.environ.get("PIX2PIX_CHECKPOINT"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
