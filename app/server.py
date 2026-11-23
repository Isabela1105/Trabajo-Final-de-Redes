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
import os
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.gan.infer import load_generator, restore_image

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


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
) -> FastAPI:
    """Fábrica de la app. Los tests inyectan un generador pequeño sin checkpoint."""
    holder = ModelHolder(checkpoint, generator=generator, image_size=image_size)
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

    @api.post("/api/restore")
    async def restore(file: UploadFile = File(...)) -> Response:  # noqa: B008
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="El archivo está vacío.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="La imagen supera 20 MB.")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=415, detail="No es una imagen válida.") from error

        generator = holder.get()
        restored = restore_image(generator, image, holder.image_size)
        buffer = io.BytesIO()
        restored.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

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
