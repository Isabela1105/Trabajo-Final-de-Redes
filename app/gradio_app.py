"""Demo de respaldo con Gradio (plan B de la defensa).

Autor: Juan Sebastian Castillo Acevedo (@Juansec01)

La demo principal es `app/server.py` (FastAPI + web/). Esta versión existe por si el
navegador o el servidor fallan en la sustentación: usa exactamente la misma función
`restore_image` de `src/gan/infer.py`.

Uso:
    python -m app.gradio_app --checkpoint outputs/base/checkpoints
"""

from __future__ import annotations

import argparse
import os


def build_interface(checkpoint: str):
    import gradio as gr

    from src.gan.infer import load_generator, restore_image

    generator, config = load_generator(checkpoint)

    def run(image, limpiar):
        if image is None:
            return None
        return restore_image(generator, image, config.image_size, quitar_puntos=limpiar)

    return gr.Interface(
        fn=run,
        inputs=[
            gr.Image(type="pil", label="Foto degradada"),
            gr.Checkbox(value=True, label="Quitar puntos de polvo"),
        ],
        outputs=gr.Image(type="pil", label="Restaurada"),
        title="Restauración de fotografías con Pix2Pix",
        description="U-Net + PatchGAN entrenados con el dataset propio del curso.",
        allow_flagging="never",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo Gradio (respaldo).")
    parser.add_argument("--checkpoint", default=os.environ.get("PIX2PIX_CHECKPOINT"))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        parser.error("Indique --checkpoint o PIX2PIX_CHECKPOINT.")
    build_interface(args.checkpoint).launch(server_port=args.port)


if __name__ == "__main__":
    main()
