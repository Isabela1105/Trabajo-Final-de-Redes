# Demo de restauración — ejecución y despliegue

Responsable: Juan Sebastian Castillo Acevedo (`@Juansec01`).

La demo principal es una página web propia servida por **FastAPI** (`app/server.py` + `web/`).
Gradio (`app/gradio_app.py`) queda como respaldo. Ninguna de las dos necesita internet: el
modelo se carga desde una carpeta local y la página no usa CDNs.

## 1. Qué se necesita

| Elemento | Dónde |
|---|---|
| Carpeta `checkpoints/` del entrenamiento (`ckpt-N.index`, `ckpt-N.data-*`, `config.yaml`, `metadata.json`) | Drive del equipo → `outputs/base/checkpoints/` |
| Dependencias | `pip install -r requirements.txt` |
| Ejemplos del selector (opcional) | `data/processed/test/input/` (se genera con `scripts/build_dataset.py`) o `web/examples/` |

Los pesos **no** se suben a git (`.gitignore` bloquea `outputs/` y `checkpoints/`).

## 2. Ejecutar en local (plan A de la defensa)

```bash
# 1. Copiar la carpeta de checkpoints del Drive a outputs/base/checkpoints/
# 2. Levantar el servidor
python -m app.server --checkpoint outputs/base/checkpoints
# 3. Abrir http://127.0.0.1:8000
```

Equivalente con uvicorn:

```bash
PIX2PIX_CHECKPOINT=outputs/base/checkpoints uvicorn app.server:app --port 8000
```

En Windows PowerShell:

```powershell
$env:PIX2PIX_CHECKPOINT = "outputs/base/checkpoints"
uvicorn app.server:app --port 8000
```

## 3. API

| Método | Ruta | Qué hace |
|---|---|---|
| `GET` | `/` | Página `web/index.html`. |
| `GET` | `/api/health` | `{"model_loaded", "checkpoint", "image_size", "error"}`. |
| `GET` | `/api/examples` | Lista de ejemplos del selector. |
| `GET` | `/api/examples/{nombre}` | Imagen de ejemplo (solo archivos del índice). |
| `POST` | `/api/restore` | `multipart/form-data` con `file`; responde `image/png` restaurada. |

Errores: `400` archivo vacío, `413` más de 20 MB, `415` no es imagen, `503` sin modelo.

Prueba rápida sin navegador:

```bash
curl -F "file=@foto.jpg" http://127.0.0.1:8000/api/restore -o restaurada.png
```

## 4. Ejecutar en Colab (plan B)

```python
!git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
%cd Trabajo-Final-de-Redes
!pip -q install fastapi uvicorn python-multipart gradio
# Gradio crea un enlace público temporal, útil si el portátil de la defensa falla:
!python -m app.gradio_app --checkpoint /content/drive/MyDrive/<ruta>/checkpoints
```

## 5. Plan C

Si no hay ni portátil ni Colab: la galería precalculada `docs/img/results_grid.png` y
`docs/img/failures.png` en las diapositivas, más un video corto de la demo grabado antes.

## 6. Limitaciones

- El modelo trabaja a 256×256: en fotos grandes la salida se ve más suave que la original.
- Puede **inventar** detalles (caras, texto) que no existían; no sirve como prueba documental.
- Solo conoce las degradaciones del dataset (`docs/DATASET.md`); daños muy distintos (manchas
  de agua, partes faltantes grandes) no se corrigen bien.
