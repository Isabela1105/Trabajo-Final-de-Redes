# Demo de restauración — ejecución y despliegue

Responsable: Juan Sebastian Castillo Acevedo (`@Juansec01`).

La demo principal es una página web propia servida por **FastAPI** (`app/server.py` + `web/`).
Gradio (`app/gradio_app.py`) queda como respaldo. Ninguna de las dos necesita internet: el
modelo se carga desde una carpeta local y la página no usa CDNs.

## 1. Qué se necesita

| Elemento | Dónde |
|---|---|
| Modelo vigente: `pix2pix_100ep_mejor.zip` (generador de la época 48, ≈ 210 MB) | Release del repositorio → descomprimir en la raíz: `outputs/pix2pix_100ep_mejor/best/` |
| Modelo de la entrega `v1.0.0` (opcional) | `pix2pix_20ep.zip` del release `v1.0.0` → `outputs/base/checkpoints/` |
| Dependencias | `pip install -r requirements.txt` |
| Ejemplos del selector (opcional) | `data/processed/test/input/` (se genera con `scripts/build_dataset.py`) o `web/examples/` |

Los pesos **no** se suben a git (`.gitignore` bloquea `outputs/` y `checkpoints/`).

## 2. Ejecutar en local (plan A de la defensa)

```bash
# 1. Descomprimir pix2pix_100ep_mejor.zip en la raíz del repositorio
# 2. Levantar el servidor
python -m app.server --checkpoint outputs/pix2pix_100ep_mejor/best
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
| `GET` | `/api/examples/{nombre}/original` | Original sin daño del ejemplo (si existe `target/`). |
| `POST` | `/api/restore` | `multipart/form-data` con `file` (y opcionales `example`, `limpiar` —`true` por defecto— y `nitidez` —`false` por defecto—); responde `image/png`. Con `example`, la cabecera `X-Metricas` trae PSNR/SSIM de entrada y restaurada frente a la original. |

Errores: `400` archivo vacío, `413` más de 20 MB, `415` no es imagen, `503` sin modelo.

Prueba rápida sin navegador:

```bash
curl -F "file=@foto.jpg" http://127.0.0.1:8000/api/restore -o restaurada.png
```

## 4. Qué pasa con la imagen

1. Se corrige la orientación EXIF y la transparencia se compone sobre blanco.
2. Si mide más de 4096 px de lado, se reduce antes de procesar (protege la memoria).
3. Se rellena por reflexión hasta quedar cuadrada, se lleva a 256×256 y se normaliza a `[-1, 1]`.
4. Si la casilla *Quitar puntos de polvo* está marcada (por defecto), se borran las motas
   negras y blancas aisladas (`src/gan/despeckle.py`).
5. El generador predice con **TTA**: restaura la foto y sus 3 versiones volteadas y promedia
   (menos ruido y artefactos; +0.16 dB y +0.016 SSIM en test). Se vuelve a limpiar el polvo.
6. Si la casilla *Realzar nitidez* está marcada, se aplica una máscara de enfoque (0.5). Se ve
   más definida, pero baja PSNR/SSIM porque también realza el ruido.
7. La salida se desnormaliza, se recorta el relleno y vuelve al tamaño de entrada (acotado).

Así una foto vertical de celular no sale estirada ni rotada.

## 5. Ejecutar en Colab (plan B)

```python
!git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
%cd Trabajo-Final-de-Redes
!pip -q install fastapi uvicorn python-multipart gradio
# Gradio crea un enlace público temporal, útil si el portátil de la defensa falla:
!python -m app.gradio_app --checkpoint /content/drive/MyDrive/<ruta>/checkpoints
```

## 6. Plan C

Si no hay ni portátil ni Colab: la galería precalculada `docs/img/results_grid.png` y
`docs/img/failures.png` en las diapositivas, más un video corto de la demo grabado antes.

## 7. Comparación con la original

Si existe `data/processed/test/` (descomprimiendo `datasetv1.zip`), el selector muestra las 45
fotos de prueba. Al elegir una, o al arrastrar un archivo con el mismo nombre, la página agrega
una tercera columna **Original (referencia)** y muestra el PSNR y el SSIM de la entrada y de la
restaurada frente a la original. Así se ve cuánto mejora el modelo en cada foto.

## 8. Limitaciones

- El modelo trabaja a 256×256: en fotos grandes la salida se ve más suave que la original.
- Puede **inventar** detalles (caras, texto) que no existían; no sirve como prueba documental.
- Solo conoce las degradaciones del dataset (`docs/DATASET.md`); daños muy distintos (manchas
  de agua, partes faltantes grandes) no se corrigen bien.
- La limpieza quita la mayoría de los puntos de polvo, no siempre todos: las motas de 1 px y
  los rayones (líneas) se dejan para no borrar detalles reales. Ver `docs/RESULTADOS.md` §2.5.
