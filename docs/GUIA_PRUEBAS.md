# Guía para probar el proyecto

Cuatro niveles, de menos a más exigente. Los dos primeros no necesitan GPU ni datos reales.

## Nivel 1 — Pruebas automáticas (5 min, CPU)

```bash
git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
cd Trabajo-Final-de-Redes
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

ruff check . && black --check .
pytest -q
```

Esperado: `ruff` y `black` sin errores y todas las pruebas en verde (unas 187, ~1 min).

| Archivo | Qué comprueba |
|---|---|
| `tests/test_metrics.py` | PSNR/SSIM/L1 con valores conocidos; imagen idéntica → SSIM 1, L1 0; FID. |
| `tests/test_infer.py` | Carga de checkpoint, mismo resultado al cargar dos veces, fotos de 12 MP, PNG transparente, EXIF, modos raros. |
| `tests/test_evaluate.py` | Filtro de mediana, resumen y desglose, CLI completa con CSVs y figuras. |
| `tests/test_app.py` | API de la demo: PNG de salida, errores 400/415/503, ejemplos. |
| `tests/test_synthetic_sample.py` | Muestra sintética + entrenamiento de 2 pasos (lo mismo que el CI). |
| resto | Dataset, modelos, pérdidas y entrenamiento (de los otros integrantes). |

## Nivel 2 — Pipeline completo con datos falsos (3 min, CPU)

Prueba que entrenar → evaluar → tabla → demo funciona, aunque el modelo salga malo.

```bash
python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
python -m src.gan.evaluate --config configs/smoke.yaml \
  --checkpoint outputs/smoke/checkpoints --out outputs/smoke/eval
python scripts/results_table.py --summary outputs/smoke/eval/summary.csv
python -m src.gan.infer --checkpoint outputs/smoke/checkpoints \
  --input data/sample/val/input/sample_00.png --output outputs/smoke/restaurada.png
python -m app.server --checkpoint outputs/smoke/checkpoints
```

Abre <http://127.0.0.1:8000>: el indicador debe decir *Modelo se carga al restaurar*, al subir
una imagen se ve la entrada y la restaurada, y aparece *Descargar PNG*. La imagen saldrá gris o
ruidosa: es normal, el modelo de humo solo entrenó 2 pasos.

Prueba de la API sin navegador:

```bash
curl http://127.0.0.1:8000/api/health
curl -F "file=@data/sample/val/input/sample_00.png" http://127.0.0.1:8000/api/restore -o r.png
```

## Nivel 3 — Con el modelo entrenado (Colab o PC)

1. Descarga del Drive la carpeta `checkpoints/` de la corrida base
   (`pix2pix_outputs/metricas_loss_existente/checkpoints` o
   `Redes-Pix2Pix/resultados_pix2pix/base/checkpoints`) y cópiala a `outputs/base/checkpoints/`.
   Debe contener `ckpt-N.index`, `ckpt-N.data-*`, `checkpoint`, `config.yaml` y `metadata.json`.
2. Descomprime `datasetv1.zip` para que quede `data/processed/{train,val,test}/{input,target}`.
3. Evalúa y publica la tabla:

   ```bash
   make eval        # = python -m src.gan.evaluate --config configs/base.yaml --checkpoint outputs/base/checkpoints
   make results     # inserta la tabla en README, informe y docs/RESULTADOS.md y copia las figuras
   ```

4. Demo real: `make demo` y abre <http://127.0.0.1:8000>. Elige una foto del selector (son las
   entradas del split de prueba) y pulsa **Restaurar**. Toma la captura → `docs/img/app.png`.

Qué revisar en el resultado:

- En `outputs/base/eval/summary.csv`, Pix2Pix debe superar a *Entrada (sin restaurar)* en PSNR y
  SSIM. Si no lo hace, el modelo no aporta: revisa que se cargó el checkpoint correcto.
- `failures.png` muestra los peores casos; úsalos en la defensa para hablar de limitaciones.
- `by_degradation.csv` dice qué daño corrige mejor y peor.

Si no tienes el entorno local, `notebooks/evaluacion_y_demo_colab.ipynb` hace estos pasos en
Colab y te descarga los archivos para subirlos.

## Nivel 4 — Casos difíciles para la demo

| Caso | Cómo probar | Esperado |
|---|---|---|
| Foto de celular de 12 MP | subir una foto original del teléfono | responde en segundos, mismo tamaño de salida |
| Foto vertical con EXIF | foto tomada en vertical | sale derecha, no acostada |
| PNG transparente | logo con fondo transparente | fondo blanco, no negro |
| Escala de grises | foto en blanco y negro | se procesa como RGB |
| Archivo que no es imagen | subir un `.txt` | mensaje de error, código 415 |
| Sin checkpoint | `python -m app.server` sin `--checkpoint` | indicador *Sin checkpoint configurado*, error 503 al restaurar |

## Entrenar de nuevo o correr la ablación (Colab, GPU)

`notebooks/pix2pix_base_y_ablacion_colab.ipynb`:

1. Sube `datasetv1.zip` a `MyDrive/Redes-Pix2Pix/` (o a `/content`).
2. *Entorno de ejecución → Cambiar tipo → GPU*.
3. Ejecuta las celdas 1–2 (preparación), 3 (base) y/o 4 (ablación). Las salidas van a Drive.
4. Si Colab se desconecta: pon `RESUME = True` (o `RESUME_ABL = True`) y vuelve a ejecutar la
   celda; continúa desde el último checkpoint.
5. Luego evalúa con `--model "U-Net solo L1=<ruta>/ablation/checkpoints"` para la fila de la
   ablación.
