# Test del modelo vigente (Pix2Pix 100 épocas, mejor validación en la época 48)

Generado con:

```bash
python -m src.gan.evaluate --config configs/base.yaml \
  --checkpoint outputs/pix2pix_100ep_mejor/best \
  --name "Pix2Pix 100 épocas (mejor val., ép. 48)" --tta \
  --model "Pix2Pix 20 épocas (v1.0.0)=outputs/base/checkpoints" \
  --model "U-Net solo L1 (20 épocas)=outputs/ablation/checkpoints"
python scripts/error_analysis.py --per-image per_image.csv \
  --model "Pix2Pix 100 épocas (mejor val., ép. 48) + TTA"
```

- `summary.csv`, `per_image.csv`, `by_degradation.csv`: salida de `src.gan.evaluate`.
- `analisis_errores.md`: salida de `scripts/error_analysis.py`.
- `colab_salida.txt`: corridas de 100 épocas en Colab (T4). La #1 completó 100 épocas; la #2
  llegó a la época 70 y su modelo `best/` se perdió porque el Drive se llenó (ver
  `docs/RESULTADOS.md` §6).
- Curvas del modelo vigente: `docs/resultados/pix2pix_100ep_metrics.csv` y
  `docs/img/curvas_100ep.png`.
