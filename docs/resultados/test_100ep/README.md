# Test con la corrida de 100 épocas (Colab T4, 24 nov 2026)

- `summary.csv`: tabla de README e informe. Las tres primeras filas y la de 20 épocas salen de
  `docs/resultados/test/summary.csv`; la fila de 100 épocas es la salida de
  `python -m src.gan.evaluate --checkpoint pix2pix_100ep_v2/best` en Colab (cifras impresas,
  redondeadas a 2 y 4 decimales).
- `colab_salida.txt`: salida completa de entrenamiento, evaluación y `scripts/error_analysis.py`.
