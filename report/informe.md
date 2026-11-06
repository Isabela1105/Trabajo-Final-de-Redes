# Restauración de fotografías con Pix2Pix

**Curso:** Redes Neuronales y Deep Learning
**Universidad:** Universidad Autónoma de Occidente
**Integrantes:** Isabella Ortiz, Juan Pablo D�vila, Jeremy Caicedo, Cristian Mu�oz, Juan Sebastian Castillo, Isabela
**Repositorio:** `<URL>`
**Tag de resultados:** `v1.0.0`

---

## Resumen

<!-- 150–250 palabras. Problema, método, dataset, resultados principales, limitación. -->

Escribir al final, cuando estén las métricas.

---

## 1. Introducción

### 1.1 Motivación

<!-- Por qué la restauración de fotografías es relevante. -->

### 1.2 Problema

<!-- Fotos con ruido, blur, JPEG, rayones, color desvanecido. -->

### 1.3 Objetivos

**General:** entrenar una GAN condicional para restaurar fotografías degradadas.

**Específicos:**

1. Construir un dataset propio de pares degradado/limpio.
2. Implementar Pix2Pix (U-Net + PatchGAN) en TensorFlow/Keras.
3. Entrenar y validar con PSNR, SSIM y L1.
4. Desplegar una demo de restauración.

### 1.4 Aportes

<!-- Dataset, pipeline reproducible, evaluación con baselines, demo. -->

---

## 2. Marco teórico

### 2.1 Redes generativas adversariales

<!-- GAN, generador, discriminador, objetivo minimax. -->

### 2.2 Pix2Pix

<!-- GAN condicional, pares, pérdida adversarial + L1. -->

### 2.3 U-Net

<!-- Encoder-decoder, skip connections. -->

### 2.4 PatchGAN

<!-- Discriminador por parches, 70x70. -->

### 2.5 Métricas

<!-- PSNR, SSIM, L1; qué mide cada una y sus límites. -->

---

## 3. Dataset

### 3.1 Recolección

### 3.2 Degradaciones sintéticas

### 3.3 Splits y prevención de fuga

### 3.4 Estadísticas

### 3.5 Ética, consentimiento y licencias

### 3.6 Limitaciones

---

## 4. Arquitectura

### 4.1 Contratos de tensores

### 4.2 Generador U-Net

### 4.3 Discriminador PatchGAN

### 4.4 Función de pérdida

### 4.5 Hiperparámetros

---

## 5. Entrenamiento

### 5.1 Hardware y entorno

### 5.2 Configuración

### 5.3 Checkpoints y reanudación

### 5.4 Estabilidad

### 5.5 Curvas de pérdida

---

## 6. Experimentos y resultados

### 6.1 Protocolo

### 6.2 Baselines

### 6.3 Resultados cuantitativos

| Modelo | PSNR ↑ | SSIM ↑ | L1 ↓ |
|---|---|---|---|
| Entrada | — | — | — |
| Filtro clásico | — | — | — |
| U-Net solo L1 | — | — | — |
| Pix2Pix (λ=100) | — | — | — |

### 6.4 Resultados cualitativos

### 6.5 Galería de fallos

### 6.6 Discusión de resultados

---

## 7. Aplicación y demo

### 7.1 Cómo ejecutar

### 7.2 Capturas

### 7.3 Limitaciones de uso

---

## 8. Discusión

### 8.1 Limitaciones

### 8.2 Riesgo de alucinación

### 8.3 Consideraciones éticas

### 8.4 Trabajo futuro

---

## 9. Conclusiones

<!-- Responder a los objetivos con evidencia. -->

---

## 10. Referencias

<!-- Generadas desde referencias.bib con Pandoc. -->

---

## Anexo A — Participación

<!-- Enlazar docs/PARTICIPACION.md y resumir la tabla. -->

## Anexo B — Estructura del repositorio

## Anexo C — Instrucciones de reproducción

```bash
pip install -r requirements-dev.txt
python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42
python -m src.gan.train --config configs/base.yaml
python -m src.gan.evaluate --config configs/base.yaml --checkpoint <ruta> --split test
python app/app.py
```

## Anexo D — Reflexiones individuales

<!-- Opcional, si el curso lo pide. -->
