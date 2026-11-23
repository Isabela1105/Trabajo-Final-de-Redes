# Experimento de nitidez

Responsable: Juan Sebastian Castillo Acevedo (`@Juansec01`). Posterior a la entrega `v1.0.0`:
no cambia `configs/base.yaml`, `src/gan/losses.py`, la tabla de resultados ni el release.

## Problema

El modelo entregado corrige color, contraste y ruido, pero deja la imagen **borrosa**
(`docs/RESULTADOS.md` §2.3: desenfoque +0.4 dB, baja resolución +0.8 dB). Tres causas:

1. **La L1 promedia.** Ante varias reconstrucciones posibles, el mínimo de L1 es una mezcla
   suave de todas. Con λ = 100 domina la pérdida del generador.
2. **El discriminador gana demasiado pronto.** Su pérdida baja de 0.04 desde la época 2;
   la señal adversarial, que es la que pide texturas reales, casi desaparece.
3. **Pocas épocas.** 20 de las 100 previstas; las curvas aún bajaban.

## Cambios (`configs/nitidez.yaml`)

| Parámetro | Base | Nitidez | Para qué |
|---|---|---|---|
| Épocas | 20 (corrida evaluada) | 100 | Dejar que converja. |
| `losses_module` | `src.gan.losses` | `src.gan.losses_nitidez` | Añade la pérdida perceptual y el etiquetado suave. |
| Pérdida perceptual | — | λ = 10, VGG16 `block3_conv3` | Castiga bordes y texturas perdidos, no solo píxeles. |
| `lambda_l1` | 100 | 50 | Menos peso al "promedio" borroso. |
| `label_smoothing` | 0 (la pérdida original lo ignora) | 0.1 | D apunta a 0.9 en lo real: no domina. |
| `lr_d` | 2e-4 | 1e-4 | D aprende más lento que G. |

**Pérdida perceptual** (`src/gan/losses_nitidez.py`): las imágenes pasan por VGG16
preentrenada en ImageNet (congelada, en float32) y se compara la capa `block3_conv3`:

$$\mathcal{L}_{perc} = \frac{\lVert \phi(G(x)) - \phi(y) \rVert_1}{\overline{|\phi(y)|}}$$

La división por la magnitud media deja el valor en el rango de la L1 de píxeles, así λ = 10
pesa parecido a λ_L1 · L1. Pruebas en `tests/test_losses_nitidez.py` (sin descargar pesos).

## Cómo correrlo (Colab con GPU T4, ~1 h a 1 h 15 min)

1. Abrir `notebooks/pix2pix_colab.ipynb` desde GitHub y elegir **GPU T4**.
2. Sección 0: `EXPERIMENTO_NITIDEZ = True` (deja `ENTRENAR_BASE` y `ENTRENAR_ABLACION` en `False`).
3. *Ejecutar todas*. Si Colab se desconecta: `RESUME = True` y volver a ejecutar la sección 3b.
4. La sección **5b** compara Entrada, Mediana, U-Net solo L1, Pix2Pix base y Pix2Pix nitidez en
   `outputs/eval_nitidez/`, sin tocar los resultados de la entrega.
5. La sección 7 descarga un zip con `eval_nitidez/` y las métricas de la corrida.

En local (sin GPU, ~8 h): `python -m src.gan.train --config configs/nitidez.yaml`.

## Qué esperar y cómo juzgarlo

- **Visualmente** más nitidez y textura en `results_grid.png` de `eval_nitidez`.
- **PSNR/SSIM pueden bajar un poco**: premian la fidelidad píxel a píxel, y la textura
  generada no coincide exactamente con la original. Por eso se juzga mirando las imágenes y,
  si se quiere un número perceptual, con `--fid`.
- **Riesgo**: detalles inventados (alucinación). Revisar `failures.png` y las fotos reales.

## Resultado

<!-- NITIDEZ:INICIO -->
_Pendiente de correr en Colab._
<!-- NITIDEZ:FIN -->
