# Comparación de modelos y arquitectura elegida

## 1. Tabla de decisión

| Modelo | Tipo | Datos que necesita | Ventajas | Riesgos | ¿Sirve para este trabajo? |
|---|---|---|---|---|---|
| **Pix2Pix** | GAN condicional *paired* | Pares (entrada, salida) | Resultados nítidos, métricas objetivas, arquitectura explicable, tutorial oficial TF | Requiere pares; puede "inventar" detalles | **Sí — elegido** |
| CycleGAN | GAN no pareada | Dos dominios sin pares | No requiere pares; ideal para estilo | Entrenamiento largo, inestable, evaluación subjetiva | Sí, pero más riesgoso |
| SRGAN / ESRGAN | GAN de super-resolución | Pares LR/HR | Métricas perceptuales; buen resultado visual | Pérdida perceptual y arquitectura más compleja | Sí, como variante de super-resolución |
| DCGAN | GAN no condicional | Solo imágenes | Simple, buena base teórica | Genera desde ruido; no restaura una entrada | No cumple el objetivo |
| StyleGAN2 | GAN de estilo | Miles de imágenes | Calidad altísima | Mucha GPU, no condiciona una foto concreta | No recomendado para 4 semanas |
| Audio GAN (WaveGAN/GANSynth) | GAN de audio | Audio | Tema alternativo | Evaluación y demo más difíciles; menos visual | Solo si el grupo domina audio |

**Decisión:** Pix2Pix para restauración de fotografías.

**Por qué esta combinación gana para 6 personas y ~4 semanas:**

1. El dataset se puede **construir** de forma determinista: se toma una foto limpia y se degrada.
   Así se obtiene el par y el *ground truth*.
2. Permite **evaluación objetiva** (PSNR, SSIM, L1) además de la visual.
3. La arquitectura se explica en tres piezas: **U-Net**, **PatchGAN** y **pérdida adversarial + L1**.
4. Se puede entrenar en Colab/Kaggle sin infraestructura propia.
5. El tema no está prohibido y es visualmente atractivo para la sustentación.

---

## 2. Formulación de Pix2Pix

Dado un par $(x, y)$:

- $x$: imagen de entrada (foto degradada).
- $y$: imagen objetivo (foto limpia).
- $G$: generador.
- $D$: discriminador.

### Pérdida del discriminador

$$
\mathcal{L}_D = \frac{1}{2}\left[
\mathbb{E}_{x,y}\left[\log D(x,y)\right] +
\mathbb{E}_{x}\left[\log\left(1 - D(x,G(x))\right)\right]
\right]
$$

En código se usa `BinaryCrossentropy(from_logits=True)` y se separan los dos términos para
poder monitorearlos.

### Pérdida del generador

$$
\mathcal{L}_G = \mathbb{E}_{x}\left[\log\left(1 - D(x,G(x))\right)\right]
+ \lambda \cdot \mathbb{E}_{x,y}\left[\lVert y - G(x) \rVert_1\right]
$$

En la práctica se maximiza $\log D(x,G(x))$ en lugar de minimizar $\log(1-D)$, que tiene
mejor gradiente al inicio. Con `BCEWithLogits`, el término adversarial es
`BCE(D(fake), 1)`.

- $\lambda = 100$: el L1 pesa 100 veces más que el adversarial. Es el valor del paper y del
  tutorial; hace que la salida se parezca a la foto original y que la GAN añada textura.
- Sin L1, el generador puede producir imágenes realistas pero **no fieles** a la entrada.
- Sin adversarial, el generador aprende un promedio borroso (eso es el baseline L1-only).

---

## 3. Generador U-Net

### 3.1 Diagrama textual

```text
Entrada (256,256,3)
  │
  ├─ Encoder 1: Conv 4x4, stride 2, 64  ──────────────── skip 1 (128,128,64)
  ├─ Encoder 2: Conv 4x4, stride 2, 128, IN, LReLU ──── skip 2 (64,64,128)
  ├─ Encoder 3: Conv 4x4, stride 2, 256, IN, LReLU ──── skip 3 (32,32,256)
  ├─ Encoder 4: Conv 4x4, stride 2, 512, IN, LReLU ──── skip 4 (16,16,512)
  ├─ Encoder 5: Conv 4x4, stride 2, 512, IN, LReLU ──── skip 5 (8,8,512)
  ├─ Encoder 6: Conv 4x4, stride 2, 512, IN, LReLU ──── skip 6 (4,4,512)
  ├─ Encoder 7: Conv 4x4, stride 2, 512, IN, LReLU ──── skip 7 (2,2,512)
  └─ Encoder 8: Conv 4x4, stride 2, 512, IN, LReLU ──── skip 8 (1,1,512)
        │
      Bottleneck (1,1,512)
        │
  ├─ Decoder 1: ConvTranspose 4x4, stride 2, 512, IN, Dropout, ReLU
  │             concat con skip 8  → (2,2,1024)
  ├─ Decoder 2: ConvTranspose → 512, concat skip 7 → (4,4,1024)
  ├─ Decoder 3: ConvTranspose → 512, concat skip 6 → (8,8,1024)
  ├─ Decoder 4: ConvTranspose → 512, concat skip 5 → (16,16,1024)
  ├─ Decoder 5: ConvTranspose → 256, concat skip 4 → (32,32,512)
  ├─ Decoder 6: ConvTranspose → 128, concat skip 3 → (64,64,256)
  ├─ Decoder 7: ConvTranspose → 64,  concat skip 2 → (128,128,128)
  └─ Decoder 8: ConvTranspose → 3,   concat skip 1 → (256,256,64)
        │
      Conv 4x4 + tanh → (256,256,3) en [-1,1]
```

> Nota: en la implementación se parametriza `n_downsample` para poder usar 4 niveles en el
> smoke test de 64×64 y 8 niveles en el entrenamiento real de 256×256.

### 3.2 Decisiones de diseño

| Decisión | Elección | Justificación |
|---|---|---|
| Normalización | **InstanceNorm** | Batch size pequeño (1–4); BatchNorm es inestable. Se explica la diferencia con el paper. |
| Activación encoder | LeakyReLU 0.2 | Evita gradientes muertos, estándar en GANs. |
| Activación decoder | ReLU | Estándar en U-Net. |
| Dropout | 0.5 en los primeros 3 bloques del decoder | Regularización y reducción de artefactos. |
| Up-sampling | `Conv2DTranspose` | Aprendizaje de la mejor interpolación; explicable. |
| Salida | `tanh` | Rango [-1,1] consistente con la normalización. |
| Skip connections | Concatenación | Conservan detalle espacial; son la clave de U-Net. |

### 3.3 Tamaño

Con `ngf=64` y 8 niveles: **~54 M parámetros**. Con `ngf=8` y 4 niveles (smoke): **~0.7 M**.
El CI usa la versión pequeña para tardar segundos, no minutos.

---

## 4. Discriminador PatchGAN

### 4.1 Arquitectura

Entrada: concatenación de `x` e `y` (o `x` y `G(x)`) → `(256,256,6)`.

```text
Conv 4x4, stride 2, 64, LReLU              → (128,128,64)
Conv 4x4, stride 2, 128, InstanceNorm, LReLU → (64,64,128)
Conv 4x4, stride 2, 256, InstanceNorm, LReLU → (32,32,256)
ZeroPadding2D                               → (34,34,256)
Conv 4x4, stride 1, 512, InstanceNorm, LReLU → (31,31,512)
ZeroPadding2D                               → (33,33,512)
Conv 4x4, stride 1, 1                       → (30,30,1) logits
```

Cada celda de la salida $30 \times 30$ evalúa un parche de $70 \times 70$ píxeles. Por eso se
llama **PatchGAN**: en vez de decidir "toda la imagen es real/falsa", decide por regiones. Esto
enfatiza texturas y detalles de alta frecuencia y usa menos parámetros que un discriminador
que clasifica la imagen completa.

### 4.2 Por qué PatchGAN

- Más fácil de entrenar con imágenes grandes.
- Penaliza texturas locales, no solo la estructura global.
- Menos parámetros y más rápido.
- Es la elección del paper original de Pix2Pix.

---

## 5. Algoritmo de entrenamiento

```text
para cada época:
    para cada batch (x, y):
        # 1. Actualizar discriminador
        fake = G(x, training=True)
        d_real = D([x, y], training=True)
        d_fake = D([x, fake], training=True)
        loss_d = 0.5 * (BCE(d_real, 1) + BCE(d_fake, 0))
        gradientes y paso de Adam sobre D

        # 2. Actualizar generador
        fake = G(x, training=True)
        d_fake = D([x, fake], training=True)
        loss_g_adv = BCE(d_fake, 1)
        loss_g_l1 = mean(|y - fake|)
        loss_g = loss_g_adv + 100 * loss_g_l1
        gradientes y paso de Adam sobre G

    guardar checkpoint
    cada N épocas: guardar grid de muestras y métricas de validación
```

Detalles:

- **Dos optimizadores separados** (uno para G, uno para D). No compartir.
- En el paso de G, **congelar D** (`trainable = False` o no aplicar sus gradientes).
- `tf.function` para acelerar; `jit_compile=False` para evitar problemas en CPU/Colab.
- **Label smoothing** opcional: real = 0.9 en vez de 1.0. Ayuda a estabilizar.
- **TTUR** opcional: lr_D = 1e-4, lr_G = 2e-4 si D domina.
- **Gradient clipping** opcional: `clip_by_global_norm(1.0)`.
- **Mixed precision** en GPU: `mixed_float16`; mantener la pérdida en `float32`.

### Señales de problemas

| Síntoma | Causa probable | Acción |
|---|---|---|
| `loss_d` → 0 y `loss_g` explota | D demasiado fuerte | Bajar lr_D, label smoothing, menos pasos de D. |
| Salidas borrosas | GAN no aporta | Revisar λ, D, normalización; comparar con L1-only. |
| Ruido/artefactos de cuadrícula | ConvTranspose | Usar `UpSampling2D + Conv2D` o ajustar kernel/stride. |
| Modo colapso | G produce lo mismo | Aumentar diversidad, revisar D, reducir lr. |
| Color apagado | Normalización inconsistente | Verificar [-1,1] en train y en inferencia. |
| NaN | Pérdida en fp16 o logits extremos | Pérdida en fp32, clip de gradientes. |

---

## 6. Hiperparámetros

| Parámetro | Base (256) | Smoke (64) |
|---|---|---|
| `image_size` | 256 | 64 |
| `batch_size` | 1–4 | 2 |
| `ngf` / `ndf` | 64 | 8 |
| `n_downsample` | 8 | 4 |
| `epochs` | 100 | 1 |
| `lr_g` / `lr_d` | 2e-4 / 2e-4 | 2e-4 / 2e-4 |
| `beta_1` | 0.5 | 0.5 |
| `lambda_l1` | 100 | 100 |
| `dropout` | 0.5 | 0.5 |
| `sample_interval` | 5 | 1 |
| `seed` | 42 | 42 |

---

## 7. Líneas base y ablación (para el informe)

| Experimento | Qué demuestra |
|---|---|
| Imagen de entrada | Piso mínimo; no hay restauración. |
| Filtro clásico (mediana/Gaussiano) | Lo que se logra sin deep learning. |
| U-Net solo L1 (λ→∞, sin D) | La GAN aporta textura/nitidez o no. |
| Pix2Pix completo (λ=100) | Resultado principal. |
| Pix2Pix sin skip connections (opcional) | Importancia de U-Net. |
| λ=10 vs λ=100 (opcional) | Sensibilidad al peso del L1. |

Reportar **PSNR, SSIM y L1** en test, media ± std, y una figura cualitativa por experimento.
Si el tiempo alcanza, añadir FID y una encuesta humana de realismo.

---

## 8. Métricas

### PSNR

$$
\mathrm{PSNR} = 10 \log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

- `MAX = 1` si las imágenes están en [0,1] (o `2` si se usa [-1,1] y se ajusta).
- Más alto es mejor. Rango típico en restauración: 20–35 dB.
- Es sensible a errores globales de color; no siempre coincide con la percepción.

### SSIM

- Compara luminancia, contraste y estructura en ventanas locales.
- Rango [-1,1]; más cercano a 1 es mejor.
- `tf.image.ssim` o `skimage.metrics.structural_similarity`.

### L1

$$
\mathrm{L1} = \frac{1}{N}\sum |y - \hat{y}|
$$

- Más bajo es mejor. Es la métrica que optimiza el generador.

### FID (opcional)

- Distancia entre distribuciones de features de Inception.
- Más bajo es mejor.
- Requiere instalar `clean-fid` o `pytorch-fid`; añade complejidad. Solo si sobra tiempo.

---

## 9. Presupuesto de cómputo

| Configuración | Hardware | Épocas | Tiempo estimado |
|---|---|---|---|
| Smoke 64×64, 8 pares | CPU | 1 | < 1 min |
| 128×128, 400 pares | Colab T4 | 50 | 2–4 h |
| 256×256, 900 pares | Colab T4 | 100 | 8–12 h |
| 256×256, 900 pares | Kaggle P100 | 100 | 6–10 h |

Recomendaciones:

- Empezar con 128×128 para validar el pipeline; luego subir a 256.
- Guardar checkpoint **cada época** en Drive (`/content/drive/MyDrive/...`).
- Usar `tf.keras.callbacks.ModelCheckpoint` o el `CheckpointManager` de TF.
- No entrenar en GitHub Actions.

---

## 10. Referencias

- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*.
- Goodfellow et al. (2014). *Generative Adversarial Nets*.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- TensorFlow. *pix2pix: Image-to-image translation with a conditional GAN*.
