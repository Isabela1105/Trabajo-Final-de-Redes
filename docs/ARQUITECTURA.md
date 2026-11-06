# Arquitectura y contratos

> Documento vivo. Cada dueño completa su sección en su PR. Al final se usa para el informe y la
> defensa.

## 1. Contratos de tensores

| Componente | Entrada | Salida | Rango |
|---|---|---|---|
| Dataset | rutas | `(input, target)` float32 `(256,256,3)` | [-1, 1] |
| Generador | `(B,256,256,3)` | `(B,256,256,3)` | [-1, 1] |
| Discriminador | `[(B,256,256,3), (B,256,256,3)]` | `(B,30,30,1)` logits | ℝ |
| Pérdida G | logits fake + fake + target | escalar | ℝ |
| Pérdida D | logits real + logits fake | escalar | ℝ |

Los tests verifican estas formas. Si se cambia una, se actualiza este documento y los tests.

---

## 2. Generador U-Net

### 2.1 Diagrama

<!-- Insertar diagrama en docs/img/generator.png y referenciarlo aquí. -->

```text
Entrada → encoder (8 bloques) → bottleneck → decoder (8 bloques + skip) → tanh
```

### 2.2 Tabla de capas

| Bloque | Tipo | Filtros | Kernel | Stride | Norm | Activación | Salida |
|---|---|---|---|---|---|---|---|
| E1 | Conv2D | 64 | 4 | 2 | — | LeakyReLU | 128×128×64 |
| E2 | Conv2D | 128 | 4 | 2 | Instance | LeakyReLU | 64×64×128 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| D1 | ConvTranspose | 512 | 4 | 2 | Instance | ReLU + Dropout | 2×2×512 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Salida | Conv2D | 3 | 4 | 1 | — | tanh | 256×256×3 |

### 2.3 Parámetros

- `ngf=64`, 8 niveles: ~54 M.
- `ngf=8`, 4 niveles (smoke): ~0.7 M.

### 2.4 Decisiones

- InstanceNorm en vez de BatchNorm (batch pequeño).
- Dropout 0.5 en los primeros 3 bloques del decoder.
- Skip connections por concatenación.
- `tanh` para salida en [-1,1].

---

## 3. Discriminador PatchGAN

### 3.1 Diagrama

<!-- Insertar diagrama en docs/img/discriminator.png. -->

```text
Concat(input, target) → Conv 64 → Conv 128 → Conv 256 → Conv 512 → Conv 1 → 30×30 logits
```

### 3.2 Tabla de capas

| Bloque | Tipo | Filtros | Kernel | Stride | Norm | Activación | Salida |
|---|---|---|---|---|---|---|---|
| C1 | Conv2D | 64 | 4 | 2 | — | LeakyReLU | 128×128×64 |
| C2 | Conv2D | 128 | 4 | 2 | Instance | LeakyReLU | 64×64×128 |
| C3 | Conv2D | 256 | 4 | 2 | Instance | LeakyReLU | 32×32×256 |
| C4 | Conv2D (con ZeroPadding) | 512 | 4 | 1 | Instance | LeakyReLU | 31×31×512 |
| Salida | Conv2D (con ZeroPadding) | 1 | 4 | 1 | — | — | 30×30×1 |

Cada celda evalúa un parche de 70×70.

---

## 4. Pérdidas

$$
\mathcal{L}_D = \frac{1}{2}\left[\mathrm{BCE}(D(x,y),1) + \mathrm{BCE}(D(x,G(x)),0)\right]
$$

$$
\mathcal{L}_G = \mathrm{BCE}(D(x,G(x)),1) + \lambda \cdot \mathrm{L1}(G(x), y)
$$

- `BCEWithLogits`, λ = 100.
- Label smoothing opcional.
- Gradient penalty experimental (no usar en el resultado principal si no está validado).

---

## 5. Entrenamiento

- Dos optimizadores Adam (G y D), lr 2e-4, β1 0.5.
- `train_step` con `tf.function`.
- Checkpoints cada época; reanudación.
- TensorBoard + CSV.
- Mixed precision en GPU.
- Early stopping por L1 de validación.

---

## 6. Evaluación

- PSNR, SSIM, L1 en test.
- Baselines: entrada, filtro clásico, U-Net L1, Pix2Pix.
- Figuras: `input | target | predicción`.
- Galería de fallos.
- FID opcional.

---

## 7. Preguntas que este documento debe responder

- ¿Qué forma tiene cada tensor y por qué?
- ¿Por qué U-Net y no una CNN simple?
- ¿Por qué PatchGAN y no un discriminador de imagen completa?
- ¿Por qué λ = 100?
- ¿Cómo se evita la fuga de datos?
- ¿Qué se guarda en un checkpoint y cómo se reanuda?
