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

Responsable: Cristian Muñoz (`@CristianMP355`). Implementación en `src/gan/train.py`,
`src/gan/config.py` y `src/gan/utils/viz.py`.

### 5.1 Adaptación del dataset

El módulo de Isabella ya entrega el cargador real
`make_dataset(split_dir, batch_size, shuffle, seed, size, num_parallel_calls) -> tf.data.Dataset`,
que lee `split/input/*.png` y `split/target/*.png` y devuelve pares `(input, target)` float32
`(B,256,256,3)` en `[-1,1]` (normalización `x / 127.5 - 1`, con `tf.ensure_shape` del tamaño).
`_make_split_dataset` solo traduce las claves de `configs/base.yaml` (`data.image_size`,
`data.batch_size`, `data.num_parallel_calls`, `seed`) a esa firma; no se reimplementa ni se
modifica el módulo ajeno. La forma estática del batch evita retracing dentro de `tf.function`
y `_validate_pair` sigue verificando en grafo los 3 canales y el rango `[-1,1]` de cada par.
`make_tf_dataset` (envolvente sobre el generador numpy `build_dataset`) se conserva solo como
fallback legado para scripts antiguos; el bucle ya no lo usa.

### 5.2 Orden estricto D → G

`make_train_step` devuelve un `tf.function` con `input_signature` fija y captura los modelos y
optimizadores como objetos Python (no como tensores), para evitar retracing:

```text
1) D primero, G congelado:
   fake = G(x, training=False)                 # fuera del tape de G
   d_real = D([x, y], training=True)
   d_fake = D([x, stop_gradient(fake)], training=True)
   L_D = 0.5 * (BCE(d_real, 1-s) + BCE(d_fake, 0))
   grads_D -> clip -> opt_D.apply_gradients

2) G después, D congelado:
   fake = G(x, training=True)
   d_fake = D([x, fake], training=False)       # D no actualiza ni sus stats
   L_G = BCE(d_fake, 1) + λ · L1(fake, y)
   grads_G -> clip -> opt_G.apply_gradients
```

Dos `GradientTape` independientes garantizan que los gradientes de cada red no se mezclen.
A diferencia del esquema del tutorial, aquí D se actualiza **antes** y G ve al D ya actualizado.

### 5.3 Optimizadores y estabilidad

- Dos `Adam` separados: `lr_g = lr_d = 2e-4`, `beta_1 = 0.5`, `beta_2 = 0.999`.
- `label_smoothing`: `_call_discriminator_loss` lo pasa solo si la pérdida lo acepta. La pérdida
  real de Jeremy tiene firma `discriminator_loss(real_logits, fake_logits)` (sin smoothing), así
  que con los módulos reales el valor efectivo es 0.0; se avisa por pantalla si se pide otro
  valor. La adaptación vive en `train.py`, sin tocar `src/gan/losses.py`.
- Clip global de gradientes (`clip_norm = 1.0`) antes de `apply_gradients`.
- Mixed precision `mixed_float16` **solo si hay GPU**; las pérdidas y el L1 se calculan
  explícitamente en `float32` para evitar NaN.
- Early stopping por L1 de validación (`early_stopping_patience = 15`).

### 5.4 Checkpoints y reanudación

`tf.train.CheckpointManager` guarda cada época (máx. 3):

| Objeto | Propósito |
|---|---|
| `generator`, `discriminator` | arquitectura + pesos |
| `generator_optimizer`, `discriminator_optimizer` | momentos e iteraciones de Adam |
| `epoch`, `seed` | punto exacto de reanudación |

Junto al checkpoint se escriben de forma atómica (temporal + `os.replace`) un `config.yaml`
con el YAML usado y un `metadata.json` con época, semilla y `sha256` de la configuración.
`--resume` restaura el último estado y continúa en `epoch + 1`; `--resume-from <ruta>`
permite elegir un checkpoint explícito. Si el YAML cambió, se avisa pero se respeta la config
actual. Las variables de modelos y optimizadores se crean con un paso ficticio antes de
restaurar, para que no falten slots en el checkpoint.

### 5.5 Evidencia

- `logs/`: escalares `loss_g`, `loss_d`, `loss_adv`, `loss_l1`, `val_l1` + imágenes
  `input | target | prediction` por época (TensorBoard).
- `samples/epoch_XXXX.png`: grid matplotlib guardado atómicamente.
- `metrics.csv`: columnas `epoch, split, loss_g, loss_d, l1, psnr, ssim`. Hay una fila por época y
  split: `train` lleva las pérdidas; `val` lleva L1, PSNR y SSIM (`max_val=2.0` porque las
  imágenes están en `[-1,1]`).
- El generador se reconstruye desde los parámetros del YAML + `build_generator`, sin
  hardcodear arquitectura en el checkpoint.

### 5.6 Hiperparámetros (configs/base.yaml)

| Parámetro | Valor | Parámetro | Valor |
|---|---|---|---|
| `image_size` | 256 | `batch_size` | 2 (rango 1–4) |
| `ngf` / `ndf` | 64 / 64 | `n_downsample` | 8 |
| `epochs` | 100 | `seed` | 42 |
| `lr_g` / `lr_d` | 2e-4 / 2e-4 | `beta_1` / `beta_2` | 0.5 / 0.999 |
| `lambda_l1` | 100 | `label_smoothing` | 0.0 |
| `clip_norm` | 1.0 | `mixed_precision` | true (solo GPU) |
| `sample_interval` | 5 | `checkpoint_interval` | 1 |

### 5.7 Ablación

`train.use_gan: false` desactiva la actualización de D y el término adversarial: el
generador queda como U-Net solo L1 (línea base). `configs/ablation_l1.yaml` es la receta
lista para correrla con `python -m src.gan.train --config configs/ablation_l1.yaml`.
El resto del pipeline (métricas, muestras, checkpoints) es idéntico, de modo que la
comparación con Pix2Pix es justa. `tests/test_smoke_train.py` congela el orden D→G y verifica
que, en ablación, D no se actualiza.

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
