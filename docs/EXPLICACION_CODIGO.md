# Explicación del código — recorrido guiado

> Objetivo: que cualquier integrante pueda abrir un archivo y explicar qué hace, qué recibe, qué
> devuelve y por qué está así. Se completa a medida que avanza el proyecto.

## 1. Mapa de archivos

| Archivo | Qué hace | Dueño | Pregunta típica de defensa |
|---|---|---|---|
| `src/gan/config.py` | Carga y valida el YAML; fija semillas. | D | ¿Qué pasa si falta una clave? |
| `src/gan/data/dataset.py` | Lista pares, carga imágenes, normaliza a [-1,1]. | A | ¿Cómo se emparejan input y target? |
| `src/gan/data/degradations.py` | Aplica blur, ruido, JPEG, rayones, etc. | A | ¿Cómo garantizan reproducibilidad? |
| `src/gan/models/blocks.py` | Bloques encoder/decoder reutilizables. | B | ¿Dónde va la normalización? |
| `src/gan/models/generator.py` | U-Net con skip connections. | B | ¿Por qué `tanh`? |
| `src/gan/models/discriminator.py` | PatchGAN condicional. | C | ¿Por qué 30×30? |
| `src/gan/losses.py` | BCE + L1, label smoothing. | C | ¿Por qué λ = 100? |
| `src/gan/train.py` | Bucle, `train_step`, checkpoints. | D | ¿En qué orden se actualizan G y D? |
| `src/gan/evaluate.py` | PSNR, SSIM, L1, CSV, figuras. | E | ¿Qué mide SSIM que no mide PSNR? |
| `src/gan/infer.py` | Carga un checkpoint y restaura una imagen. | E | ¿Cómo se maneja una imagen de otro tamaño? |
| `app/app.py` | Interfaz Gradio. | E | ¿Dónde se normaliza la imagen? |
| `scripts/build_dataset.py` | Construye pares y splits. | A | ¿Cómo evitan la fuga de datos? |
| `scripts/check_dataset.py` | Valida estructura y manifiesto. | A | ¿Qué falla si una fuente está en dos splits? |
| `scripts/make_synthetic_sample.py` | Datos falsos para tests/CI. | F | ¿Por qué el CI no usa el dataset real? |
| `tests/*` | Verifican contratos y smoke. | Todos | ¿Qué cubre y qué no cubre? |

---

## 2. Recorrido de una imagen (de disco a predicción)

```text
1. data/processed/test/input/foto.png
2. dataset.py: load_image → decode → resize/crop → float32 → [-1,1]
3. batch: (B,256,256,3)
4. generator.py: U-Net → (B,256,256,3) en [-1,1]
5. infer.py: desnormaliza a [0,1] → guarda PNG
6. evaluate.py: compara con target → PSNR, SSIM, L1
```

Pregunta: **¿en qué paso se puede introducir un error de rango?** Respuesta: si se normaliza a
[0,1] en train y a [-1,1] en inferencia, el modelo recibe una distribución distinta y la salida
sale mal. Por eso el contrato es explícito y hay tests.

---

## 3. Explicación de las funciones clave

### `build_generator`

- **Entrada:** `image_size`, `ngf`, `n_downsample`.
- **Salida:** modelo Keras.
- **Qué hace:** apila bloques encoder, bottleneck, decoder con concatenación de skip.
- **Por qué:** U-Net conserva detalle espacial.
- **Qué mirar:** formas de los skip; `tanh` al final.

### `build_discriminator`

- **Entrada:** `image_size`, `ndf`, `n_downsample`.
- **Salida:** modelo Keras que recibe `[input, target]` y devuelve logits por parche.
- **Qué hace:** concatena canales (6) y aplica convoluciones con stride 2.
- **Por qué:** PatchGAN evalúa textura local.
- **Qué mirar:** padding válido y forma final 30×30.

### `generator_loss` / `discriminator_loss`

- **Entrada:** logits e imágenes.
- **Salida:** escalar.
- **Qué hace:** BCE con logits + L1 ponderado.
- **Por qué:** fidelidad + realismo.
- **Qué mirar:** `from_logits=True`; no aplicar sigmoid dos veces.

### `train_step`

- **Entrada:** batch `(input, target)`.
- **Salida:** losses.
- **Qué hace:** paso D, luego paso G, con `GradientTape` separados.
- **Por qué:** objetivos opuestos; no compartir gradientes.
- **Qué mirar:** `training=True` en G y D; no actualizar D en el paso G.

### `evaluate`

- **Entrada:** checkpoint, split.
- **Salida:** `metrics.csv`, figuras.
- **Qué hace:** recorre el test set, calcula métricas y guarda grids.
- **Por qué:** evidencia objetiva para el informe.
- **Qué mirar:** que el test no se use para elegir hiperparámetros.

---

## 4. Preguntas de "explícame esta línea"

Preparar respuestas para:

- `x = tf.concat([skip, x], axis=-1)` → concatenación de skip en canales.
- `layers.Conv2DTranspose(...)` → upsampling aprendido.
- `tf.keras.layers.LeakyReLU(0.2)` → evita gradientes muertos.
- `tf.image.random_flip_left_right` → augmentación consistente en input y target.
- `tf.GradientTape()` → registro de operaciones para calcular gradientes.
- `optimizer.apply_gradients(zip(grads, vars))` → actualización de pesos.
- `checkpoint_manager.save()` → guarda G, D y optimizadores.
- `tf.float32` en la pérdida con mixed precision → estabilidad numérica.

---

## 5. Cómo mantener este documento

- Cada PR que cambie una interfaz actualiza la tabla.
- Cada dueño agrega 3–5 preguntas de su módulo.
- En los ensayos de defensa se usa como guía.
- Al final, se convierte en el anexo "Explicación del código" del informe.
