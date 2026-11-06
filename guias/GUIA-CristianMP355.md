# Guía individual — Cristian Fabian Muñoz Piedrahita

**GitHub:** `@CristianMP355`
**Código:** 2230865
**Rol:** Entrenamiento
**Respaldo:** Juan Sebastian Castillo (`@Juansec01`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `CONTRIBUTING.md`, `docs/MODELOS.md` y `docs/REFERENCIA_CODIGO.md`.

---

## 1. Tu misión

Hacer que el pipeline de entrenamiento **corra, se reanude y sea reproducible**:

- Cargar y validar la configuración YAML.
- Implementar `train_step` con dos optimizadores.
- Integrar dataset + generador + discriminador + pérdidas.
- Guardar checkpoints (G, D, optimizadores) y muestras.
- TensorBoard + CSV de métricas.
- Mixed precision, gradient clipping y early stopping.
- Ejecutar el entrenamiento real en Colab/Kaggle y documentar hiperparámetros.

Eres el integrador técnico: si tu train loop falla, nadie puede ver resultados.

---

## 2. Archivos que posees

```text
src/gan/config.py
src/gan/train.py
src/gan/utils/seed.py
src/gan/utils/viz.py
configs/base.yaml
configs/smoke.yaml
tests/test_config.py
tests/test_smoke_train.py
docs/ARQUITECTURA.md          # sección entrenamiento
docs/DEFENSA.md               # guion del train loop
```

No edites el generador, el discriminador ni las pérdidas sin avisar.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `chore/06-config` | PR-D1 | `chore(config): agregar cargador y validación de YAML` | Juan Sebastian |
| 1 | 2–4 | `feat/14-train-loop` | PR-D2 | `feat(train): implementar train_step con tf.function`<br>`feat(train): agregar checkpoints y reanudación`<br>`feat(train): agregar TensorBoard y guardado de muestras` | Juan Pablo, Jeremy |
| 1 | 5 | `test/15-smoke-train` | PR-D3 | `test(train): smoke test de dos pasos en CPU` | Isabela |
| 2 | 8–11 | `feat/23-train-integration` | PR-D4 | `feat(train): integrar dataset y modelo en el bucle`<br>`feat(train): agregar precisión mixta y clip de gradientes`<br>`feat(train): agregar early stopping por L1 de validación`<br>`docs(train): documentar hiperparámetros y reanudación` | Isabella, Juan Pablo, Juan Sebastian |
| 3 | 15–20 | `feat/29-train-experiments` | PR-D5 | `feat(train): registrar métricas en CSV y resumen`<br>`fix(train): corregir orden de pasos D/G`<br>`feat(train): agregar configuración de ablación sin GAN`<br>`docs(train): preparar guion del bucle de entrenamiento` | Jeremy, Juan Sebastian, Isabela |
| 4 | 22–24 | `fix/34-colab-checkpoint` | PR-D6 | `fix(train): estabilizar checkpoints en Colab`<br>`docs(train): congelar hiperparámetros finales` | Juan Sebastian |

**Total:** 6 PRs, 15 commits.

---

## 4. Día 1: cómo empezar

```bash
git clone https://github.com/Isabela1105/Trabajo-Final-de-Redes.git
cd Trabajo-Final-de-Redes

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install

git switch main
git pull --rebase
git switch -c chore/06-config
```

Implementa `config.py`:

- Cargar YAML con `yaml.safe_load`.
- Validar claves obligatorias (`data`, `model`, `train`, `output`).
- Resolver rutas relativas a la raíz del repo.
- Fijar semillas (`random`, `numpy`, `tensorflow`).

Commit:

```bash
git add src/gan/config.py configs/base.yaml tests/test_config.py
git commit -m "chore(config): agregar cargador y validación de YAML (#6)"
git push -u origin HEAD
```

---

## 5. Semana por semana

### Semana 1 — Train loop v1

**`train_step`:**

```python
@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real = discriminator([input_image, target_image], training=True)
        disc_fake = discriminator([input_image, gen_output], training=True)
        gen_total, gen_adv, gen_l1 = generator_loss(disc_fake, gen_output, target_image, lambda_l1)
        disc_loss = discriminator_loss(disc_real, disc_fake, label_smoothing)
    gen_grads = gen_tape.gradient(gen_total, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return gen_total, gen_adv, gen_l1, disc_loss
```

**Checkpoints:**

```python
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator,
    discriminator=discriminator,
)
manager = tf.train.CheckpointManager(checkpoint, "outputs/base/checkpoints", max_to_keep=3)
manager.save(checkpoint_number=epoch)
```

**TensorBoard + muestras:**

- `tf.summary.scalar` para pérdidas.
- `tf.summary.image` para un batch fijo de validación.
- Guardar grid cada `sample_interval` épocas.

### Semana 2 — Integración

- Crear `tf.data.Dataset` de train/val con el dataset de Isabella.
- Mixed precision: `tf.keras.mixed_precision.set_global_policy("mixed_float16")` solo en GPU.
- Gradient clipping: `tf.clip_by_norm(g, 1.0)`.
- Early stopping por L1 de validación (paciencia 15).
- Reanudar con `--resume`.
- Documentar hiperparámetros.

**Criterio de aceptación:**

```bash
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
pytest tests/test_smoke_train.py -q
```

### Semana 3 — Experimentos

- `metrics.csv` con `epoch, loss_g, loss_d, l1, val_l1, val_psnr, val_ssim`.
- Corregir orden D/G si hay inestabilidad.
- Configuración de ablación sin GAN (`lambda_l1` alto, sin D) para comparar.
- Entrenamiento real en Colab T4 / Kaggle P100.
- Guardar checkpoints en Drive cada época.

### Semana 4 — Cierre

- Estabilizar checkpoints en Colab (evitar rutas temporales, guardar en Drive).
- Congelar hiperparámetros finales en `configs/base.yaml`.
- Verificar que el checkpoint se puede cargar en `infer.py`.
- Documentar tiempo y hardware en el informe.

---

## 6. Hiperparámetros de referencia

| Parámetro | Valor |
|---|---|
| Imagen | 256×256 |
| Batch | 1–4 |
| Optimizador | Adam, lr 2e-4, β1 0.5, β2 0.999 |
| Pérdida | BCEWithLogits + 100·L1 |
| Épocas | 100 |
| Early stopping | paciencia 15 por val L1 |
| Checkpoint | cada época, max 3 |
| Mixed precision | sí en GPU |
| Semilla | 42 |

---

## 7. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-A2 | Isabella | Que el dataset devuelva batches alineados y sin fuga. |
| PR-B2 | Juan Pablo | Que el generador acepte el batch y devuelva `[-1,1]`. |
| PR-C2 | Jeremy | Que las pérdidas usen logits y que `train_step` las llame bien. |
| PR-E4 | Juan Sebastian | Que la app cargue el checkpoint y use la misma normalización. |
| PR-F4 | Isabela | Que el CI cachee pip y que el smoke no tarde más de 10 min. |

---

## 8. Qué debes poder explicar en la defensa

1. ¿Qué hace `train_step` paso a paso?
2. ¿Por qué dos optimizadores y dos `GradientTape`?
3. ¿En qué orden se actualizan G y D?
4. ¿Qué se guarda en un checkpoint y cómo se reanuda?
5. ¿Qué es mixed precision y por qué la pérdida queda en `float32`?
6. ¿Cómo detectas overfitting?
7. ¿Qué pasa si el discriminador domina?
8. ¿Por qué el CI solo hace 2 pasos y no entrena completo?

---

## 9. Entregables

- [ ] `config.py` con validación y semillas.
- [ ] `train.py` con `train_step`, checkpoints, TensorBoard y CSV.
- [ ] `configs/base.yaml` y `configs/smoke.yaml`.
- [ ] `tests/test_config.py` y `tests/test_smoke_train.py`.
- [ ] Entrenamiento real con checkpoint descargable.
- [ ] Curvas de pérdida y tabla de hiperparámetros.
- [ ] 15 commits y 6 PRs.
- [ ] 5 revisiones a otros.
- [ ] Sección **Entrenamiento** del informe.

---

## 10. Checklist final

- [ ] `pytest -q` pasa.
- [ ] Smoke training en CPU pasa en < 2 min.
- [ ] `--resume` restaura G, D y optimizadores.
- [ ] `metrics.csv` tiene todas las columnas.
- [ ] El checkpoint se carga en `infer.py`.
- [ ] Tiempo y hardware documentados.
- [ ] Tu nombre aparece en `docs/PARTICIPACION.md`.
