# Guía individual — Jeremy Santiago Caicedo Cabezas

**GitHub:** `@JeremyCaicedo`
**Código:** 2210721
**Rol:** Discriminador y pérdidas
**Respaldo:** Juan Pablo Dávila (`@Juanpablodavila`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `CONTRIBUTING.md`, `docs/MODELOS.md` y `docs/REFERENCIA_CODIGO.md`.

---

## 1. Tu misión

Implementar el **discriminador PatchGAN** y las **pérdidas** de Pix2Pix:

- Discriminador condicional que recibe `[input, target]`.
- Salida de parches `(B,30,30,1)` logits para 256×256.
- Pérdida adversarial BCE con logits + L1 ponderado (λ = 100).
- Label smoothing opcional y gradient penalty experimental.
- Tests de formas, valores conocidos y estabilidad numérica.
- Documentar las ecuaciones y justificar las decisiones.

Sin un discriminador estable, el generador no aprende textura. Sin una pérdida bien implementada,
el modelo se vuelve borroso o colapsa.

---

## 2. Archivos que posees

```text
src/gan/models/discriminator.py
src/gan/losses.py
tests/test_losses.py
tests/test_models.py            # parte del discriminador
docs/ARQUITECTURA.md            # secciones PatchGAN y pérdidas
docs/DEFENSA.md                 # guion de pérdidas
```

No edites el generador ni `train.py` sin avisar.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `docs/05-contrato-discriminador` | PR-C1 | `docs(model): definir contrato y diagrama del discriminador` | Juan Pablo |
| 1 | 2–4 | `feat/12-patchgan-losses` | PR-C2 | `feat(model): implementar discriminador PatchGAN`<br>`feat(loss): implementar pérdidas adversarial y L1`<br>`test(model): verificar parches 30x30 y pérdidas finitas` | Juan Pablo, Cristian |
| 1 | 5 | `docs/13-loss-theory` | PR-C3 | `docs(model): explicar PatchGAN y pérdida adversarial` | Isabela |
| 2 | 8–11 | `feat/20-loss-options` | PR-C4 | `feat(loss): agregar label smoothing configurable`<br>`test(loss): verificar valores con casos conocidos`<br>`feat(loss): permitir ponderación de L1 por configuración`<br>`docs(model): comparar BCE vs MSE y justificar la elección` | Juan Pablo, Cristian |
| 3 | 15–19 | `feat/27-gradient-penalty` | PR-C5 | `feat(loss): agregar gradient penalty experimental`<br>`test(loss): probar estabilidad numérica de las pérdidas`<br>`docs(model): preparar guion de explicación de pérdidas` | Cristian, Isabela |
| 4 | 22–24 | `fix/32-loss-nan` | PR-C6 | `fix(loss): evitar NaN con logits extremos`<br>`docs(model): congelar ecuaciones y notación` | Juan Pablo |

**Total:** 6 PRs, 14 commits.

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
git switch -c docs/05-contrato-discriminador
```

Escribe en `docs/ARQUITECTURA.md` el contrato:

```text
Discriminador: [(B,256,256,3), (B,256,256,3)] -> (B,30,30,1) logits
```

Commit:

```bash
git add docs/ARQUITECTURA.md
git commit -m "docs(model): definir contrato y diagrama del discriminador (#5)"
git push -u origin HEAD
```

---

## 5. Semana por semana

### Semana 1 — PatchGAN + pérdidas

**Discriminador:**

- Concatena `input` y `target` por canales: `(B,256,256,6)`.
- Bloques: `64 stride 2` → `128 stride 2` → `256 stride 2` → `ZeroPadding + 512 stride 1`
  → `ZeroPadding + 1 stride 1`.
- InstanceNorm + LeakyReLU(0.2) en los bloques intermedios.
- Salida `(B,30,30,1)` logits.

**Pérdidas:**

```python
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits, label_smoothing=0.0):
    real_labels = tf.ones_like(real_logits) * (1.0 - label_smoothing)
    fake_labels = tf.zeros_like(fake_logits)
    return 0.5 * (bce(real_labels, real_logits) + bce(fake_labels, fake_logits))

def generator_loss(fake_logits, fake_image, target_image, lambda_l1=100.0):
    adv = bce(tf.ones_like(fake_logits), fake_logits)
    l1 = tf.reduce_mean(tf.abs(target_image - fake_image))
    return adv + lambda_l1 * l1, adv, l1
```

**Tests:**

- Discriminador 256 → `(1,30,30,1)`.
- Discriminador 64 → `(1,6,6,1)` (para smoke).
- Pérdidas finitas y diferenciables.
- Caso conocido: si `fake == target`, L1 = 0.
- `from_logits=True` (no aplicar sigmoid dos veces).

### Semana 2 — Opciones de pérdida

- `label_smoothing` configurable (0.0 por defecto, 0.1 opcional).
- `lambda_l1` desde YAML.
- Comparar BCE vs MSE (LSGAN) en documentación; no cambiar el resultado principal sin evidencia.
- Tests con valores conocidos.

### Semana 3 — Gradient penalty experimental

- Implementar `gradient_penalty` opcional (WGAN-GP) sin romper el camino principal.
- Test de estabilidad numérica: logits extremos no producen `NaN`.
- Guion de defensa de pérdidas.

### Semana 4 — Cierre

- Evitar `NaN` con logits muy grandes (clipping o `tf.clip_by_value`).
- Congelar ecuaciones y notación en `docs/ARQUITECTURA.md`.
- Verificar que el train loop usa tus pérdidas correctamente.

---

## 6. Ecuaciones que debes dominar

$$
\mathcal{L}_D = \frac{1}{2}\left[\mathrm{BCE}(D(x,y),1) + \mathrm{BCE}(D(x,G(x)),0)\right]
$$

$$
\mathcal{L}_G = \mathrm{BCE}(D(x,G(x)),1) + \lambda \cdot \mathrm{L1}(G(x), y)
$$

- λ = 100: el L1 domina, obliga a ser fiel; el adversarial añade textura.
- Sin L1: el modelo inventa detalles.
- Sin adversarial: salida borrosa (promedio).
- `BCEWithLogits` evita el `sigmoid` explícito y es más estable.

---

## 7. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-B2 | Juan Pablo | Que el generador devuelva `[-1,1]` y que el discriminador reciba ese rango. |
| PR-A4 | Isabella | Que el dataset no tenga fuga y que los pares estén alineados. |
| PR-D4 | Cristian | Que el `train_step` use tus pérdidas y actualice G/D en el orden correcto. |
| PR-E5 | Juan Sebastian | Que las métricas no se confundan con las pérdidas; que PSNR/SSIM se calculen en `[0,1]`. |
| PR-F6 | Isabela | Que la tabla de participación y el release sean correctos. |

---

## 8. Qué debes poder explicar en la defensa

1. ¿Qué es PatchGAN y por qué 30×30?
2. ¿Por qué el discriminador recibe entrada y salida concatenadas?
3. ¿Por qué `BCEWithLogits` y no `sigmoid` + BCE?
4. ¿Por qué λ = 100?
5. ¿Qué pasa si el discriminador es demasiado fuerte?
6. ¿Qué hace label smoothing?
7. ¿Qué es gradient penalty y por qué es experimental?
8. ¿Cómo se evita `NaN` en las pérdidas?

---

## 9. Entregables

- [ ] `discriminator.py` comentado.
- [ ] `losses.py` con BCE + L1 + label smoothing.
- [ ] Tests de formas, valores conocidos y estabilidad.
- [ ] Ecuaciones y justificación en `docs/ARQUITECTURA.md`.
- [ ] Guion de defensa de pérdidas.
- [ ] 14 commits y 6 PRs.
- [ ] 5 revisiones a otros.
- [ ] Sección **Pérdidas y discriminador** del informe.

---

## 10. Checklist final

- [ ] `pytest tests/test_losses.py tests/test_models.py -q` pasa.
- [ ] Discriminador 256 → `(1,30,30,1)`.
- [ ] Discriminador 64 → `(1,6,6,1)`.
- [ ] `generator_loss` devuelve `(total, adv, l1)`.
- [ ] No hay `sigmoid` duplicado.
- [ ] No hay `NaN` con logits extremos.
- [ ] Tu nombre aparece en `docs/PARTICIPACION.md`.
