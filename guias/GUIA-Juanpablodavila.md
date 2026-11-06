# Guía individual — Juan Pablo Dávila Moncayo

**GitHub:** `@Juanpablodavila`
**Código:** 2216168
**Rol:** Generador U-Net
**Respaldo:** Jeremy Caicedo (`@JeremyCaicedo`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `CONTRIBUTING.md`, `docs/MODELOS.md` y `docs/REFERENCIA_CODIGO.md`.

---

## 1. Tu misión

Implementar el **generador U-Net** de Pix2Pix:

- Bloques encoder/decoder reutilizables.
- Skip connections por concatenación.
- Parametrización de `ngf`, `n_downsample`, normalización y dropout.
- Verificación de formas 64/128/256 y rango `tanh`.
- Documentar la arquitectura capa por capa para el informe y la defensa.

Tu generador es el corazón del modelo: recibe la foto degradada y produce la restaurada. Cristian
lo usará en el train loop y Juan Sebastian en la app.

---

## 2. Archivos que posees

```text
src/gan/models/__init__.py
src/gan/models/blocks.py
src/gan/models/generator.py
tests/test_models.py          # parte del generador
docs/ARQUITECTURA.md          # sección U-Net
docs/DEFENSA.md               # guion del generador
```

No edites el discriminador, las pérdidas ni `train.py` sin avisar.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `docs/04-contrato-modelos` | PR-B1 | `docs(model): definir contrato de tensores y diagrama U-Net` | Jeremy |
| 1 | 2–4 | `feat/10-generator-unet` | PR-B2 | `feat(model): implementar bloques encoder y decoder`<br>`feat(model): implementar generador U-Net configurable`<br>`test(model): verificar formas 64/128/256 y rango tanh` | Jeremy, Cristian |
| 1 | 5 | `docs/11-generator-layers` | PR-B3 | `docs(model): agregar tabla de capas y conteo de parámetros` | Isabela |
| 2 | 8–11 | `refactor/17-generator-norm` | PR-B4 | `refactor(model): parametrizar normalización y dropout`<br>`fix(model): corregir concatenación de skip connections`<br>`test(model): agregar prueba de gradientes del generador`<br>`docs(model): explicar decisiones de diseño y alternativas` | Jeremy, Cristian |
| 3 | 15–19 | `feat/24-generator-tuning` | PR-B5 | `feat(model): agregar inicialización y variantes de ngf`<br>`perf(model): reducir memoria en 256x256`<br>`docs(model): preparar guion de explicación del generador` | Cristian, Isabela |
| 4 | 22–24 | `fix/30-generator-borders` | PR-B6 | `fix(model): manejar imágenes no cuadradas sin deformar`<br>`test(model): congelar pruebas de arquitectura` | Isabella |

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
git switch -c docs/04-contrato-modelos
```

Escribe en `docs/ARQUITECTURA.md` el contrato:

```text
Generador: (B,256,256,3) -> (B,256,256,3) en [-1,1]
```

Commit:

```bash
git add docs/ARQUITECTURA.md
git commit -m "docs(model): definir contrato de tensores y diagrama U-Net (#4)"
git push -u origin HEAD
```

---

## 5. Semana por semana

### Semana 1 — Generador v1

Implementa en `blocks.py`:

- `downsample(filters, size=4, norm="instance", apply_norm=True)`.
- `upsample(filters, size=4, norm="instance", dropout=0.0)`.

Implementa en `generator.py`:

- `build_generator(image_size=256, ngf=64, n_downsample=8, norm="instance", dropout=0.5)`.
- Filtros: `[min(ngf*2**i, ngf*8) for i in range(n_downsample)]`.
- Primer bloque sin normalización.
- Skip connections con `Concatenate`.
- Salida `Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")`.

Tests:

```python
@pytest.mark.parametrize("size,ngf,n_down", [(64,8,4), (128,16,6), (256,32,8)])
def test_generator_shape(size, ngf, n_down):
    model = build_generator(image_size=size, ngf=ngf, n_downsample=n_down)
    y = model(tf.zeros((1, size, size, 3)), training=False)
    assert y.shape == (1, size, size, 3)
    assert tf.reduce_min(y) >= -1.0 and tf.reduce_max(y) <= 1.0
```

**Criterio de aceptación:**

```bash
pytest tests/test_models.py -k generator -q
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
```

### Semana 2 — Refactor y estabilidad

- Parametriza `norm` (`instance`/`batch`) y `dropout`.
- Corrige cualquier desalineación de skip connections.
- Test de gradientes: `tape.gradient(loss, generator.trainable_variables)` no debe ser `None`.
- Documenta por qué InstanceNorm y no BatchNorm.

### Semana 3 — Tuning

- Inicialización `tf.random_normal_initializer(0., 0.02)`.
- Variantes de `ngf` para comparar parámetros/calidad.
- Mide memoria en 256×256; documenta si hay que reducir batch.
- Escribe el guion de defensa del generador.

### Semana 4 — Cierre

- Maneja imágenes no cuadradas en inferencia (resize + padding, sin deformar).
- Congela tests de arquitectura (formas, número de niveles, rango).
- Revisa que el checkpoint del generador se pueda guardar/cargar.

---

## 6. Tabla de capas que debes completar

| Bloque | Tipo | Filtros | Kernel | Stride | Norm | Activación | Salida |
|---|---|---|---|---|---|---|---|
| E1 | Conv2D | 64 | 4 | 2 | — | LeakyReLU | 128×128×64 |
| E2 | Conv2D | 128 | 4 | 2 | Instance | LeakyReLU | 64×64×128 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| D1 | ConvTranspose | 512 | 4 | 2 | Instance | ReLU + Dropout | 2×2×512 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Salida | ConvTranspose | 3 | 4 | 2 | — | tanh | 256×256×3 |

Número de parámetros: ~54 M con `ngf=64`, 8 niveles; ~0.7 M con `ngf=8`, 4 niveles.

---

## 7. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-A2 | Isabella | Que el dataset devuelva `(input, target)` alineados y en `[-1,1]`. |
| PR-C2 | Jeremy | Que el discriminador reciba `[input, target]` concatenados y devuelva logits. |
| PR-E4 | Juan Sebastian | Que la app cargue el generador con la misma normalización del entrenamiento. |
| PR-F5 | Isabela | Que el README y el release describan bien el modelo. |

Si no entiendes una línea, pide explicación antes de aprobar.

---

## 8. Qué debes poder explicar en la defensa

1. ¿Por qué U-Net y no una CNN simple?
2. ¿Qué hacen las skip connections y por qué concatenan?
3. ¿Por qué `tanh` al final?
4. ¿Por qué InstanceNorm y no BatchNorm?
5. ¿Cuántos parámetros tiene y cómo cambia con `ngf`?
6. ¿Qué pasa si `n_downsample` no coincide con el tamaño de imagen?
7. ¿Cómo se guarda y se carga el generador?
8. ¿Por qué el primer bloque no lleva normalización?

---

## 9. Entregables

- [ ] `blocks.py` y `generator.py` comentados.
- [ ] Tests de forma 64/128/256 y rango.
- [ ] Test de gradientes.
- [ ] Tabla de capas y conteo de parámetros.
- [ ] Diagrama U-Net en `docs/img/generator.png`.
- [ ] Guion de defensa del generador.
- [ ] 14 commits y 6 PRs.
- [ ] 4 revisiones a otros.
- [ ] Sección **Generador** del informe.

---

## 10. Checklist final

- [ ] `pytest tests/test_models.py -q` pasa.
- [ ] El generador funciona en 64, 128 y 256.
- [ ] El smoke training usa tu generador sin errores.
- [ ] El checkpoint se guarda y se reanuda.
- [ ] Puedes dibujar la U-Net de memoria.
- [ ] Tu nombre aparece en `docs/PARTICIPACION.md`.
