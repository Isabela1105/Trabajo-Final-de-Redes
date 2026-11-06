# Reparto de roles y responsabilidades

> Este documento explica el **porqué** de cada rol. La tabla operativa con nombres, usuarios,
> códigos, archivos y revisores está en [`EQUIPO.md`](EQUIPO.md).
> Todos programan. No hay rol de "solo documentación" ni de "solo presentación".

**Equivalencias:** Persona A = Isabella Ortiz · Persona B = Juan Pablo Dávila · Persona C = Jeremy
Caicedo · Persona D = Cristian Muñoz · Persona E = Juan Sebastian Castillo · Persona F = Isabela.

## 1. Principio de diseño

El trabajo se divide en **módulos con interfaces claras**, no en tareas sueltas. Cada persona es
dueña de un módulo, pero **todos revisan código de otros** y **todos escriben una sección del
informe**. Así se cumple lo que pide el profesor: participación visible de los 6 en la generación
de código y capacidad de explicar cualquier parte.

### Contratos que permiten trabajar en paralelo

| Componente | Entrada | Salida | Dueño |
|---|---|---|---|
| Dataset | rutas de imágenes | `(input, target)` float32 [-1,1], 256×256 | A |
| Generador | `(B,256,256,3)` | `(B,256,256,3)` en [-1,1] | B |
| Discriminador | `(B,256,256,3)` × 2 | `(B,30,30,1)` logits | C |
| Pérdidas | logits + imágenes | escalares | C |
| Entrenamiento | config + dataset + modelos | checkpoints, logs | D |
| Evaluación | checkpoint + test set | PSNR, SSIM, L1, figuras | E |
| CI/CD | código | checks verdes + release | F |

Los tests de forma verifican estos contratos. Si alguien rompe uno, el CI falla.

---

## 2. Roles

### Persona A — Datos y dataset

**Misión:** construir el dataset personalizado, reproducible y sin fuga de datos.

- Recolectar/organizar las fotos y documentar procedencia, consentimiento y licencia.
- Implementar `src/gan/data/dataset.py` y `degradations.py`.
- Crear `scripts/build_dataset.py` y `scripts/check_dataset.py`.
- Generar splits 80/10/10 **por fuente**, manifiesto y estadísticas.
- Mantener `data/README.md` y `docs/DATASET.md`.
- Escribir la sección **Dataset** del informe y las diapositivas de datos.

**Entregable verificable:** `python scripts/build_dataset.py --config configs/base.yaml`
reconstruye el dataset y `pytest tests/test_dataset.py` pasa.

**Respaldo:** B.

---

### Persona B — Generador U-Net

**Misión:** implementar y explicar el generador.

- Implementar bloques encoder/decoder y skip connections.
- Parametrizar `ngf`, `n_downsample`, normalización, dropout.
- Verificar formas 64/128/256 y rango `tanh`.
- Medir parámetros y memoria; documentar tabla de capas.
- Escribir la sección **Generador** del informe y las diapositivas de U-Net.

**Entregable verificable:** `pytest tests/test_models.py -k generator` pasa y el generador
produce una imagen de la forma correcta con pesos aleatorios.

**Respaldo:** C.

---

### Persona C — Discriminador y pérdidas

**Misión:** implementar el discriminador PatchGAN y las funciones de pérdida.

- Implementar `discriminator.py` con salida de parches.
- Implementar `losses.py`: BCE con logits, L1 ponderado, label smoothing opcional.
- Verificar formas, valores conocidos y estabilidad numérica.
- Documentar ecuaciones, λ = 100 y alternativas (BCE vs MSE, LSGAN).
- Escribir la sección **Pérdidas y discriminador** del informe.

**Entregable verificable:** `pytest tests/test_losses.py tests/test_models.py -k discriminator`
pasa; las pérdidas son finitas y diferenciables.

**Respaldo:** B.

---

### Persona D — Entrenamiento

**Misión:** que el pipeline de entrenamiento corra, se reanude y sea reproducible.

- Implementar `train.py` con `tf.function`, dos optimizadores, checkpoints y TensorBoard.
- Cargar configuración YAML, validar claves y fijar semillas.
- Integrar dataset + generador + discriminador + pérdidas.
- Añadir mixed precision, gradient clipping y early stopping.
- Ejecutar el entrenamiento real en Colab/Kaggle y documentar hiperparámetros.
- Escribir la sección **Entrenamiento** del informe.

**Entregable verificable:** `python -m src.gan.train --config configs/smoke.yaml --max-steps 2`
corre en CPU; un entrenamiento de 20 épocas genera checkpoints y muestras.

**Respaldo:** E.

---

### Persona E — Evaluación, inferencia y demo

**Misión:** medir, comparar y mostrar el modelo.

- Implementar PSNR, SSIM, L1 y (opcional) FID.
- Evaluar el test set y guardar CSV + figuras.
- Implementar `infer.py` para una imagen.
- Crear la app Gradio y desplegarla (local y, si se puede, HF Spaces).
- Escribir la sección **Resultados y demo** del informe.

**Entregable verificable:** `python -m src.gan.evaluate --config configs/base.yaml --checkpoint
<ruta>` produce `metrics.csv` y `docs/img/grid.png`; `python app/app.py` abre la demo.

**Respaldo:** D.

---

### Persona F — CI/CD, QA e informe

**Misión:** que el repositorio no se rompa y que la entrega sea reproducible.

- Configurar GitHub Actions, pre-commit, plantillas y protección de `main`.
- Mantener `README.md`, `CONTRIBUTING.md`, `docs/CI_CD.md` y `docs/PARTICIPACION.md`.
- Construir el PDF del informe y el release en tags.
- Coordinar revisiones, merges y checklist de entrega.
- Escribir **Introducción, conclusiones y anexos** del informe.

**Entregable verificable:** un PR con un error de lint es bloqueado por el CI; un tag `v1.0.0`
genera un release con el PDF.

**Respaldo:** A.

---

## 3. Matriz de respaldos

| Tema | Dueño | Respaldo | Qué debe saber el respaldo |
|---|---|---|---|
| Dataset | A | B | Ejecutar `build_dataset.py`, explicar splits y degradaciones. |
| Generador | B | C | Formas, skip connections, `ngf`, `tanh`. |
| Discriminador/pérdidas | C | B | PatchGAN, BCE, L1, λ. |
| Entrenamiento | D | E | `train_step`, checkpoints, reanudación. |
| Evaluación/app | E | D | PSNR/SSIM, inferencia, Gradio. |
| CI/informe | F | A | Workflow, branch protection, build del PDF. |

**Regla:** el respaldo debe poder ejecutar y explicar el módulo del dueño. Se prueba en los
ensayos de defensa con selección aleatoria.

---

## 4. Dedicación semanal estimada

| Actividad | Horas/semana |
|---|---|
| Reunión de planeación (lunes) | 0.5 |
| Trabajo de módulo (2–3 sesiones) | 4–6 |
| Revisiones de PR | 1 |
| Informe / figuras | 1–2 |
| Demo interna (viernes) | 0.5 |
| **Total** | **7–10** |

Si alguien dispone de menos tiempo, debe avisarlo en la semana 1 para redistribuir, no en la
semana 4.

---

## 5. Reuniones

| Reunión | Cuándo | Duración | Salida |
|---|---|---|---|
| Planeación | Lunes | 30 min | Issues de la semana asignados, dependencias claras. |
| Bloqueos | Miércoles | 15 min | Cada uno dice qué está bloqueado y quién ayuda. |
| Demo + merge | Viernes | 45 min | PRs revisados/mergeados, demo de lo avanzado. |
| Ensayo defensa | Semana 4, 2 veces | 30 min | Presentador aleatorio, preguntas cruzadas. |

**Regla:** las decisiones técnicas se registran en `docs/DECISIONES.md` o en el issue, no solo en
WhatsApp. Eso alimenta el informe y evita malentendidos.

---

## 6. Reparto del informe

| Sección | Redactor principal | Revisor |
|---|---|---|
| Resumen | F | D |
| 1. Introducción | F | A |
| 2. Marco teórico (GAN, Pix2Pix) | C | B |
| 3. Dataset | A | E |
| 4. Arquitectura (U-Net, PatchGAN) | B + C | D |
| 5. Entrenamiento | D | B |
| 6. Experimentos y resultados | D + E | C |
| 7. Aplicación/demo | E | F |
| 8. Discusión y limitaciones | Todos | F |
| 9. Conclusiones | F | A |
| 10. Referencias | F | C |
| Anexos (participación, código) | F + todos | A |

Detalle en `docs/INFORME.md`.

---

## 7. Reglas de convivencia técnica

1. **No aprobar un PR que no se entiende.** Pedir explicación es parte del trabajo.
2. **No mergear código de otro sin revisarlo.** El historial debe reflejar autoría real.
3. **No borrar trabajo ajeno** sin discutirlo en el issue.
4. **No subir datos personales, pesos o archivos grandes.**
5. **No usar IA para generar todo sin entenderlo.** El profesor puede preguntar cualquier línea.
   Si se usa una herramienta de IA, se documenta en el PR y se explica.
6. **Los desacuerdos técnicos se resuelven con evidencia:** un experimento, una métrica o una
   referencia, no por quién habla más.
7. **Si alguien no puede cumplir, avisa temprano.** El respaldo toma la tarea y se reasigna el
   issue; no se deja morir el módulo.

---

## 8. Tabla de participación (plantilla)

```markdown
| Integrante | GitHub | Módulos | PRs autorados | PRs revisados | Commits | Informe | Defensa |
|---|---|---|---|---|---|---|---|
| Isabella Ortiz (@Isabellaortizher-collab) | Datos | #3, #7, #9... | #10, #12... | 14 | Dataset | Datos |
| Juan Pablo Dávila (@Juanpablodavila) | Generador | ... | ... | 14 | Arquitectura | U-Net |
| Jeremy Caicedo (@JeremyCaicedo) | D/pérdidas | ... | ... | 14 | Marco teórico | Pérdidas |
| Cristian Muñoz (@CristianMP355) | Entrenamiento | ... | ... | 15 | Entrenamiento | Train loop |
| Juan Sebastian Castillo (@Juansec01) | Evaluación/app | ... | ... | 15 | Resultados | Demo |
| Isabela (@Isabela1105) | CI/informe | ... | ... | 15 | Intro/conclusiones | Cierre |
```

Se completa al final de cada semana en `docs/PARTICIPACION.md`.
