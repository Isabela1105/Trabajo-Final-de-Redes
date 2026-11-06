# Guía de sustentación

## 1. Lo que evalúa el profesor

- ¿Entienden la arquitectura y el código?
- ¿El dataset es propio y está bien construido?
- ¿Hay validación real (métricas + evidencia visual)?
- ¿Todos participaron y todos pueden explicar?
- ¿El repositorio está ordenado y reproducible?

**Riesgo principal:** el docente elige a un integrante al azar. Por eso **todos** deben preparar
**todas** las secciones, no solo la propia.

---

## 2. Guion de 15–20 minutos

| Min | Tema | Quién (sugerido) | Contenido |
|---|---|---|---|
| 0–2 | Problema y objetivos | A | Restauración de fotos, motivación, objetivos. |
| 2–4 | Dataset | A | Recolección, degradaciones, splits, ética. |
| 4–7 | Arquitectura | B | U-Net, skip connections, `ngf`, `tanh`. |
| 7–9 | Discriminador y pérdidas | C | PatchGAN, BCE + L1, λ = 100. |
| 9–11 | Entrenamiento | D | `train_step`, checkpoints, hiperparámetros, estabilidad. |
| 11–14 | Resultados | E | PSNR/SSIM/L1, baselines, figuras, fallos. |
| 14–16 | Demo y limitaciones | E + A | Gradio, ejemplos, alucinación, trabajo futuro. |
| 16–20 | Preguntas | Todos | Responder con precisión y honestidad. |

**Regla:** cada diapositiva debe poder ser explicada por cualquier integrante. Las diapositivas
llevan notas del orador en `docs/presentacion_notas.md`.

---

## 3. Estructura de diapositivas (12–15)

1. Portada: título, curso, integrantes, repo.
2. Problema: fotos antiguas/degradadas, por qué es difícil.
3. Objetivos.
4. Dataset: ejemplos `input/target`, cantidad, splits.
5. Pipeline de degradaciones.
6. GAN: generador vs discriminador (recordatorio visual).
7. Pix2Pix: condicional, pares.
8. Generador U-Net: diagrama y skip connections.
9. Discriminador PatchGAN: parches 70×70.
10. Pérdidas: ecuaciones y λ.
11. Entrenamiento: hiperparámetros, checkpoints, hardware.
12. Resultados: tabla PSNR/SSIM/L1 vs baselines.
13. Resultados visuales: grid y fallos.
14. Demo: captura o en vivo.
15. Limitaciones, conclusiones y trabajo futuro.
16. Gracias / preguntas / link del repo.

---

## 4. Demo: plan A, B y C

- **Plan A:** app Gradio local o en Hugging Face Spaces, con 3 imágenes de ejemplo.
- **Plan B:** video/GIF pregrabado de la demo (30–60 s).
- **Plan C:** galería de resultados precalculada en `docs/img/` y en las diapositivas.

**Nunca** depender de internet en vivo. Tener el modelo y las imágenes en local.

---

## 5. Ensayos

| Ensayo | Cuándo | Dinámica |
|---|---|---|
| 1 | Día 26 | Presentación completa; el presentador de cada bloque se elige al azar. |
| 2 | Día 27 | Igual, pero con preguntas cruzadas y cronómetro. |
| Final | Día 28 | Solo ajustes de tiempo y diapositivas. |

En cada ensayo, cada integrante debe exponer al menos un bloque que **no escribió**. Eso simula
al docente y revela huecos de comprensión.

---

## 6. Banco de preguntas (con respuestas cortas)

### Dataset

1. **¿Por qué el dataset es personalizado?**
   Porque recolectamos las fotos y generamos los pares degradado/limpio con nuestro propio
   pipeline; no usamos el dataset del tutorial.

2. **¿Cómo evitan que el modelo haga trampa?**
   Dividimos por foto fuente, no por parche. Una misma escena no aparece en train y test.

3. **¿Por qué degradaciones sintéticas y no solo fotos antiguas reales?**
   Porque las sintéticas dan *ground truth* para medir PSNR/SSIM. Las reales se usan para
   evaluación cualitativa, sin métrica objetiva.

4. **¿Cuántos datos necesitan?**
   Mínimo ~150 fuentes × 3 variantes = 450 pares; recomendado 300 × 3 = 900.

5. **¿Qué sesgos tiene el dataset?**
   Geográfico/cultural, pocas fotos nocturnas o de personas, y degradaciones que no cubren todos
   los daños reales.

### Arquitectura

6. **¿Por qué U-Net y no una CNN normal?**
   Las skip connections conservan detalle espacial de alta resolución, clave para restaurar
   texturas y bordes.

7. **¿Qué hace `tanh`?**
   Limita la salida a [-1,1], igual que la normalización de las imágenes de entrenamiento.

8. **¿Por qué InstanceNorm y no BatchNorm?**
   Con batch pequeño (1–4), BatchNorm es inestable; InstanceNorm normaliza por imagen y canal.

9. **¿Qué es PatchGAN?**
   Un discriminador que evalúa parches de 70×70 en vez de la imagen completa; la salida es
   30×30 logits para 256×256.

10. **¿Por qué el discriminador recibe la entrada y la salida concatenadas?**
    Porque es condicional: debe juzgar si la salida es coherente **con esa entrada**, no solo si
    parece una foto real.

11. **¿Cuántos parámetros tiene el generador?**
    ~54 M con `ngf=64` y 8 niveles; la versión de smoke (~0.7 M) usa `ngf=8` y 4 niveles.

### Pérdidas y entrenamiento

12. **¿Por qué λ = 100?**
    Es el valor del paper/tutorial: el L1 domina y obliga a ser fiel a la foto; el adversarial
    añade textura. Sin L1, el modelo inventa; sin adversarial, sale borroso.

13. **¿Qué pasa si el discriminador es demasiado fuerte?**
    El generador no recibe gradiente útil. Se baja `lr_D`, se usa label smoothing o TTUR.

14. **¿Cómo detectan overfitting?**
    Comparando L1/PSNR de train vs val y mirando las muestras de validación; early stopping por
    L1 de validación.

15. **¿Por qué dos optimizadores?**
    G y D tienen objetivos opuestos; se actualizan por separado. En el paso de G, D se congela.

16. **¿Cómo reanudan si Colab se desconecta?**
    `CheckpointManager` guarda cada época en Drive; `--resume` restaura G, D y optimizadores.

17. **¿Qué es mixed precision y por qué se usa?**
    `float16` en GPU para acelerar y ahorrar memoria; la pérdida se mantiene en `float32` por
    estabilidad.

### Resultados

18. **¿Qué mide PSNR y qué no?**
    Mide error cuadrático medio en dB; castiga desviaciones de color pero no siempre coincide con
    la percepción. Por eso se complementa con SSIM y evaluación visual.

19. **¿Por qué comparan con U-Net solo L1?**
    Para demostrar si la GAN aporta algo. Si Pix2Pix no mejora, hay que reportarlo con honestidad.

20. **¿Qué fallos encontraron?**
    Ejemplos: borra texturas finas, inventa detalles en zonas muy dañadas, color ligeramente
    distinto. Se documentan en la galería de fallos.

### Participación y repo

21. **¿Cómo sabemos que todos participaron?**
    GitHub Insights, `git shortlog`, PRs autorados/revisados, issues asignados y
    `docs/PARTICIPACION.md`.

22. **¿Qué hace el CI?**
    Lint, formato, tests, smoke training de 2 pasos y validación del título del PR. Bloquea el
    merge si algo falla.

23. **¿Por qué no entrenan en CI?**
    Porque no hay GPU y sería costoso/lento; el CI verifica que el pipeline no se rompa. El
    entrenamiento real se hace en Colab/Kaggle y se documenta.

24. **¿Pueden reproducir el resultado?**
    Sí: `configs/base.yaml`, semilla 42, tag `v1.0.0`, instrucciones en README y pesos
    descargables.

---

## 7. Hoja de trucos por integrante

Cada uno debe poder escribir de memoria:

```text
Entrada:        (B, 256, 256, 3) en [-1, 1]
Generador:      U-Net, 8 down/up, skip connections, tanh
Discriminador:  PatchGAN, 30x30 logits, parche 70x70
L_D:            0.5 * (BCE(real, 1) + BCE(fake, 0))
L_G:            BCE(fake, 1) + 100 * L1(fake, target)
Optimizador:    Adam, lr 2e-4, beta1 0.5
Batch:          1-4
Épocas:         100 con early stopping
Métricas:       PSNR, SSIM, L1
Dataset:        300 fuentes x 3 variantes = 900 pares
Splits:         80/10/10 por fuente
```

---

## 8. Cómo responder si no se sabe

- No inventar. Decir: "No lo medimos directamente; lo que sí observamos fue X. Podemos
  verificarlo en el repo/experimento Y."
- Si es una línea de código: abrir el archivo y explicar el contrato de entrada/salida, aunque no
  se recuerde el detalle.
- Si es una decisión: explicar las alternativas y por qué se eligió esa, con la evidencia
  disponible.
- Si es un error: reconocerlo y explicar cómo lo detectaron y qué harían para corregirlo.

---

## 9. Checklist de defensa

- [ ] Diapositivas en PDF y en el repo.
- [ ] Notas del orador.
- [ ] Demo local probada sin internet.
- [ ] Video/GIF de respaldo.
- [ ] Galería de resultados y fallos.
- [ ] Tabla de métricas impresa o visible.
- [ ] Diagrama de arquitectura legible.
- [ ] Cada integrante resolvió las 24 preguntas.
- [ ] Dos ensayos con presentador aleatorio.
- [ ] Cronómetro: 16 min de presentación + 4 de preguntas.
- [ ] Link del repo y del informe listos para compartir.
