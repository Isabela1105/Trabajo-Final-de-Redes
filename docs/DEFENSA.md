# Guía de sustentación

## 1. Lo que evalúa el profesor

- ¿Entienden la arquitectura y el código?
- ¿El dataset es propio y está bien construido?
- ¿Hay validación real (métricas + evidencia visual)?
- ¿Todos participaron y todos pueden explicar?
- ¿El repositorio está ordenado y reproducible?

### 1.1 Rúbrica de la entrega anterior y qué responde a cada punto

| Componente | Nota anterior | Qué mostrar ahora | Dónde |
|---|---|---|---|
| Dataset | 5.0 (pondera 4.0 si es público) | Dataset **propio**: 152 fotos fuente (100 DIV2K, 30 Pexels, 22 propias), 456 pares, split por foto fuente sin fuga, `check_dataset.py` = OK | `docs/DATASET.md`, `docs/dataset_stats.md` |
| Arquitectura | 4.5 | U-Net 8 niveles (54.5 M parámetros) + PatchGAN 70×70; por qué cada pieza | `docs/ARQUITECTURA.md` |
| Paso a paso | 4.5 | Pipeline reproducible: degradar → entrenar → elegir por validación → evaluar → demo | README §6–9 |
| Métricas | 2.0 / 1.0 (otra tarea) | **21.30 dB, SSIM 0.7165, L1 0.0681**: +1.00 dB sobre la entrada (antes +0.37), mejor SSIM que la entrega anterior en las 45 fotos; ablación justa (§7.4) | `docs/RESULTADOS.md` §6–7 |
| Análisis | 3.5 | Análisis de errores por nivel de daño y por tipo; comparación pareada con t; por qué la última época no es la mejor | `docs/RESULTADOS.md` §6.4 |
| Demo | **0.0** | Demo web local funcionando, con original de referencia, PSNR/SSIM en vivo, limpieza de polvo y TTA | `docs/DEMO.md` |
| Exposición | 3.5 | Todos dominan todo: banco de preguntas de la sección 6 | este documento |

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
| 11–14 | Resultados | E | Tabla final, 20 vs 100 épocas, selección por validación, TTA, análisis de errores. |
| 14–16 | Demo y limitaciones | E + A | Demo web (FastAPI), ejemplos del test, alucinación, trabajo futuro. |
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

- **Plan A:** demo web local (`python -m app.server --checkpoint outputs/pix2pix_100ep_mejor/best`) con
  el selector de ejemplos del test. Si falla el navegador, `python -m app.gradio_app` (respaldo).
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

### Nueva sustentación (las que más van a preguntar)

21. **¿Cuántas épocas y cuántas imágenes?**
    Objetivo 100 épocas; el modelo vigente es el de la **época 48**, la de menor L1 de
    validación. Dataset: 152 fotos fuente × 3 variantes = **456 pares** (366 train, 45 val,
    45 test), 256×256. Con lote 2 son 183 iteraciones por época.

22. **¿Por qué no usan la época 100?**
    Porque en una GAN la calidad oscila: la corrida que completó 100 épocas terminó en
    20.41 dB en test, peor que la de 20 épocas (20.67). Se elige la época con mejor
    **validación**; el test nunca se usa para elegir, solo para reportar.

23. **¿Qué es TTA y por qué no es trampa?**
    Se restaura la foto y sus 3 versiones volteadas y se promedian. No cambia el modelo ni ve el
    original. Se decidió con validación (+0.05 dB, +0.014 SSIM) antes de mirar el test.

24. **¿La mejora es real o es ruido?**
    Comparación pareada por foto frente a la entrega anterior: +0.63 dB con t = 4.75, y mejor
    SSIM en las 45 imágenes. Frente a la entrada: +1.00 dB.

25. **¿Dónde falla?**
    En fotos poco dañadas (entrada ≥ 22 dB) empeora 10 de 11 (−2.5 dB): nunca vio pares sin
    daño y sobre-corrige color y brillo. Correlación entrada–ganancia: −0.71. Solución
    propuesta: agregar pares limpio→limpio.

26. **¿Por qué la imagen sigue borrosa?**
    La red trabaja a 256×256 con pérdida L1, que promedia lo que no puede inferir. La casilla
    *Realzar nitidez* la hace más definida a la vista, pero baja PSNR y SSIM (también realza el
    ruido); por eso no entra en la tabla.

27. **¿Entonces la GAN sirve?**
    Con el mismo protocolo, la U-Net solo L1 empata en PSNR y L1 y gana en SSIM (§7.4). Es lo
    esperado: la pérdida adversarial busca realismo y bordes, no menor error por píxel (Isola et
    al., 2017). Además el discriminador gana muy pronto (pérdida cerca de 0) y su señal es débil.
    Elegimos Pix2Pix porque queda marginalmente mejor en validación, que es el criterio fijado.

28. **¿Qué probaron que no funcionó?**
    Pares sin daño para que no sobre-corrija: peor en validación y en prueba, se descartó (§7.1).
    Mezclar salida y entrada: mejora pequeña y dudosa en validación, no se adoptó (§7.2). Reportar
    lo que no funcionó también es parte del análisis.

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

### Bloque de entrenamiento (guion del Rol D, ~90 s)

> "El bucle vive en `src/gan/train.py`. Usamos el `make_dataset` de Isabella ya mergeado en
> `main`, con `tf.ensure_shape` y una firma fija, y validamos en grafo que cada imagen esté en
> [-1,1]. Los builders reales de Juan Pablo y Jeremy se importan por ruta desde el YAML y sus
> pérdidas se adaptan aquí a la firma del bucle sin modificar sus archivos. En cada paso
> actualizamos **primero el discriminador**: congelamos G evaluándolo con `training=False`,
> pasamos su salida con `stop_gradient` a D y aplicamos su Adam. **Después actualizamos G** con
> D en `training=False`, para que no se mueva ni cambie sus estadísticas; el gradiente de la
> pérdida adversarial llega a G a través de D. Usamos dos `GradientTape` y dos Adams con
> lr 2e-4 y beta1 0.5, con clip global de 1.0. En GPU activamos `mixed_float16`, pero las
> pérdidas y el L1 se calculan en float32 para que no aparezcan NaNs. Cada época escribimos
> escalares e imágenes en TensorBoard, una fila por split en `metrics.csv` y un checkpoint con
> G, D, ambos optimizadores, época y semilla; junto a él guardamos el YAML y su hash de forma
> atómica. Si Colab se cae, `--resume` lee el `metadata.json` y sigue en la época siguiente.
> Para demostrar que la GAN aporta algo corremos `configs/ablation_l1.yaml`, que es lo mismo
> pero con `use_gan: false` (solo L1)."

Preguntas cruzadas que pueden caer en este bloque:

- **¿Por qué D primero y no los dos a la vez?** Para que G reciba una señal adversarial del D
  ya actualizado y para que los gradientes de una red nunca modifiquen la otra.
- **¿Por qué la pérdida en float32 si el modelo va en float16?** BCE con logits extremos y L1
  pueden desbordarse en fp16; calcular la pérdida en fp32 mantiene la estabilidad sin perder
  la velocidad de la convolución en fp16.
- **¿Qué pasa si el discriminador domina?** Se baja `lr_d` o se reduce su número de pasos; se
  detecta porque `loss_d → 0` mientras `loss_g` crece. El etiquetado suave no está disponible
  en la `discriminator_loss` real (no expone ese parámetro): `_call_discriminator_loss` lo
  ignora con un aviso y habría que pedirlo en `src/gan/losses.py`, que no es nuestro archivo.
- **¿Qué tienen que coincidir para reanudar?** La arquitectura (viene del YAML + `build_generator`)
  y la forma de los optimizadores; por eso se guarda `config.yaml` junto al checkpoint y se
  compara su hash.

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
