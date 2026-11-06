# Informe — estructura, responsables y plan de escritura

## 1. Formato

- **Idioma:** español (términos técnicos en inglés entre paréntesis si aporta claridad).
- **Extensión:** 10–15 páginas + anexos (ajustar a la rúbrica del curso).
- **Formato:** IEEE (doble columna o una columna, según exigencia del profesor).
- **Fuente del documento:** `report/informe.md` + `report/referencias.bib`.
- **Exportación:** PDF con Pandoc (`make report`) y, si el curso lo exige, `.docx`.
- **Figuras:** generadas por código en `docs/img/`; en el informe se referencian.
- **Código:** no pegar archivos completos; solo fragmentos clave (pérdida, train_step).
- **Link del repo:** en la portada y en la sección de aplicación.
- **Commit de referencia:** indicar el tag `v1.0.0` o el hash usado para los resultados.

> **Regla:** el informe se escribe desde la semana 1. Cada sección se integra por PR a
> `report/`. Al final solo se corrige y se compila.

---

## 2. Estructura y responsables

| # | Sección | Contenido | Páginas | Redactor | Revisor |
|---|---|---|---|---|---|
| 0 | Portada | Curso, título, integrantes, fecha, repo | 1 | F | Todos |
| 1 | Resumen | Problema, método, resultados principales, conclusión (150–250 palabras) | 0.5 | F | D |
| 2 | Introducción | Motivación, problema, objetivos general y específicos, aportes | 1.5 | F + A | A |
| 3 | Marco teórico | GAN, minimax, Pix2Pix, U-Net, PatchGAN, pérdidas, métricas | 2.5 | C + B | B |
| 4 | Dataset | Recolección, ética, degradaciones, splits, estadísticas | 2 | A | E |
| 5 | Arquitectura | Diagrama, tablas de capas, hiperparámetros, pseudocódigo | 2.5 | B + C | D |
| 6 | Entrenamiento | Setup, hardware, configuración, checkpoints, estabilidad | 1.5 | D | B |
| 7 | Experimentos y resultados | Baselines, ablación, tablas, figuras, análisis | 2.5 | D + E | C |
| 8 | Aplicación/demo | Cómo ejecutar, capturas, limitaciones de uso | 1 | E | F |
| 9 | Discusión | Limitaciones, fallos, sesgos, trabajo futuro | 1 | Todos | F |
| 10 | Conclusiones | Respuesta a objetivos, aportes, aprendizajes | 0.5 | F | A |
| 11 | Referencias | IEEE, 10–20 fuentes | 0.5 | F | C |
| 12 | Anexos | Participación, estructura del repo, instrucciones, reflexiones | 1–2 | F + todos | A |

**Total:** ~17 páginas de contenido + anexos. Si la rúbrica pide menos, comprimir marco
teórico y anexos.

---

## 3. Contenido mínimo por sección

### 3.1 Resumen

- Qué problema se resolvió.
- Qué modelo y dataset se usaron.
- Métricas principales (PSNR/SSIM) y comparación con baseline.
- Limitación principal.

### 3.2 Introducción

- Por qué la restauración de fotos es relevante.
- Diferencia entre "mejorar una foto" y "alucinar detalles".
- Objetivo general: entrenar una GAN condicional para restaurar fotos degradadas.
- Objetivos específicos:
  1. Construir un dataset propio con pares degradado/limpio.
  2. Implementar Pix2Pix (U-Net + PatchGAN).
  3. Entrenar y validar con métricas objetivas.
  4. Desplegar una demo.
- Aportes: dataset, pipeline reproducible, evaluación con baselines, demo.

### 3.3 Marco teórico

- GAN: generador, discriminador, objetivo minimax.
- Pix2Pix: GAN condicional, pérdida adversarial + L1.
- U-Net: encoder-decoder con skip connections.
- PatchGAN: por qué parches de 70×70.
- Métricas: PSNR, SSIM, L1; qué mide cada una y sus limitaciones.
- Ecuaciones bien formateadas y referenciadas.

### 3.4 Dataset

- Fuentes y cantidad.
- Proceso de degradación (tabla de operaciones y probabilidades).
- Splits por fuente y prevención de fuga.
- Estadísticas y ejemplos visuales.
- Ética, consentimiento y licencias.
- Limitaciones del dataset sintético.

### 3.5 Arquitectura

- Diagrama del generador y del discriminador.
- Tabla de capas: tipo, filtros, kernel, stride, activación, salida.
- Número de parámetros.
- Justificación de InstanceNorm, dropout, `tanh`.
- Pseudocódigo del `train_step`.
- Contratos de tensores.

### 3.6 Entrenamiento

- Hardware (Colab/Kaggle/GPU local) y tiempo.
- Hiperparámetros y por qué.
- Checkpoints, reanudación, mixed precision.
- Estabilidad: label smoothing, clipping, early stopping.
- Curvas de pérdida (figura).

### 3.7 Experimentos y resultados

- Setup: splits, épocas, semilla, checkpoint elegido.
- Baselines: entrada, filtro clásico, U-Net L1, Pix2Pix.
- Tabla:

| Modelo | PSNR ↑ | SSIM ↑ | L1 ↓ | Tiempo |
|---|---|---|---|---|
| Entrada | ... | ... | ... | — |
| Filtro clásico | ... | ... | ... | ... |
| U-Net L1 | ... | ... | ... | ... |
| Pix2Pix | ... | ... | ... | ... |

- Figuras: `input | target | predicción` para 4–6 ejemplos.
- Galería de fallos: casos donde el modelo borra texturas o inventa detalles.
- Análisis: ¿la GAN mejora nitidez? ¿A costa de fidelidad? ¿Qué tipo de degradación falla más?

### 3.8 Aplicación/demo

- Cómo ejecutar (`pip install`, `python app/app.py`).
- Captura de la interfaz.
- Link a Hugging Face Spaces (si se desplegó).
- Limitaciones: tamaño de imagen, tipos de daño, rostros.

### 3.9 Discusión

- Limitaciones: dataset sintético, sesgo, resolución, costo.
- Riesgo de alucinación: el modelo puede "inventar" detalles que no estaban.
- Consideraciones éticas.
- Trabajo futuro: CycleGAN para datos no pareados, super-resolución, más datos reales,
  evaluación con usuarios, despliegue móvil.

### 3.10 Conclusiones

- ¿Se cumplieron los objetivos?
- Métricas vs baselines.
- Aprendizajes del equipo.
- Aporte reproducible.

---

## 4. Figuras y tablas mínimas

| Elemento | Fuente | Responsable |
|---|---|---|
| Diagrama U-Net | `docs/ARQUITECTURA.md` / draw.io | B |
| Diagrama PatchGAN | `docs/ARQUITECTURA.md` | C |
| Grid del dataset (input/target) | `docs/img/dataset_grid.png` | A |
| Curvas de pérdida | TensorBoard / CSV | D |
| Grid de resultados | `docs/img/results_grid.png` | E |
| Galería de fallos | `docs/img/failures.png` | E |
| Captura de la app | `docs/img/app.png` | E |
| Tabla de hiperparámetros | `configs/base.yaml` | D |
| Tabla de resultados | `metrics.csv` | E |
| Tabla de participación | `docs/PARTICIPACION.md` | F |

---

## 5. Plan de escritura por semana

| Semana | Qué se escribe | Quién | Estado |
|---|---|---|---|
| 1 | Esqueleto, introducción, marco teórico (borrador), dataset (protocolo) | F, C, A | |
| 2 | Dataset (versión 1), arquitectura, entrenamiento (borrador) | A, B, C, D | |
| 3 | Resultados preliminares, demo, discusión (borrador) | D, E, todos | |
| 4 | Resultados finales, conclusiones, referencias, revisión y PDF | Todos | |

**Meta semanal:** cada sección tiene al menos un borrador con texto real (no solo títulos) al
final de la semana 2.

---

## 6. Citas y referencias

Usar `report/referencias.bib` y estilo IEEE. Mínimo:

- Goodfellow et al., 2014 (GAN).
- Isola et al., 2017 (Pix2Pix).
- Ronneberger et al., 2015 (U-Net).
- Radford et al., 2016 (DCGAN) — opcional.
- Zhu et al., 2017 (CycleGAN) — para trabajo futuro.
- Documentación de TensorFlow/Keras.
- Paper de SSIM (Wang et al., 2004).
- Dataset: fuentes y licencias.

Citar figuras tomadas de otros sitios. No copiar texto sin cita. Si se usa una herramienta de
IA para redactar, revisar y declarar según la política del curso.

---

## 7. Build del PDF

`report/Makefile`:

```makefile
report:
	pandoc informe.md \
	  --bibliography=referencias.bib \
	  --citeproc \
	  --template=eisvogel \
	  -o informe.pdf

docx:
	pandoc informe.md \
	  --bibliography=referencias.bib \
	  --citeproc \
	  -o informe.docx
```

Requisitos: Pandoc + LaTeX (o `eisvogel`). Alternativa sin LaTeX: exportar a `.docx` y luego a
PDF desde Word.

En CI, el workflow `release.yml` construye el PDF al crear el tag. Si LaTeX no está disponible,
se adjunta el `.docx` y se documenta.

---

## 8. Checklist del informe

- [ ] Portada con nombres, correos y link del repo.
- [ ] Resumen con resultados concretos.
- [ ] Objetivos general y específicos.
- [ ] Marco teórico con ecuaciones y citas.
- [ ] Dataset con cantidad, splits y ética.
- [ ] Diagramas de arquitectura.
- [ ] Tabla de hiperparámetros.
- [ ] Tabla de resultados con baselines y media ± std.
- [ ] Figuras cualitativas y galería de fallos.
- [ ] Discusión de limitaciones (incluida alucinación).
- [ ] Conclusiones que responden a los objetivos.
- [ ] Referencias IEEE.
- [ ] Anexo de participación con enlaces a PRs/commits.
- [ ] PDF con figuras legibles y sin texto cortado.
- [ ] Revisión cruzada por al menos 2 integrantes.
- [ ] Verificar que el PDF corresponde al tag `v1.0.0`.
