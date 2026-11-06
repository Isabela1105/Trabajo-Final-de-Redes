# Merge, integración y revisión final

> Este documento lo usa principalmente `@Isabela1105` (coordinación), pero cualquier integrante
> debe entenderlo para revisar y mergear.

---

## 1. Preparación del repositorio (día 1)

### 1.1 Invitar al equipo

GitHub → repo → **Settings → Collaborators → Add people**:

- `@Isabellaortizher-collab`
- `@Juanpablodavila`
- `@JeremyCaicedo`
- `@CristianMP355`
- `@Juansec01`
- `@Isabela1105` (owner)

Rol: **Write** (pueden crear ramas y PRs, no cambiar settings).

### 1.2 Crear estructura inicial

Subir el contenido de `plantillas/` a la raíz del repo (o hacer el PR `chore/01-init`).

### 1.3 Proteger `main`

**Settings → Branches → Add branch protection rule**:

- [x] Require a pull request before merging.
- [x] Require approvals: **1**.
- [x] Dismiss stale approvals.
- [x] Require status checks: `Calidad y pruebas`, `Título de PR`.
- [x] Require conversation resolution.
- [x] Require linear history.
- [x] Do not allow bypassing.
- [x] Automatically delete head branches.

Si el repo es privado y no aparece la opción, hacerlo público o usar GitHub Education. Mientras
tanto, regla de equipo: **nadie mergea sin PR y CI verde**.

### 1.4 Labels y milestones

Labels: `area:data`, `area:model`, `area:train`, `area:eval`, `area:ci`, `area:docs`,
`priority:high`, `priority:low`, `blocked`, `bug`, `enhancement`.

Milestones: `M1 Fundaciones`, `M2 Primer entrenamiento`, `M3 Experimentos`, `M4 Entrega`.

### 1.5 Tablero

Columnas: `Backlog`, `Esta semana`, `En progreso`, `En revisión`, `Hecho`.

Crear un issue por cada PR del plan (`docs/PLAN_COMMITS.md`) y asignarlo.

---

## 2. Orden de integración

### Semana 1 — Fundaciones

| Orden | PR | Dueño | Depende de | Revisor |
|---|---|---|---|---|
| 1 | F1 init + CI | Isabela | — | Isabella |
| 2 | A1 protocolo dataset | Isabella | — | Isabela |
| 3 | B1 contrato modelos | Juan Pablo | — | Jeremy |
| 4 | C1 contrato discriminador | Jeremy | — | Juan Pablo |
| 5 | D1 config | Cristian | — | Juan Sebastian |
| 6 | E1 protocolo eval | Juan Sebastian | — | Juan Pablo |
| 7 | F2 tooling + plantillas | Isabela | F1 | Juan Pablo |
| 8 | F3 datos sintéticos | Isabela | F1 | Isabella |
| 9 | A2 dataset pairs | Isabella | D1 | Juan Pablo |
| 10 | B2 generador | Juan Pablo | D1 | Jeremy |
| 11 | C2 PatchGAN + losses | Jeremy | D1 | Juan Pablo |
| 12 | D2 train loop | Cristian | A2, B2, C2 | Juan Sebastian |
| 13 | E2 métricas + inferencia | Juan Sebastian | D1 | Jeremy |
| 14 | A3 build script | Isabella | A2 | Isabela |
| 15 | B3 docs capas | Juan Pablo | B2 | Isabela |
| 16 | C3 docs pérdidas | Jeremy | C2 | Isabela |
| 17 | D3 smoke test | Cristian | D2 | Isabela |
| 18 | E3 docs baselines | Juan Sebastian | E2 | Isabela |

**Cierre semana 1:** `pytest` verde, smoke training verde, cada integrante con al menos 2 commits.

### Semana 2 — Integración

| Orden | PR | Dueño | Depende de | Revisor |
|---|---|---|---|---|
| 19 | A4 dataset v2 | Isabella | A3 | Juan Pablo |
| 20 | B4 refactor generador | Juan Pablo | B2 | Jeremy |
| 21 | C4 opciones de pérdida | Jeremy | C2 | Juan Pablo |
| 22 | D4 integración + mixed precision + early stopping | Cristian | A4, B4, C4 | Juan Sebastian |
| 23 | E4 eval + app | Juan Sebastian | D4 | Cristian |
| 24 | F4 quality gates | Isabela | F2 | Cristian |

**Cierre semana 2:** primera imagen restaurada (aunque sea mala), app mínima, CI con cache.

### Semana 3 — Experimentos

| Orden | PR | Dueño | Depende de | Revisor |
|---|---|---|---|---|
| 25 | A5 dataset real + licencias | Isabella | A4 | Juan Sebastian |
| 26 | B5 tuning + guion | Juan Pablo | B4 | Cristian |
| 27 | C5 gradient penalty + guion | Jeremy | C4 | Cristian |
| 28 | D5 experimentos + ablación | Cristian | D4 | Juan Sebastian |
| 29 | E5 resultados + app final | Juan Sebastian | D4 | Isabela |
| 30 | F5 informe + release | Isabela | D5, E5 | Juan Pablo |

**Cierre semana 3:** tabla PSNR/SSIM/L1, figuras, app desplegada, borrador del informe.

### Semana 4 — Cierre

| Orden | PR | Dueño | Depende de | Revisor |
|---|---|---|---|---|
| 31 | A6 dataset final | Isabella | A5 | Cristian |
| 32 | B6 bordes + tests | Juan Pablo | B5 | Isabella |
| 33 | C6 NaN + ecuaciones | Jeremy | C5 | Juan Pablo |
| 34 | D6 checkpoints Colab + HP finales | Cristian | D5 | Juan Sebastian |
| 35 | E6 formatos + figuras finales | Juan Sebastian | E5 | Juan Pablo |
| 36 | F6 participación + release + checklist | Isabela | Todos | Isabella, Jeremy |

**Día 25:** congelamiento. Solo docs/defensa.
**Día 26–28:** ensayos y correcciones del informe.

---

## 3. Cómo mergear un PR

```bash
# 1. Verificar que el PR está actualizado con main
git fetch origin
git switch <rama>
git rebase origin/main
# resolver conflictos, probar
git push --force-with-lease

# 2. Revisar en GitHub
# - CI verde
# - 1 aprobación (2 en rutas críticas)
# - conversaciones resueltas
# - sin archivos grandes

# 3. Squash and merge
# Título: el mismo del PR (Conventional Commits)
# Borrar rama: sí
```

Después del merge:

```bash
git switch main
git pull --rebase
```

---

## 4. Revisión de código: checklist

### 4.1 Correctitud

- [ ] El código hace lo que dice el issue.
- [ ] Los contratos de tensores se respetan.
- [ ] No hay `sigmoid` + `from_logits=True`.
- [ ] No hay fuga de datos (split por fuente).
- [ ] Los optimizadores actualizan las variables correctas.
- [ ] El checkpoint guarda G, D y optimizadores.
- [ ] No hay `NaN` en pérdidas.

### 4.2 Calidad

- [ ] Nombres claros en inglés; comentarios en español.
- [ ] Funciones pequeñas y con docstring.
- [ ] Sin código muerto ni imports sin usar.
- [ ] Sin duplicación evidente.
- [ ] Semillas fijas y configuración en YAML.

### 4.3 Tests

- [ ] Tests nuevos para el cambio.
- [ ] Formas, rangos y casos borde.
- [ ] Smoke test sigue pasando.
- [ ] `pytest -q` local y CI verdes.

### 4.4 Documentación

- [ ] `docs/ARQUITECTURA.md` actualizado si cambió una interfaz.
- [ ] `README.md` actualizado si cambió un comando.
- [ ] `docs/DECISIONES.md` si es una decisión técnica.
- [ ] `docs/PARTICIPACION.md` al final de la semana.

### 4.5 Integridad

- [ ] Sin datos, pesos, `.env` ni archivos > 5 MB.
- [ ] Sin credenciales.
- [ ] Sin cambios ajenos no relacionados.
- [ ] El PR tiene menos de ~400 líneas (si no, se divide).

### 4.6 Defensa

- [ ] ¿Podrías explicar este código sin mirar?
- [ ] ¿Sabes por qué se eligió así y no de otra forma?
- [ ] ¿Sabes qué pasa si se rompe?

Si la respuesta es no, no se aprueba todavía.

---

## 5. Resolución de conflictos

### Conflicto típico en `requirements.txt`

1. Rebase contra `main`.
2. Conservar ambas dependencias si no se contradicen.
3. Ordenar alfabéticamente.
4. Probar instalación y tests.
5. Avisar en el PR.

### Conflicto en `docs/ARQUITECTURA.md`

1. No borrar secciones de otros.
2. Integrar ambos cambios.
3. Pedir revisión a los dos dueños.

### Conflicto en un notebook

1. Evitar editar el mismo notebook en dos ramas.
2. Si pasa, reconstruir la celda conflictiva a mano.
3. Guardar sin outputs (`nbstripout`).

### Regla general

- Nunca resolver un conflicto "aceptando todo" sin leer.
- Después de resolver: `pytest -q` y smoke.
- Si no se entiende, pedir pairing.

---

## 6. Revisión final de todos los aspectos

### 6.1 Repositorio

- [ ] Todos los integrantes aparecen en **Insights → Contributors**.
- [ ] Cada uno tiene 14–15 commits significativos.
- [ ] Cada uno tiene 6 PRs y 4–5 revisiones.
- [ ] `docs/PARTICIPACION.md` coincide con GitHub.
- [ ] `main` protegida y CI verde.
- [ ] No hay ramas abiertas sin mergear.
- [ ] No hay secretos, datos personales ni archivos grandes.
- [ ] Tag `v1.0.0` y release con notas.

### 6.2 Código

- [ ] `pytest -q` verde.
- [ ] `ruff check .` y `black --check .` verdes.
- [ ] Smoke training funciona en CPU.
- [ ] `configs/base.yaml` reproduce el resultado.
- [ ] Checkpoint del generador descargable.
- [ ] `infer.py` funciona con una imagen nueva.
- [ ] `app/app.py` funciona localmente.

### 6.3 Dataset

- [ ] 150+ fuentes, 450+ pares.
- [ ] Splits por fuente, sin fuga.
- [ ] Manifiesto con semilla y parámetros.
- [ ] `check_dataset.py` pasa.
- [ ] Ficha con licencias y consentimiento.
- [ ] Ejemplos visuales en `docs/img/`.

### 6.4 Modelo y entrenamiento

- [ ] Generador y discriminador cumplen contratos.
- [ ] Pérdidas implementadas y probadas.
- [ ] Entrenamiento con checkpoints y reanudación.
- [ ] Curvas de pérdida documentadas.
- [ ] Hiperparámetros justificados.
- [ ] Baselines y ablación.

### 6.5 Resultados

- [ ] Tabla PSNR/SSIM/L1 con media ± std.
- [ ] Comparación con entrada, filtro clásico y U-Net L1.
- [ ] Grid `input | target | predicción`.
- [ ] Galería de fallos.
- [ ] Conclusiones coherentes con las métricas.

### 6.6 Demo

- [ ] Gradio funciona sin internet.
- [ ] Ejemplos incluidos.
- [ ] Captura en `docs/img/app.png`.
- [ ] Enlace a HF Spaces (si se desplegó).
- [ ] Video/GIF de respaldo.

### 6.7 Informe

- [ ] PDF con portada, equipo, repo y tag.
- [ ] Resumen con resultados concretos.
- [ ] Marco teórico con ecuaciones y citas.
- [ ] Dataset con ética y limitaciones.
- [ ] Arquitectura con diagramas y tablas.
- [ ] Resultados con baselines.
- [ ] Discusión de alucinación y sesgos.
- [ ] Referencias IEEE.
- [ ] Anexo de participación.
- [ ] Revisado por 2 integrantes.

### 6.8 Defensa

- [ ] Diapositivas en PDF.
- [ ] Notas del orador.
- [ ] Cada integrante resolvió las 24 preguntas.
- [ ] Dos ensayos con presentador aleatorio.
- [ ] Demo de respaldo.
- [ ] Link del repo y del informe.

---

## 7. Release final

```bash
git switch main
git pull --rebase
git tag -a v1.0.0 -m "Entrega final: Pix2Pix para restauración de fotografías"
git push origin v1.0.0
```

Verificar:

- [ ] El workflow `Release` corrió.
- [ ] El release tiene el informe (DOCX/PDF).
- [ ] El README apunta al release.
- [ ] Los pesos están enlazados (Release/HF/Drive).
- [ ] El tag coincide con el commit usado en el informe.

---

## 8. Plan de contingencia

| Problema | Acción |
|---|---|
| Un PR queda rojo y no se arregla | Revertir el merge (`git revert`) y reabrir el PR. |
| Un integrante no puede terminar | El respaldo toma el PR; el autor revisa y explica. |
| El entrenamiento no converge | Bajar a 128×128, revisar λ, label smoothing, TTUR. |
| No hay GPU | Colab/Kaggle; reducir épocas y batch. |
| El informe no compila | Exportar DOCX; adjuntar PDF local. |
| Se acaba el tiempo | Congelar código el día 24; priorizar informe y defensa. |
| El repo se vuelve privado sin protección | Hacerlo público o aplicar la regla de equipo. |
