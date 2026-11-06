# Guía individual — Isabela (`@Isabela1105`)

**GitHub:** `@Isabela1105` (owner del repositorio)
**Código:** _(completar — pendiente de confirmar)_
**Rol:** CI/CD, integración, informe y cierre
**Respaldo:** Isabella Ortiz (`@Isabellaortizher-collab`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `docs/CI_CD.md`, `docs/MERGE_Y_REVISION.md`, `docs/INFORME.md` y `CONTRIBUTING.md`.

---

## 1. Tu misión

Eres la **coordinadora técnica y de integración**. Tu trabajo no es escribir el modelo, es hacer
que el modelo de los demás exista, se pruebe, se integre y se entregue:

- Inicializar el repositorio y el tooling (lint, formato, pre-commit).
- Crear el CI que **bloquea merges rotos**.
- Proteger `main` y definir el flujo de PRs.
- Mantener el orden de merge entre las 6 personas.
- Coordinar el informe y el release `v1.0.0`.
- Verificar la participación de todos.
- Dirigir la revisión final y los ensayos de defensa.

Eres la única persona que puede tocar `.github/` sin pedir permiso, y la que más revisiones hace.
**No mergees tu propio PR** y no saltes el CI ni con prisa.

---

## 2. Archivos que posees

```text
.github/workflows/ci.yml
.github/workflows/release.yml
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/tarea.md
.github/CODEOWNERS
.github/dependabot.yml
README.md
CONTRIBUTING.md
LICENSE
.gitignore
.pre-commit-config.yaml
pyproject.toml
requirements.txt
requirements-dev.txt
Makefile
docs/CI_CD.md
docs/PARTICIPACION.md
docs/EQUIPO.md
docs/MERGE_Y_REVISION.md
report/
scripts/make_synthetic_sample.py
```

No edites el código de datos, modelos, pérdidas, train, eval ni app sin avisar a su dueño. Si un
PR ajeno necesita un cambio en CI, se coordina en el issue.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `chore/01-init` | PR-F1 | `chore: inicializar repositorio con README, licencia y gitignore`<br>`ci: agregar workflow de lint, tests y smoke training` | Isabella |
| 1 | 2 | `chore/02-tooling` | PR-F2 | `chore: agregar pre-commit y configuración de calidad`<br>`docs: agregar guía de contribución y plantillas` | Juan Pablo, Jeremy |
| 1 | 3 | `test/03-synthetic-data` | PR-F3 | `test(ci): agregar generador de datos sintéticos para pruebas` | Isabella |
| 2 | 8–11 | `ci/29-quality-gates` | PR-F4 | `ci: agregar cache, cobertura y artefactos`<br>`ci: agregar verificación de título de PR`<br>`docs(ci): documentar protección de rama y flujo`<br>`chore: configurar Dependabot y etiquetas` | Cristian, Juan Sebastian |
| 3 | 15–18 | `ci/37-report-release` | PR-F5 | `ci: agregar build del informe en PDF`<br>`docs: actualizar README con resultados y demo`<br>`ci: agregar release automático en tags` | Juan Pablo, Juan Sebastian |
| 4 | 20–24 | `docs/40-participacion` | PR-F6 | `docs: crear tabla de participación y evidencia`<br>`ci: congelar versiones y validar release`<br>`docs: revisión final y checklist de entrega` | Isabella, Jeremy |

**Total:** 6 PRs, 15 commits.

> El plan de commits está calibrado para que **todos** tengan al menos 12 commits significativos y
> 6 PRs. No regales commits triviales; cada PR debe tener contenido real.

---

## 4. Día 1: arranque del repositorio

### 4.1 Subir la plantilla

Copia el contenido de `plantillas/` a la raíz del repo:

```powershell
Copy-Item -Recurse -Force .\plantillas\* .\Trabajo-Final-de-Redes\
```

Luego:

```bash
cd Trabajo-Final-de-Redes
git switch -c chore/01-init
git add .
git commit -m "chore: inicializar repositorio con README, licencia y gitignore (#1)"
git push -u origin HEAD
```

Abre el PR con la plantilla, asígnate a ti misma como revisora **no** (no puedes aprobar tu PR):
pide a Isabella Ortiz o Juan Pablo que lo revise. Si es el primer PR y nadie puede revisar aún,
usa una excepción documentada en el PR y mergea con `--admin` solo una vez, dejando constancia.

### 4.2 Invitar al equipo

**Settings → Collaborators → Add people** con rol **Write**:

- `@Isabellaortizher-collab`
- `@Juanpablodavila`
- `@JeremyCaicedo`
- `@CristianMP355`
- `@Juansec01`

### 4.3 Proteger `main`

**Settings → Branches → Add branch protection rule** para `main`:

- [x] Require a pull request before merging.
- [x] Require approvals: **1**.
- [x] Dismiss stale pull request approvals when new commits are pushed.
- [x] Require review from Code Owners.
- [x] Require status checks to pass: `Calidad y pruebas`, `Título de PR`.
- [x] Require conversation resolution.
- [x] Require linear history.
- [x] Do not allow bypassing the above settings.
- [x] Automatically delete head branches.

> Si el repo es privado y GitHub no ofrece protección de rama, hazlo público con
> **Settings → General → Danger Zone → Change visibility**. Si el curso exige privado, aplica la
> regla de equipo: nadie mergea sin PR ni CI verde, y se evidencia en `docs/PARTICIPACION.md`.

### 4.4 Labels, milestones y tablero

Labels: `area:data`, `area:model`, `area:train`, `area:eval`, `area:ci`, `area:docs`,
`priority:high`, `priority:low`, `blocked`, `bug`, `enhancement`.

Milestones: `M1 Fundaciones`, `M2 Primer entrenamiento`, `M3 Experimentos`, `M4 Entrega`.

Tablero: `Backlog`, `Esta semana`, `En progreso`, `En revisión`, `Hecho`.

Crea un issue por cada PR del plan (36 issues) y asígnalo antes del día 2.

---

## 5. El CI que debes construir

### 5.1 `ci.yml` — en cada push y PR

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Calidad y pruebas
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: black --check .
      - run: pytest -q --cov=src --cov-report=xml
      - run: python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
      - run: python -m src.gan.train --config configs/smoke.yaml --max-steps 2
```

### 5.2 Verificación del título del PR

```yaml
  pr-title:
    name: Título de PR
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: amannn/action-semantic-pull-request@v5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Patrón aceptado: `tipo(area): descripción`. Tipos: `feat`, `fix`, `docs`, `test`, `refactor`,
`perf`, `ci`, `chore`.

### 5.3 `release.yml` — en tags `v*`

```yaml
name: Release
on:
  push:
    tags: ["v*"]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements-dev.txt
      - run: python scripts/build_report.py || echo "Sin build de informe todavía"
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            report/informe.pdf
            report/informe.docx
```

### 5.4 Reglas del CI

1. Debe terminar en **menos de 10 minutos**. Si tarda más, se cachea pip y se reduce el smoke.
2. No entrena de verdad: 2 pasos, batch 1, 64×64.
3. No descarga el dataset completo: usa `data/sample/`.
4. Un check rojo **bloquea** el merge; no se ignora.
5. Si el CI falla por una dependencia, se arregla en un PR de CI, no se desactiva el check.

---

## 6. Semana por semana

### Semana 1 — Fundaciones

- Subir plantilla, README, LICENSE, `.gitignore`.
- CI con lint + tests + smoke.
- Pre-commit y `pyproject.toml`.
- `CONTRIBUTING.md`, plantilla de PR e issues.
- `scripts/make_synthetic_sample.py`: genera 8 imágenes sintéticas reproducibles (semilla fija)
  en `data/sample/{input,target}` para que el CI no dependa del dataset real.
- Verificar que el primer PR de cada compañero tenga CI verde y revisor asignado.

**Cierre de semana 1:** los 6 tienen rama, PR y al menos 2 commits; `main` protegida; CI verde.

### Semana 2 — Quality gates

- Cache de pip y de datos sintéticos.
- Cobertura con `--cov=src`, reportar pero no bloquear por umbral todavía.
- Artefactos: `metrics.csv` y figuras del smoke.
- Verificación de título de PR.
- Dependabot para `pip` y GitHub Actions.
- `docs/CI_CD.md` con capturas y solución de problemas.
- Revisar que el PR de integración de Cristian (D4) no rompa el smoke.

### Semana 3 — Informe y release

- Workflow de release en tags.
- Script de build del informe (Pandoc o `python-docx`); si falla, adjuntar PDF manualmente.
- Actualizar README con resultados reales y enlace a la demo.
- Verificar que los números del README coincidan con `metrics.csv` (no aceptar placeholders).
- Preparar `report/informe.md` con las secciones de todos.

### Semana 4 — Cierre

- `docs/PARTICIPACION.md` con enlaces reales a PRs y commits.
- Congelar versiones en `requirements.txt`.
- Revisión final completa con `docs/MERGE_Y_REVISION.md`.
- Tag `v1.0.0` y release.
- Ensayos de defensa con presentador aleatorio.
- Checklist de entrega del README, punto por punto.

---

## 7. Orden de merge que debes hacer cumplir

| Semana | Orden |
|---|---|
| 1 | F1 → A1 → B1 → C1 → D1 → E1 → F2 → F3 → A2 → B2 → C2 → D2 → E2 → A3 → B3 → C3 → D3 → E3 |
| 2 | A4 → B4 → C4 → D4 → E4 → F4 |
| 3 | A5 → B5 → C5 → D5 → E5 → F5 |
| 4 | A6 → B6 → C6 → D6 → E6 → F6 |

Reglas de ventana:

- **Días 1–2:** solo fundaciones (F1, A1, B1, C1, D1, E1).
- **Días 3–5:** esqueletos (F2, F3, A2, B2, C2, D2, E2).
- **Día 6:** cierre semana 1 (A3–E3).
- **Días 8–11:** integración (A4–F4).
- **Días 15–20:** experimentos (A5–F5).
- **Días 22–24:** cierre de código (A6–F6).
- **Día 25:** congelamiento y tag.
- **Días 26–28:** solo informe y defensa.

Si un PR llega fuera de ventana, espera. Si bloquea a alguien, se permite avanzar con mocks o
datos sintéticos (ver `docs/PLAN_COMMITS.md`, sección 10).

---

## 8. Cómo revisar y mergear (tu rutina)

```bash
# 1. Traer el PR
git fetch origin
git switch <rama-del-pr>
git rebase origin/main
pytest -q

# 2. Revisar en GitHub
# - CI verde
# - 1 aprobación (2 en rutas críticas)
# - conversaciones resueltas
# - sin archivos > 5 MB, sin datos ni pesos
# - título en Conventional Commits

# 3. Squash and merge, borrar rama

# 4. Actualizar local
git switch main
git pull --rebase
```

Checklist de revisión completo en `docs/MERGE_Y_REVISION.md` (sección 4). Si no puedes explicar
el código del PR, no lo apruebes: pide una explicación o pairing.

---

## 9. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-A1 / PR-A3 | Isabella | Que el protocolo y el script de dataset sean reproducibles y no versionen datos. |
| PR-B3 | Juan Pablo | Que la tabla de capas coincida con el código y el conteo de parámetros. |
| PR-C3 | Jeremy | Que las ecuaciones estén bien y que BCE use logits. |
| PR-D3 | Cristian | Que el smoke test corra en CPU y no dependa de data real. |
| PR-E4 / PR-E5 | Juan Sebastian | Que la app cargue el checkpoint correcto y que las métricas estén en `[0,1]`. |
| PR-F6 (tu propio PR) | — | Lo revisan Isabella, Jeremy y Juan Sebastian; tú no lo apruebas. |

Mínimo 5 revisiones reales. Una revisión no es "LGTM": señala al menos un punto concreto
(nombre, test, comentario o riesgo).

---

## 10. Informe y release

### 10.1 Responsabilidad sobre el informe

- Eres la editora: recibes las secciones, unificas formato, numeración y referencias.
- Verificas que cada sección tenga su autor en `docs/PARTICIPACION.md`.
- Revisas que las cifras del informe coincidan con `metrics.csv` y con el README.
- Generas el PDF final y lo subes a `report/informe.pdf`.

Estructura y responsables en `docs/INFORME.md`.

### 10.2 Release

```bash
git switch main
git pull --rebase
git tag -a v1.0.0 -m "Entrega final: Pix2Pix para restauración de fotografías"
git push origin v1.0.0
```

Verifica que el workflow `Release` corra y publique el PDF. Si falla, adjunta el PDF manualmente
al release y documenta el problema en el informe (no ocultes el fallo).

---

## 11. Qué debes poder explicar en la defensa

Aunque tu rol es CI/integración, el docente puede preguntarte cualquier cosa:

1. ¿Qué hace el CI paso a paso y por qué bloquea el merge?
2. ¿Por qué el smoke entrena solo 2 pasos y en CPU?
3. ¿Cómo está protegida `main`?
4. ¿Qué es un squash merge y por qué se usa?
5. ¿Cómo se garantiza que los 6 participaron?
6. ¿Qué pasa si un PR rompe `main`? (revert + PR de corrección)
7. ¿Cómo se reproduce el resultado? (tag, semilla, config, pesos)
8. ¿Qué contiene el release?
9. ¿Por qué no se versiona el dataset?
10. ¿Cuál es la mayor limitación del proyecto?

Además, debes poder explicar **la arquitectura, las pérdidas, el dataset, el entrenamiento y las
métricas** a nivel de guion: eres la última persona que puede decir "eso lo hizo otro".

---

## 12. Entregables

- [ ] Repo inicializado, 6 colaboradores, `main` protegida.
- [ ] `ci.yml` y `release.yml` funcionando.
- [ ] Pre-commit, `pyproject.toml`, `CONTRIBUTING.md`, plantillas.
- [ ] `scripts/make_synthetic_sample.py`.
- [ ] `docs/CI_CD.md` y `docs/PARTICIPACION.md`.
- [ ] Informe unificado y PDF.
- [ ] Release `v1.0.0` con PDF.
- [ ] 15 commits y 6 PRs.
- [ ] 5 revisiones a otros.
- [ ] Dos ensayos de defensa coordinados.

---

## 13. Checklist final de entrega

- [ ] Los 6 aparecen en **Insights → Contributors** con 12+ commits.
- [ ] Los 6 tienen 6 PRs y al menos 4 revisiones.
- [ ] `main` sin commits directos.
- [ ] CI verde en el último commit de `main`.
- [ ] Tag `v1.0.0` y release con PDF.
- [ ] `README.md` con tabla de resultados real y enlaces vivos.
- [ ] `docs/PARTICIPACION.md` coincide con GitHub.
- [ ] Dataset reproducible y no versionado.
- [ ] Checkpoint descargable.
- [ ] Demo funcionando sin internet.
- [ ] Informe PDF revisado por 2 integrantes.
- [ ] Cada integrante resolvió las 24 preguntas de `docs/DEFENSA.md`.
- [ ] Dos ensayos con presentador aleatorio.
- [ ] Nombre y código de Isabela completados en README y `docs/EQUIPO.md`.

---

## 14. Riesgos y plan B

| Riesgo | Señal | Acción |
|---|---|---|
| CI lento | > 10 min | Cachear pip, bajar smoke a 1 paso y 32×32. |
| CI inestable | falla a veces | Semilla fija, sin descargas, sin GPU. |
| PR gigante | > 400 líneas | Pedir división antes de revisar. |
| Conflicto en `requirements.txt` | rebase falla | Conservar ambas, ordenar, probar instalación. |
| Nadie revisa | PRs estancados | Asignar revisores en la reunión del lunes y recordar por issue. |
| Informe sin cifras | placeholders | Bloquear el release hasta que Juan Sebastian entregue `metrics.csv`. |
| Repo privado sin protección | no aparece la opción | Hacerlo público o aplicar la regla de equipo documentada. |
| Alguien se atrasa | sin commits al día 5 | Activar respaldo, reasignar issue, dejar evidencia en el PR. |
