# CI/CD y protección de la rama `main`

## 1. Qué se automatiza y qué no

| Etapa | ¿Automático? | Dónde | Por qué |
|---|---|---|---|
| Lint y formato | Sí | GitHub Actions | Evita ruido y discusiones de estilo. |
| Tests unitarios | Sí | GitHub Actions | Detecta rupturas de contratos (formas, pérdidas, dataset). |
| Smoke training (2 pasos) | Sí | GitHub Actions (CPU) | Verifica que el pipeline completo no está roto. |
| Entrenamiento real | No | Colab / Kaggle | Requiere GPU y horas; no cabe en CI. |
| Evaluación completa | No | Colab / local | Depende de checkpoints y dataset. |
| Build del informe PDF | Sí, en tag | GitHub Actions | Entrega reproducible. |
| Release y demo | Sí, en tag | GitHub Actions + HF Spaces | Publica la entrega. |

**Mensaje para el docente:** el CI garantiza calidad y reproducibilidad; el entrenamiento pesado
se ejecuta en un entorno con GPU y se documenta con checkpoints, configuración y métricas.

---

## 2. Workflow de CI

Archivo: `.github/workflows/ci.yml` (plantilla lista en `plantillas/.github/workflows/ci.yml`).

### Etapas

```text
checkout
  └─ setup-python 3.11 + cache pip
       └─ pip install -r requirements-dev.txt
            ├─ ruff check .
            ├─ black --check .
            ├─ pytest -q --cov=src
            ├─ python scripts/make_synthetic_sample.py
            └─ python -m src.gan.train --config configs/smoke.yaml --max-steps 2
                 └─ upload artifact outputs/smoke
```

Además, un job independiente valida el título del PR con
`amannn/action-semantic-pull-request`.

### Por qué un smoke test y no un test de 1 época

- 2 pasos verifican: carga de datos, forward del generador, forward del discriminador, cálculo de
  pérdidas, gradientes, optimizadores y guardado de checkpoint.
- Con 64×64 y `ngf=8` tarda segundos.
- Un test de 1 época completa sería lento y frágil (depende de datos y hardware).

### Tiempo objetivo

| Etapa | Tiempo |
|---|---|
| Instalación de dependencias (con cache) | 1–2 min |
| Lint + formato | < 30 s |
| Tests | 1–2 min |
| Smoke training | 1–2 min |
| **Total** | **4–7 min** |

Si supera 10 min, revisar: versión de TensorFlow, tamaño del smoke, cache de pip.

---

## 3. Protección de `main`

### 3.1 Crear la regla

1. GitHub → repositorio → **Settings**.
2. **Branches** → **Add branch protection rule**.
3. Branch name pattern: `main`.
4. Activar:

- [x] **Require a pull request before merging**
  - [x] Require approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [ ] Require review from Code Owners (opcional; ver nota)
- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Checks: `Calidad y pruebas`, `Título de PR`
- [x] **Require conversation resolution before merging**
- [x] **Require linear history**
- [x] **Do not allow bypassing the above settings**
- [x] **Automatically delete head branches**

### 3.2 Nota sobre CODEOWNERS

`CODEOWNERS` sirve para **enrutar** revisiones (por ejemplo, que B revise cambios del generador).
Si se activa "Require review from Code Owners", un PR puede quedar bloqueado si el dueño no está
disponible. Recomendación:

- Semanas 1–2: CODEOWNERS solo como sugerencia.
- Semanas 3–4: activar "Require review from Code Owners" para rutas críticas
  (`src/gan/data/`, `src/gan/losses.py`, `src/gan/train.py`).

### 3.3 Si el repo es privado y no aparece la opción

GitHub Free puede limitar las reglas de protección en repos privados. Opciones:

1. Hacer el repo **público** (recomendado: más evidencia y Actions ilimitadas).
2. Usar **GitHub Education** (si la universidad lo permite) o un plan Team.
3. Si no se puede, usar una regla de equipo: **nadie mergea sin PR y sin CI verde**, aunque
   GitHub no lo imponga técnicamente. Documentarlo en `CONTRIBUTING.md`.

---

## 4. CD: release al crear un tag

Archivo: `.github/workflows/release.yml`.

Se dispara con tags `v*`:

```bash
git switch main
git pull --rebase
git tag -a v1.0.0 -m "Entrega final: Pix2Pix para restauración de fotografías"
git push origin v1.0.0
```

El workflow:

1. Instala Python y Pandoc.
2. Construye `report/informe.pdf` (si existe `report/Makefile`).
3. Crea un GitHub Release con notas.
4. Adjunta el PDF.
5. Opcional: descarga el generador desde una URL y lo adjunta si pesa menos de 2 GB.

**No** se despliega la app automáticamente en cada merge. El despliegue a Hugging Face Spaces es
manual o con aprobación, para evitar publicar modelos rotos.

---

## 5. Pre-commit local (fail fast)

Archivo: `.pre-commit-config.yaml`.

```bash
pre-commit install
pre-commit run --all-files
```

Hooks:

- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (> 5 MB).
- `detect-private-key`.
- `ruff` con `--fix`.
- `black`.
- `nbstripout` (quita outputs de notebooks antes de commitear).

Ventaja: el 80 % de los errores de CI se detectan antes de hacer push.

---

## 6. Reglas de merge

| Regla | Detalle |
|---|---|
| Estrategia | **Squash and merge** |
| Título del squash | Igual al título del PR (Conventional Commits) |
| Rama | Se borra automáticamente |
| Aprobaciones | 1 (2 en rutas críticas) |
| CI | Obligatorio verde |
| Historial | Lineal, sin merge commits |
| `main` | Siempre ejecutable |

### Ejemplo de flujo

```bash
git switch main
git pull --rebase
git switch -c feat/10-generator-unet
# trabajar en commits pequeños
git add -p
git commit -m "feat(model): implementar generador U-Net configurable (#10)"
git push -u origin HEAD
gh pr create --fill --reviewer usuario-c
# esperar CI + revisión
# merge con squash desde la UI o:
gh pr merge --squash --delete-branch
```

---

## 7. Etiquetas y tablero

### Labels sugeridas

`area:data`, `area:model`, `area:train`, `area:eval`, `area:ci`, `area:docs`,
`priority:high`, `priority:low`, `blocked`, `good first issue`, `bug`, `enhancement`.

### Milestones

- `M1 — Fundaciones` (semana 1)
- `M2 — Primer entrenamiento` (semana 2)
- `M3 — Experimentos` (semana 3)
- `M4 — Entrega` (semana 4)

### Tablero

Columnas: `Backlog`, `Esta semana`, `En progreso`, `En revisión`, `Hecho`.

Cada issue debe tener: objetivo, criterios de aceptación, archivos, dependencia y responsable.

---

## 8. Seguridad y datos

- **Nunca** commitear `.env`, tokens, contraseñas ni claves de Hugging Face/W&B.
- Usar **GitHub Secrets** para credenciales.
- No subir fotos personales sin consentimiento.
- No subir pesos ni datasets grandes; usar Release, Hugging Face Hub o Drive.
- Activar **Dependabot alerts** y **secret scanning** si está disponible.
- Revisar que `data/raw/`, `data/processed/` y `outputs/` estén en `.gitignore`.

---

## 9. Problemas comunes

| Problema | Causa | Solución |
|---|---|---|
| CI falla en `black --check` | Formato distinto | `black .` y commit. |
| CI falla en `ruff` | Import sin usar, variable muerta | `ruff check . --fix`. |
| CI falla en smoke | Contrato roto (forma, pérdida, config) | Reproducir local con `configs/smoke.yaml`. |
| CI tarda mucho | Sin cache o TF completo | Revisar `cache: pip`, usar `tensorflow-cpu` en Linux. |
| No aparece el check en branch protection | El workflow no ha corrido en `main` | Hacer un PR de prueba y luego seleccionar el check. |
| `git push` rechazado en `main` | Protección activa | Crear rama y PR. |
| Conflicto en `requirements` | Dos PRs tocaron el archivo | Rebase y resolver; avisar en el PR. |
| El release no adjunta el PDF | No existe `report/informe.pdf` | Correr `make report` localmente y verificar el path. |

---

## 10. Checklist de configuración inicial

- [ ] Repo creado y compañeros invitados con rol `Write`.
- [ ] `main` protegida con PR + checks.
- [ ] Actions habilitado.
- [ ] Secrets configurados (si se usan).
- [ ] `CODEOWNERS` con usuarios reales.
- [ ] Plantillas de issue/PR en `.github/`.
- [ ] `pre-commit install` en cada máquina.
- [ ] Primer PR de prueba mergeado con CI verde.
- [ ] Tag `v0.1.0` de prueba (opcional) para validar el release.
