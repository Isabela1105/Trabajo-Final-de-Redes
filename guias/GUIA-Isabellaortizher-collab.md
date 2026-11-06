# Guía individual — Isabella Ortiz Hernandez

**GitHub:** `@Isabellaortizher-collab`
**Código:** 2226215
**Rol:** Datos y dataset
**Respaldo:** Juan Pablo Dávila (`@Juanpablodavila`)
**Repositorio:** https://github.com/Isabela1105/Trabajo-Final-de-Redes

> Antes de empezar, lee: `README.md`, `docs/EQUIPO.md`, `docs/PLAN_COMMITS.md`,
> `CONTRIBUTING.md` y `docs/DATASET.md`.

---

## 1. Tu misión

Construir el **dataset personalizado** de restauración de fotografías:

- Recolectar y documentar las fotos.
- Generar pares `(input degradado, target limpio)` de forma reproducible.
- Dividir en `train/val/test` **por foto fuente** (sin fuga de datos).
- Validar estructura, tamaños y manifiesto.
- Documentar ética, licencias y limitaciones.

Tu código es la base de todo el proyecto: sin dataset no pueden entrenar Cristian ni evaluar Juan
Sebastian. Por eso tus PRs de la semana 1 son de los primeros en mergearse.

---

## 2. Archivos que posees

```text
src/gan/data/__init__.py
src/gan/data/dataset.py
src/gan/data/degradations.py
scripts/build_dataset.py
scripts/check_dataset.py
tests/test_dataset.py
data/README.md
docs/DATASET.md
docs/img/dataset_grid.png
```

No edites archivos de modelos, train, app ni CI sin avisar a su dueño.

---

## 3. Tus ramas y PRs

| Semana | Día | Rama | PR | Commits | Revisor |
|---|---|---|---|---|---|
| 1 | 1 | `docs/03-protocolo-dataset` | PR-A1 | `docs(data): definir protocolo de recolección y criterios de calidad` | Isabela |
| 1 | 2–4 | `feat/07-dataset-pairs` | PR-A2 | `feat(data): implementar emparejamiento y split por fuente`<br>`feat(data): agregar degradaciones sintéticas configurables`<br>`test(data): validar formas, rangos y no fuga entre splits` | Juan Pablo, Cristian |
| 1 | 5 | `feat/09-build-dataset` | PR-A3 | `feat(data): crear script de construcción con manifiesto y validación` | Isabela |
| 2 | 8–11 | `feat/14-dataset-v2` | PR-A4 | `feat(data): generar tres variantes por fuente y estadísticas`<br>`fix(data): corregir orden de degradaciones y semilla`<br>`docs(data): completar ficha del dataset y galería`<br>`test(data): agregar prueba de reproducibilidad del manifiesto` | Juan Pablo, Jeremy |
| 3 | 15–18 | `feat/21-real-eval` | PR-A5 | `feat(data): agregar conjunto real para evaluación cualitativa`<br>`docs(data): documentar licencias, sesgos y limitaciones`<br>`fix(data): manejar imágenes con alfa y EXIF` | Juan Sebastian, Isabela |
| 4 | 22–24 | `docs/28-dataset-final` | PR-A6 | `docs(data): actualizar cifras finales del dataset`<br>`test(data): cubrir casos borde y cobertura final` | Cristian |

**Total:** 6 PRs, 14 commits. Objetivo mínimo: 12 commits significativos.

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
git switch -c docs/03-protocolo-dataset
```

Crea `docs/DATASET.md` y `data/README.md` con el protocolo. Commit:

```bash
git add docs/DATASET.md data/README.md
git commit -m "docs(data): definir protocolo de recolección y criterios de calidad (#3)"
git push -u origin HEAD
```

Abre el PR con la plantilla, asigna a `@Isabela1105` y espera revisión.

---

## 5. Semana por semana

### Semana 1 — Dataset v1

**Objetivo:** poder generar pares y splits desde `data/raw/`.

1. Implementa `load_image`, `load_pair`, `build_dataset` en `dataset.py`.
2. Implementa las degradaciones en `degradations.py` con semilla.
3. Implementa `build_dataset.py`:
   - Lee `data/raw/`.
   - Genera 1–3 variantes por foto.
   - Divide por fuente: 80/10/10.
   - Guarda `data/processed/{train,val,test}/{input,target}/`.
   - Escribe `data/manifest.csv`.
4. Implementa `check_dataset.py`:
   - Verifica pares completos.
   - Verifica que ninguna fuente esté en dos splits.
   - Verifica RGB y tamaño.
   - Falla con código de salida distinto de 0 si algo está mal.
5. Tests:
   - Formas y rango `[-1,1]`.
   - Reproducibilidad con la misma semilla.
   - No fuga entre splits.

**Criterio de aceptación:**

```bash
python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42
python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv
pytest tests/test_dataset.py -q
```

### Semana 2 — Dataset v2

- 3 variantes por fuente.
- Estadísticas: pares por split, distribución de degradaciones.
- `docs/img/dataset_grid.png` con 8–16 ejemplos `input | target`.
- Ficha del dataset completa.
- Test de reproducibilidad del manifiesto.

### Semana 3 — Conjunto real

- Agrega fotos reales antiguas/degradadas para evaluación **cualitativa** (sin ground truth).
- Documenta licencias, consentimiento y sesgos.
- Corrige manejo de alpha y EXIF (fotos de celular rotadas).

### Semana 4 — Cierre

- Cifras finales en `data/README.md`.
- Tests de casos borde (imagen 1×1, grayscale, corrupta).
- Revisa que `data/raw` y `data/processed` no estén en git.

---

## 6. Cómo debe quedar el dataset

```text
data/
├── raw/                     # fotos originales (ignorado por git)
├── processed/
│   ├── train/
│   │   ├── input/
│   │   └── target/
│   ├── val/
│   └── test/
├── sample/                  # 8 imágenes sintéticas para CI
├── manifest.csv
└── README.md
```

**Regla crítica:** una misma foto fuente y todas sus variantes van al mismo split. Si una variante
va a train y otra a test, hay fuga de datos y las métricas no valen.

---

## 7. Revisiones que debes hacer

| PR | Dueño | Qué revisar |
|---|---|---|
| PR-B2 | Juan Pablo | Que el generador reciba `(B,256,256,3)` y devuelva la misma forma; que los tests de forma tengan sentido. |
| PR-D4 | Cristian | Que el dataset se consuma correctamente; que el batch tenga `(input, target)` y no esté desalineado. |
| PR-E5 | Juan Sebastian | Que la evaluación use el test set y no el de train; que las figuras comparan `input/target/predicción`. |
| PR-F6 | Isabela | Que la tabla de participación tenga tus PRs y commits reales. |

Usa el checklist de `docs/MERGE_Y_REVISION.md`. Si no entiendes algo, pide explicación antes de
aprobar.

---

## 8. Qué debes poder explicar en la defensa

1. ¿Por qué el dataset es personalizado?
2. ¿Cómo se generan los pares y por qué hay *ground truth*?
3. ¿Qué degradaciones se aplican y con qué probabilidad?
4. ¿Cómo se evita la fuga de datos?
5. ¿Qué contiene el manifiesto y para qué sirve la semilla?
6. ¿Cuántas fuentes/pares hay y cómo se distribuyen?
7. ¿Qué limitaciones éticas y de sesgo tiene el dataset?
8. ¿Cómo se ejecuta `build_dataset.py` y `check_dataset.py`?

---

## 9. Entregables

- [ ] `src/gan/data/dataset.py` comentado y con type hints.
- [ ] `src/gan/data/degradations.py` reproducible.
- [ ] `scripts/build_dataset.py` y `check_dataset.py`.
- [ ] `tests/test_dataset.py` verde.
- [ ] `data/README.md` completo.
- [ ] `docs/DATASET.md` completo.
- [ ] `docs/img/dataset_grid.png`.
- [ ] 14 commits y 6 PRs.
- [ ] 4 revisiones a otros.
- [ ] Sección **Dataset** del informe.
- [ ] Diapositivas de datos para la defensa.

---

## 10. Checklist final

- [ ] `pytest tests/test_dataset.py -q` pasa.
- [ ] `check_dataset.py` falla si hay fuga (probarlo a propósito).
- [ ] `data/raw` y `data/processed` no están en git.
- [ ] No hay fotos personales sin consentimiento.
- [ ] `data/README.md` tiene licencia y procedencia.
- [ ] Tu nombre aparece en `docs/PARTICIPACION.md`.
- [ ] Puedes explicar el pipeline completo sin mirar el código.
