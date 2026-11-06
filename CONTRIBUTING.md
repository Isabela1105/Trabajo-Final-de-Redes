# Guía de contribución

Este repositorio lo mantenemos entre 6 personas. Para que el trabajo sea visible, revisable y
defendible, todo cambio entra por Pull Request.

## 1. Requisitos

- Python **3.11**.
- Git.
- Cuenta de GitHub con acceso de escritura al repo.
- Para entrenar: Colab o Kaggle (GPU). Para desarrollar: CPU es suficiente.

## 2. Instalación

```bash
git clone <URL>
cd <repo>
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```

Probar que todo funciona:

```bash
pytest -q
python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
```

## 3. Flujo diario

```bash
# 1. Actualizar main
git switch main
git pull --rebase origin main

# 2. Crear rama (incluye el número del issue)
git switch -c feat/10-generator-unet

# 3. Trabajar en commits pequeños
git add -p
git commit -m "feat(model): implementar generador U-Net configurable (#10)"

# 4. Antes de push
pre-commit run --all-files
pytest -q

# 5. Push y PR
git push -u origin HEAD
gh pr create --fill --reviewer JeremyCaicedo
```

Si `main` avanzó:

```bash
git fetch origin
git rebase origin/main
# resolver conflictos
git add <archivos>
git rebase --continue
git push --force-with-lease
```

## 4. Ramas

Formato: `tipo/numero-issue-descripcion`.

| Tipo | Ejemplo |
|---|---|
| `feat` | `feat/10-generator-unet` |
| `fix` | `fix/32-loss-nan` |
| `docs` | `docs/03-protocolo-dataset` |
| `test` | `test/15-smoke-train` |
| `refactor` | `refactor/17-generator-norm` |
| `ci` | `ci/02-workflow` |
| `perf` | `perf/25-gen-memory` |
| `chore` | `chore/06-config` |

Una rama = un PR. No reutilizar ramas viejas.

## 5. Commits

Formato **Conventional Commits**:

```text
tipo(area): descripción en imperativo (#issue)
```

Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`.

Áreas: `data`, `model`, `loss`, `train`, `eval`, `app`, `ci`, `docs`, `config`.

Buenos:

```text
feat(data): agregar degradaciones sintéticas configurables (#7)
fix(model): corregir concatenación de skip connections (#17)
test(loss): verificar valores con casos conocidos (#20)
docs(train): documentar hiperparámetros y reanudación (#25)
```

Malos:

```text
cambios
update
arreglos varios
asdf
```

Reglas:

- Un commit = un cambio coherente.
- No mezclar refactor + feature + docs.
- El mensaje explica **qué** y **por qué**.
- Si hay pair programming: `Co-authored-by: Nombre <correo>`.

## 6. Pull Requests

- Título con el mismo formato del commit.
- Descripción con la plantilla.
- `Closes #<issue>`.
- Máximo ~400 líneas cambiadas. Si es más, dividir.
- Al menos 1 aprobación (2 en rutas críticas).
- CI verde.
- Squash merge; la rama se borra.

### Revisión

- Revisar en menos de 24 h hábiles.
- Si no se entiende el código, pedir explicación antes de aprobar.
- Usar `suggestion` para cambios pequeños.
- Aprobar solo si se podría defender en la sustentación.

## 7. Estilo de código

- Python 3.11, PEP 8, `black` (línea 100), `ruff`.
- Identificadores en **inglés**; comentarios y docstrings en **español**.
- Type hints en funciones públicas.
- Funciones pequeñas y con una responsabilidad.
- Evitar notebooks para lógica reutilizable: pasar a `src/gan/`.
- Semillas fijas para reproducibilidad.

### Comentarios

Cada archivo empieza con:

```python
"""
Propósito: ...
Autor: Juan Pablo D�vila (@Juanpablodavila)
Entradas: ...
Salidas: ...
Referencia: ...
"""
```

Cada función pública:

```python
def build_generator(image_size: int = 256, ngf: int = 64, n_downsample: int = 8) -> tf.keras.Model:
    """Construye el generador U-Net.

    Args:
        image_size: tamaño de la imagen cuadrada de entrada/salida.
        ngf: número de filtros de la primera capa.
        n_downsample: número de bloques encoder/decoder.

    Returns:
        Modelo Keras que mapea (B, image_size, image_size, 3) -> misma forma, en [-1, 1].
    """
```

Comentar el **por qué**, no lo obvio:

```python
# InstanceNorm en vez de BatchNorm: con batch=1 BatchNorm es inestable.
x = layers.InstanceNormalization()(x)
```

## 8. Tests

- Todo módulo nuevo lleva tests.
- Tests rápidos (< 5 s) para formas, rangos y pérdidas.
- Un smoke test de 2 pasos para el pipeline completo.
- Datos sintéticos para no depender del dataset real en CI.
- `pytest -q` debe pasar antes de pedir revisión.

## 9. Datos y archivos grandes

No commitear:

- `data/raw/`, `data/processed/`, `outputs/`.
- Pesos (`*.keras`, `*.h5`, `*.pt`, `*.pth`, `*.ckpt`).
- `.env`, tokens, credenciales.
- Archivos > 5 MB (el hook lo bloquea).

Compartir por GitHub Release, Hugging Face Hub o Drive, y documentar el enlace.

## 10. Participación

- Primer commit en los primeros 2 días.
- Mínimo 12 commits significativos y 4 revisiones por persona.
- Cada semana: 1 PR propio + 1 revisión + 1 aporte al informe/docs.
- Actualizar `docs/PARTICIPACION.md` con enlaces.

## 11. Si algo se bloquea

1. Comentar en el issue qué falta y quién puede ayudar.
2. Avanzar con mocks o datos sintéticos.
3. Pedir pairing a un compañero.
4. Si el bloqueo es de otro módulo, tomar una tarea de `docs/` o `tests/` mientras se resuelve.

## 12. Checklist antes de pedir revisión

- [ ] `pre-commit run --all-files`
- [ ] `pytest -q`
- [ ] Smoke training (si toca train/model/data)
- [ ] Docstrings/comentarios
- [ ] Tests nuevos
- [ ] Documentación actualizada
- [ ] Sin archivos grandes ni secretos
- [ ] PR enlazado al issue
- [ ] Revisor asignado
