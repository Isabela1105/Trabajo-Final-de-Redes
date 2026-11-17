# Participación del equipo

Actualizar al final de cada semana. Al entregar, esta tabla debe coincidir con GitHub Insights y
`git shortlog -sne --all`.

## Tabla general

| Integrante | GitHub | Módulos que posee | PRs autorados | PRs revisados | Commits | Informe | Defensa |
|---|---|---|---|---|---|---|---|
| Isabella Ortiz Hernandez | @Isabellaortizher-collab | Datos, dataset, scripts | — | — | — | Dataset | Datos |
| Juan Pablo Dávila Moncayo | @Juanpablodavila | Generador U-Net | — | — | — | Arquitectura | Generador |
| Jeremy Santiago Caicedo Cabezas | @JeremyCaicedo | Discriminador, pérdidas | — | — | — | Marco teórico | Pérdidas |
| Cristian Fabian Muñoz Piedrahita | @CristianMP355 | Entrenamiento, config | — | — | — | Entrenamiento | Train loop |
| Juan Sebastian Castillo Acevedo | @Juansec01 | Evaluación, app | — | — | — | Resultados | Demo |
| Isabela _(completar nombre)_ | @Isabela1105 | CI/CD, informe | — | — | — | Intro/conclusiones | Cierre |

## Detalle por integrante

### Isabella Ortiz Hernandez (@Isabellaortizher-collab)

- **Módulos:** `src/gan/data/`, `scripts/build_dataset.py`, `scripts/check_dataset.py`.
- **PRs:** #..., #...
- **Commits representativos:**
  - `feat(data): ...` (#...)
  - `test(data): ...` (#...)
- **Aporte al informe:** sección Dataset, ética y licencias.
- **Aporte a la defensa:** recolección, degradaciones, splits.

### Juan Pablo D�vila Moncayo (@Juanpablodavila)

- **Módulos:** `src/gan/models/blocks.py`, `src/gan/models/generator.py`.
- **PRs:** #...
- **Commits representativos:** ...
- **Aporte al informe:** arquitectura U-Net, tabla de capas.
- **Aporte a la defensa:** skip connections, `ngf`, `tanh`.

### Jeremy Santiago Caicedo Cabezas (@JeremyCaicedo)

- **Módulos:** `src/gan/models/discriminator.py`, `src/gan/losses.py`.
- **PRs:** #...
- **Commits representativos:** ...
- **Aporte al informe:** PatchGAN, ecuaciones de pérdida.
- **Aporte a la defensa:** λ = 100, BCE vs MSE, estabilidad.

### Cristian Fabian Mu�oz Piedrahita (@CristianMP355)

- **Módulos:** `src/gan/train.py`, `src/gan/config.py`, `src/gan/utils/viz.py`, `configs/`,
  `tests/test_smoke_train.py`, `tests/test_train_integration.py`, sección 5 de
  `docs/ARQUITECTURA.md`, bloque de entrenamiento de `docs/DEFENSA.md` y §6–§7 de
  `docs/INFORME.md`.
- **PRs (ramas locales, pendientes de push):** `feat/14-train-loop` (PR-D2),
  `test/15-smoke-train` (PR-D3), `feat/23-train-integration` (PR-D4),
  `feat/29-train-experiments` (PR-D5), `fix/34-colab-checkpoint` (PR-D6).
- **Commits representativos:** bucle D→G y checkpoints (PR-D2), integración de builders y
  `make_dataset` reales (PR-D4), métricas y ablación (PR-D5), blindaje de `--resume` (PR-D6).
- **Aporte al informe:** hiperparámetros congelados en `configs/base.yaml`, citas del bloque de
  entrenamiento (§6) y artefactos para el build del PDF (§7).
- **Aporte a la defensa:** `train_step` con dos Adams y mixed precision, early stopping por L1
  de validación, `--resume` a prueba de cortes en Colab y ablación sin GAN.

### Juan Sebastian Castillo Acevedo (@Juansec01)

- **Módulos:** `src/gan/evaluate.py`, `src/gan/infer.py`, `app/app.py`.
- **PRs:** #...
- **Commits representativos:** ...
- **Aporte al informe:** resultados, demo, limitaciones.
- **Aporte a la defensa:** PSNR/SSIM, baselines, app.

### Isabela (@Isabela1105)

- **Módulos:** `.github/`, `README.md`, `CONTRIBUTING.md`, `report/`.
- **PRs:** #...
- **Commits representativos:** ...
- **Aporte al informe:** introducción, conclusiones, anexos.
- **Aporte a la defensa:** CI/CD, reproducibilidad, cierre.

## Verificación

```bash
git shortlog -sne --all
git log --author="Nombre" --oneline | wc -l
```

En GitHub:

- **Insights → Contributors**
- **Pull requests → Author / Reviewer**
- **Issues → Assignee**

## Declaración

Cada integrante confirma que:

- [ ] Entiende y puede explicar el código de su módulo.
- [ ] Puede ejecutar el pipeline completo (dataset, smoke train, inferencia).
- [ ] Participó en al menos una revisión de otro módulo.
- [ ] Escribió o revisó una sección del informe.
- [ ] Participó en al menos un ensayo de defensa.
