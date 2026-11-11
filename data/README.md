# Dataset — Restauración de fotografías

> Esta ficha es la fuente de verdad para el informe y la defensa. Las cifras salen de
> `data/manifest.csv` y de `docs/dataset_stats.md`; un test (`tests/test_ficha.py`) falla si
> dejan de coincidir.

## Resumen

| Campo | Valor |
|---|---|
| Nombre | Fotos UAO — Restauración |
| Versión | 1.0 (v1; las cifras finales se congelan en el PR de cierre) |
| Fecha | 2026-11-11 |
| Integrantes | Isabella Ortiz, Juan Pablo Dávila, Jeremy Caicedo, Cristian Muñoz, Juan Sebastian Castillo, Isabela |
| Fuentes | 152 |
| Variantes por fuente | 3 |
| Pares | 456 |
| Pares por split | train 366 / val 45 / test 45 |
| Resolución de entrenamiento | 256×256 |
| Resolución original mínima | ancho 721 px, alto 719 px (todas las fuentes miden al menos 512 px por lado) |
| Proporción del split | 80 % / 10 % / 10 % (por foto fuente) |
| Semilla base | 42 |

## Procedencia

| Origen | Fuentes | Licencia | Detalle |
|---|---|---|---|
| DIV2K (validación HR) | 100 | Solo uso académico | Colección con imágenes de internet; los derechos son de sus dueños. Cita: Agustsson y Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, CVPRW 2017. |
| Pexels | 30 | Pexels License | Autor y enlace de cada foto en `data/sources.csv`. |
| Propias | 22 | Propia | Tomadas por Isabella Ortiz con su celular. Algunas se guardaron desde WhatsApp, que las recomprime. |

El autor y la licencia de **cada** foto fuente están en `data/sources.csv`.

Nota: las fotos fuente son de terceros y propias; lo que es original del proyecto es el
pipeline de degradación y los pares generados con él.

## Estructura

```text
data/
├── raw/          # fotos fuente (no se suben a git)
├── processed/    # pares generados (no se suben a git; se comparten por Drive)
│   └── {train,val,test}/{input,target}/<fuente>_v<N>.png
├── sources.csv   # autor, licencia y enlace de cada fuente
└── manifest.csv  # un par por fila: split, degradaciones y semilla
```

En cada par, `input` es la foto degradada y `target` es la foto limpia. Los dos archivos
tienen el mismo nombre. Cada foto fuente y sus 3 variantes caen en un solo split.

## Degradaciones

Se aplican en este orden. Cada par recibe un subconjunto aleatorio (al menos una).

| Orden | Operación | Probabilidad | Parámetros |
|---|---|---|---|
| 1 | Baja resolución | 0.4 | factor 2–4 (reduce y vuelve a ampliar) |
| 2 | Desenfoque gaussiano | 0.7 | sigma (radio) 0.5–2.0 |
| 3 | Ruido gaussiano | 0.6 | sigma 5–25 |
| 4 | Desvanecido | 0.5 | mezcla con sepia 0.2–0.6 y saturación 0.3–0.8 |
| 5 | Contraste y brillo | 0.3 | factores 0.7–1.3 |
| 6 | Rayones | 0.4 | 1–5 líneas blancas o negras, grosor 1–3 |
| 7 | Polvo | 0.4 | 10–50 círculos de radio 1–2 |
| 8 | Compresión JPEG | 0.5 | calidad 20–60 |

Cada par tiene su propia semilla: `semilla_base × 1 000 000 + índice_de_fuente × 1 000 + variante`.
Con la misma semilla base el dataset y el manifiesto salen idénticos.

## Cómo se reproduce

```bash
python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42
python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv
python scripts/dataset_report.py   # estadísticas y galería de ejemplos
```

## Estadísticas

Tablas completas en `docs/dataset_stats.md`. Ejemplos visuales: `docs/img/dataset_grid.png`.

| Origen | train | val | test | total |
|---|---|---|---|---|
| DIV2K | 79 | 13 | 8 | 100 |
| Pexels | 25 | 1 | 4 | 30 |
| Propias | 18 | 1 | 3 | 22 |

| Degradación | Pares con ese daño | % real | % configurado |
|---|---|---|---|
| Baja resolución | 184 | 40.4 | 40 |
| Desenfoque | 322 | 70.6 | 70 |
| Ruido | 263 | 57.7 | 60 |
| Desvanecido | 218 | 47.8 | 50 |
| Contraste | 145 | 31.8 | 30 |
| Rayones | 167 | 36.6 | 40 |
| Polvo | 190 | 41.7 | 40 |
| JPEG | 229 | 50.2 | 50 |

Tamaño original de las fuentes: ancho de 721 a 7953 px (media 2392), alto de 719 a 9941 px
(media 2277).

## Limitaciones

- **Predominan las fotos de DIV2K** (66 % de las fuentes): estilo y resolución muy parecidos
  entre sí, y distintos de una foto de celular.
- **Las fotos de WhatsApp vienen recomprimidas**, así que algunos `target` no son 100 %
  prístinos.
- **Pocos rayones en `test`**: solo 6 pares tienen ese daño frente a 19–26 en los otros. Las
  métricas por tipo de daño para rayones son poco confiables.
- **Las degradaciones son sintéticas** y no cubren todos los daños reales (manchas, pliegues,
  hongos, desgarros).
- **Posibles sesgos de contenido** (escenas nocturnas, interiores, personas): por revisar con
  una muestra manual antes del informe final.
- El modelo puede inventar detalles que no estaban en la foto; no sirve para uso forense.

## Licencia y uso

- Código: MIT (ver `LICENSE`).
- Las fotos fuente de DIV2K y Pexels son de terceros: **no se publican ni se redistribuyen**.
  Los pares se comparten solo dentro del equipo, para uso académico del curso.
- No se publican fotos personales sin consentimiento de las personas que aparecen en ellas.