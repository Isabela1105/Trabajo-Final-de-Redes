# Dataset — Restauración de fotografías

> Completar esta ficha antes de la entrega. Es la fuente de verdad para el informe y la defensa.

## Resumen

| Campo | Valor |
|---|---|
| Nombre | Fotos UAO — Restauración |
| Versión | 1.0 |
| Fecha | 2026-11-XX |
| Integrantes | Isabella Ortiz, Juan Pablo D�vila, Jeremy Caicedo, Cristian Mu�oz, Juan Sebastian Castillo, Isabela |
| Fuentes | — |
| Pares | — |
| Resolución de entrenamiento | 256×256 |
| Resolución original mínima | 512×512 |
| Splits | train 80 % / val 10 % / test 10 % (por fuente) |
| Semilla base | 42 |

## Procedencia

- **Fotos propias:** describir dónde y cuándo se tomaron, quién las tomó.
- **Fotos de dominio público:** listar enlaces, autor y licencia en `data/sources.csv`.
- **Fotos antiguas escaneadas:** indicar si hay consentimiento y cómo se documentó.

## Degradaciones

| Operación | Probabilidad | Parámetros |
|---|---|---|
| Blur gaussiano | 0.7 | kernel 3–7, sigma 0.5–2.0 |
| Ruido gaussiano | 0.6 | sigma 5–25 |
| JPEG | 0.5 | calidad 20–60 |
| Baja resolución | 0.4 | factor 2–4 |
| Desvanecido | 0.5 | sepia 0.2–0.6 |
| Rayones | 0.4 | 1–5 líneas |
| Polvo | 0.4 | 10–50 círculos |
| Contraste | 0.3 | 0.7–1.3 |

Script: `scripts/build_dataset.py`
Manifiesto: `data/manifest.csv`

## Estadísticas

- Fuentes por split: train —, val —, test —.
- Pares por split: train —, val —, test —.
- Tamaño original: media —, mín —, máx —.
- Distribución de degradaciones: —.
- Ejemplos: `docs/img/dataset_grid.png`.

## Limitaciones

- Sesgo geográfico/cultural: —
- Pocas imágenes de: —
- Las degradaciones sintéticas no cubren todos los daños reales.
- El modelo puede alucinar detalles; no sirve para uso forense.

## Licencia y uso

- Código: MIT.
- Dataset: CC BY-NC 4.0 (o la acordada), salvo fotos de terceros.
- No se publican fotos personales sin consentimiento.
- Uso académico. No usar para vigilancia ni identificación de personas.

## Reproducción

```bash
python scripts/build_dataset.py --raw data/raw --out data/processed --variants 3 --seed 42
python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv
```
