# Estadísticas del dataset

Generado con `python scripts/dataset_report.py` a partir de `data/manifest.csv`.

## Fuentes y pares por split

| split | fuentes | pares |
|---|---|---|
| train | 122 | 366 |
| val | 15 | 45 |
| test | 15 | 45 |
| total | 152 | 456 |

Variantes por fuente: 3. Semillas de par: de 42000000 a 42151002.

## Degradaciones aplicadas (número de pares)

| degradacion | train | val | test | total | % de pares | % configurado |
|---|---|---|---|---|---|---|
| downscale | 149 | 20 | 15 | 184 | 40.4 | 40 |
| blur | 266 | 31 | 25 | 322 | 70.6 | 70 |
| noise | 216 | 27 | 20 | 263 | 57.7 | 60 |
| fade | 173 | 20 | 25 | 218 | 47.8 | 50 |
| contrast | 105 | 21 | 19 | 145 | 31.8 | 30 |
| scratches | 139 | 22 | 6 | 167 | 36.6 | 40 |
| dust | 151 | 20 | 19 | 190 | 41.7 | 40 |
| jpeg | 178 | 25 | 26 | 229 | 50.2 | 50 |

Cada par recibe un subconjunto aleatorio de degradaciones; por eso los porcentajes
reales se parecen a los configurados pero no coinciden exactamente.

## Fuentes por origen y split

| origen | train | val | test | total |
|---|---|---|---|---|
| DIV2K | 79 | 13 | 8 | 100 |
| Pexels | 25 | 1 | 4 | 30 |
| Propias | 18 | 1 | 3 | 22 |

## Tamaño original de las fuentes

| medida | mínimo | media | máximo |
|---|---|---|---|
| ancho (px) | 721 | 2392 | 7953 |
| alto (px) | 719 | 2277 | 9941 |
