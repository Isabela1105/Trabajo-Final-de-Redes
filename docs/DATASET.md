# Dataset personalizado — restauración de fotografías

## 1. Objetivo

Construir un dataset **propio** de pares `(foto degradada, foto limpia)` para entrenar Pix2Pix.
La foto limpia es el *ground truth*; la degradada es la entrada. Así se puede medir qué tan
bien restaura el modelo (PSNR, SSIM, L1) y no solo mirar resultados "bonitos".

> **Requisito del profesor:** el dataset debe ser personalizado. No se permite descargar el
> dataset del tutorial de caballos/cebras ni el de manzanas/naranjas.

---

## 2. Recolección

### 2.1 Fuentes permitidas

| Fuente | Ejemplo | Cuidado |
|---|---|---|
| Fotos propias | Campus, paisajes, objetos, familiares con consentimiento | No publicar rostros sin permiso. |
| Fotos de dominio público | Wikimedia Commons, Library of Congress, Openverse | Verificar licencia y citar. |
| Fotos antiguas escaneadas | Álbumes familiares | Consentimiento; preferir imágenes sin datos sensibles. |

### 2.2 Criterios de calidad

- Resolución mínima: **512×512** (se recorta a 256×256).
- Formato: JPEG/PNG; convertir a RGB.
- Sin marcas de agua, texto dominante ni collages.
- Diversidad: interiores, exteriores, retratos (con permiso), arquitectura, naturaleza.
- Evitar duplicados casi idénticos (usar hash perceptual si se quiere).

### 2.3 Cantidad objetivo

| Nivel | Fotos fuente | Pares (3 variantes) | Uso |
|---|---|---|---|
| Mínimo | 150 | 450 | Demo / entrega ajustada |
| Recomendado | 300 | 900 | Entrenamiento principal |
| Ideal | 500 | 1500 | Mejor generalización |

---

## 3. Degradaciones sintéticas

Cada foto limpia se degrada con una combinación aleatoria. La semilla y los parámetros se
guardan en el manifiesto para poder reproducir el par exacto.

### 3.1 Operaciones

| Operación | Parámetros | Simula |
|---|---|---|
| Desenfoque gaussiano | kernel 3–7, sigma 0.5–2.0 | Foto movida / lente sucia |
| Ruido gaussiano | sigma 5–25 (escala 0–255) | Grano de película / sensor |
| Compresión JPEG | calidad 20–60 | Escaneo viejo / recompresión |
| Baja resolución | factor 2–4, luego upscale | Foto pequeña ampliada |
| Desvanecido de color | mezcla con sepia, saturación 0.3–0.8 | Papel amarillento |
| Rayones | 1–5 líneas, grosor 1–3 px, blanco/negro | Daño físico |
| Polvo | 10–50 círculos pequeños | Suciedad del escáner |
| Contraste/brillo | factor 0.7–1.3 | Revelado deficiente |

### 3.2 Orden sugerido

1. Baja resolución (si aplica).
2. Desenfoque.
3. Ruido.
4. Desvanecido de color.
5. Rayones y polvo.
6. Compresión JPEG (última, para que el artefacto quede visible).

No todas las imágenes deben tener todas las degradaciones. Se sortea con probabilidades
(por ejemplo, 70 % blur, 60 % ruido, 50 % JPEG, 40 % rayones) para que el modelo no aprenda
un solo tipo de daño.

### 3.3 Ejemplo de configuración (YAML)

```yaml
degradations:
  blur:
    probability: 0.7
    kernel: [3, 7]
    sigma: [0.5, 2.0]
  noise:
    probability: 0.6
    sigma: [5, 25]
  jpeg:
    probability: 0.5
    quality: [20, 60]
  downscale:
    probability: 0.4
    factor: [2, 4]
  fade:
    probability: 0.5
    sepia: [0.2, 0.6]
  scratches:
    probability: 0.4
    count: [1, 5]
    thickness: [1, 3]
  dust:
    probability: 0.4
    count: [10, 50]
  contrast:
    probability: 0.3
    factor: [0.7, 1.3]
```

---

## 4. Splits y fuga de datos

### 4.1 Regla principal

**Dividir por foto fuente, nunca por parche ni por variante.** Si una foto fuente tiene 3
variantes y una va a `train` y otra a `test`, el modelo ve la misma escena en entrenamiento y
test: eso es fuga de datos y las métricas quedan infladas.

```text
fuente_001.jpg
  ├─ variante_0 → train
  ├─ variante_1 → train
  └─ variante_2 → train
fuente_002.jpg
  ├─ variante_0 → val
  ...
fuente_003.jpg
  ├─ variante_0 → test
  ...
```

### 4.2 Proporciones

| Split | Porcentaje | Uso |
|---|---|---|
| `train` | 80 % | Entrenamiento. |
| `val` | 10 % | Early stopping, selección de checkpoint. |
| `test` | 10 % | Evaluación final; **no se toca antes**. |

### 4.3 Verificación automática

`scripts/check_dataset.py` debe fallar si:

- Una fuente aparece en más de un split.
- Falta el par `input`/`target`.
- Las imágenes no son RGB o no tienen el tamaño esperado.
- Hay archivos vacíos o corruptos.
- El manifiesto no coincide con los archivos en disco.

Test asociado: `tests/test_dataset.py::test_no_fuga_entre_splits`.

---

## 5. Estructura en disco

```text
data/
├── raw/
│   ├── fuente_001.jpg
│   ├── fuente_002.jpg
│   └── ...
├── processed/
│   ├── train/
│   │   ├── fuente_001_var0_input.png
│   │   ├── fuente_001_var0_target.png
│   │   └── ...
│   ├── val/
│   └── test/
├── sample/                  # 8 imágenes sintéticas para tests/CI
├── manifest.csv             # fuente, split, variante, parámetros, semilla
└── README.md                # ficha del dataset
```

### Formato del manifiesto

```csv
source_id,split,variant,input_path,target_path,seed,degradations
fuente_001,train,0,data/processed/train/fuente_001_var0_input.png,data/processed/train/fuente_001_var0_target.png,42,"blur=3;noise=12;jpeg=35"
fuente_001,train,1,...,...,43,"blur=5;scratch=2"
fuente_002,val,0,...,...,44,"fade=0.4;dust=20"
```

Ventajas:

- Reproducibilidad exacta.
- Estadísticas por split y por tipo de degradación.
- El informe puede citar cuántos pares y qué porcentaje de cada degradación.

---

## 6. Ficha del dataset (`data/README.md`)

Completar esta plantilla:

```markdown
# Dataset — Restauración de fotografías

## Resumen
- Nombre: Fotos UAO Restauración
- Versión: 1.0
- Fecha: 2026-11-XX
- Integrantes: ...
- Número de fuentes: ...
- Número de pares: ...
- Resolución: 256×256 (recorte), original ≥ 512×512
- Splits: train 80 % / val 10 % / test 10 % (por fuente)

## Procedencia
- Fotos propias tomadas por el equipo: ...
- Fotos de dominio público: ... (enlaces y licencias)
- Consentimiento para rostros: sí/no, cómo se documentó.

## Degradaciones
- Tipos y probabilidades: ...
- Semilla base: 42
- Script: `scripts/build_dataset.py`
- Manifiesto: `data/manifest.csv`

## Estadísticas
- Media y desviación de tamaño original.
- Distribución de degradaciones.
- Ejemplos: `docs/img/dataset_grid.png`

## Limitaciones
- Sesgo geográfico/cultural.
- Pocas imágenes nocturnas / interiores / etc.
- Degradaciones sintéticas no cubren todos los daños reales.

## Licencia
- Código: MIT.
- Dataset: CC BY-NC 4.0 (o la que se acuerde), con excepción de fotos de terceros.
- No se publican fotos personales sin consentimiento.

## Uso previsto
- Académico, curso de Redes Neuronales y Deep Learning.
- No usar para vigilancia, identificación de personas ni fines comerciales.
```

---

## 7. Ética y privacidad

- **Consentimiento:** si una foto tiene personas identificables, pedir permiso para usarla en el
  trabajo y para publicarla en el repo. Si no hay permiso, excluirla o difuminar el rostro.
- **Datos sensibles:** nada de documentos, placas, direcciones, menores, información médica.
- **Repo público:** recordar que las fotos quedarán visibles. Si hay duda, usar solo fotos de
  paisajes/objetos o dominio público.
- **Licencias:** citar la fuente de cada foto de terceros. Guardar un `sources.csv` con URL,
  autor y licencia.
- **Uso responsable:** dejar claro en el README que el modelo es académico y puede alucinar
  detalles; no sirve para forense ni para "recuperar" pruebas.

---

## 8. Estadísticas para el informe

Generar con `scripts/build_dataset.py --stats` y reportar:

- Número de fuentes y pares por split.
- Resolución original (media, mín, máx).
- Porcentaje de pares con cada degradación.
- Distribución de severidad (leve, media, fuerte).
- 4–8 ejemplos visuales en `docs/img/dataset_grid.png`.
- Tiempo de construcción del dataset.

---

## 9. Comandos

```bash
# Construir el dataset (ejemplo)
python scripts/build_dataset.py \
  --raw data/raw \
  --out data/processed \
  --variants 3 \
  --seed 42 \
  --config configs/base.yaml

# Validar
python scripts/check_dataset.py --root data/processed --manifest data/manifest.csv

# Generar muestra sintética para tests/CI
python scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64

# Estadísticas
python scripts/build_dataset.py --stats --manifest data/manifest.csv
```

---

## 10. Checklist del dataset

- [ ] 150+ fuentes (ideal 300+).
- [ ] 3 variantes por fuente (o al menos 2).
- [ ] Splits por fuente, sin fuga.
- [ ] Manifiesto con semilla y parámetros.
- [ ] `check_dataset.py` pasa.
- [ ] `data/README.md` completo.
- [ ] Ejemplos visuales en `docs/img/`.
- [ ] Licencias y consentimientos documentados.
- [ ] `data/raw` y `data/processed` fuera de git.
- [ ] Test de no fuga en `tests/test_dataset.py`.
