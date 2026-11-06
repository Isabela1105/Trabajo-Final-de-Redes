## ¿Qué cambia este PR?

<!-- Explica el problema y la solución en 3–6 líneas. -->

## Issue relacionado

Closes #

## Tipo de cambio

- [ ] `feat`: nueva funcionalidad
- [ ] `fix`: corrección de bug
- [ ] `docs`: documentación
- [ ] `test`: pruebas
- [ ] `refactor`: refactorización sin cambio funcional
- [ ] `perf`: mejora de rendimiento
- [ ] `ci`: CI/CD
- [ ] `chore`: mantenimiento

## ¿Cómo se probó?

<!-- Comandos, capturas, métricas antes/después. -->

```bash
pytest -q
python -m src.gan.train --config configs/smoke.yaml --max-steps 2
```

## Evidencia visual (si aplica)

<!-- Arrastra una imagen o GIF: input | target | predicción. -->

## Checklist

- [ ] `pre-commit run --all-files` pasa.
- [ ] `pytest -q` pasa.
- [ ] Agregué/actualicé tests para el cambio.
- [ ] Docstrings y comentarios en español; nombres en inglés.
- [ ] Actualicé la documentación si cambió una interfaz o hiperparámetro.
- [ ] No subí datos, pesos, `.env` ni archivos mayores a 5 MB.
- [ ] El PR tiene menos de ~400 líneas cambiadas (si no, lo dividí).
- [ ] Asigné revisor distinto de mí.

## Notas para el revisor

<!-- Decisiones de diseño, dudas, partes que quieres que revise con más cuidado. -->
