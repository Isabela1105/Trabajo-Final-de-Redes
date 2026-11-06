# Guías individuales

Una guía por integrante. Cada una contiene: rol, archivos que posee, ramas, PRs, commits,
revisores, criterios de aceptación, preguntas de defensa y checklist final.

| Integrante | GitHub | Rol | Guía |
|---|---|---|---|
| Isabella Ortiz Hernandez | `@Isabellaortizher-collab` | Datos y dataset | [`GUIA-Isabellaortizher-collab.md`](GUIA-Isabellaortizher-collab.md) |
| Juan Pablo Dávila Moncayo | `@Juanpablodavila` | Generador U-Net | [`GUIA-Juanpablodavila.md`](GUIA-Juanpablodavila.md) |
| Jeremy Santiago Caicedo Cabezas | `@JeremyCaicedo` | Discriminador y pérdidas | [`GUIA-JeremyCaicedo.md`](GUIA-JeremyCaicedo.md) |
| Cristian Fabian Muñoz Piedrahita | `@CristianMP355` | Entrenamiento | [`GUIA-CristianMP355.md`](GUIA-CristianMP355.md) |
| Juan Sebastian Castillo Acevedo | `@Juansec01` | Evaluación y demo | [`GUIA-Juansec01.md`](GUIA-Juansec01.md) |
| Isabela | `@Isabela1105` | CI/CD, integración e informe | [`GUIA-Isabela1105.md`](GUIA-Isabela1105.md) |

---

## Orden de lectura sugerido

1. [`../README.md`](../README.md) — contexto, arquitectura, flujo y checklist de entrega.
2. [`../docs/EQUIPO.md`](../docs/EQUIPO.md) — roles, propiedad de archivos y revisores.
3. [`../docs/PLAN_COMMITS.md`](../docs/PLAN_COMMITS.md) — commits exactos por semana.
4. Tu guía individual.
5. [`../docs/MERGE_Y_REVISION.md`](../docs/MERGE_Y_REVISION.md) — antes de tu primer PR y al final.
6. [`../docs/DEFENSA.md`](../docs/DEFENSA.md) — desde la semana 1, no la última noche.

---

## Nota sobre la numeración

Los números que aparecen en los nombres de rama (`docs/03-...`, `feat/10-...`) y en los mensajes
de commit (`(#10)`) son **identificadores del plan**, no números de issue de GitHub. Al crear el
issue real, GitHub asigna un número consecutivo distinto. En el commit y en el PR usa el número
real: `Closes #<numero-real>`.

Si dos ramas del plan comparten número (por ejemplo `feat/14-dataset-v2` y `feat/14-train-loop`),
antepón el área para que sean únicas en el repo — ya lo son por el prefijo, pero si GitHub marca
conflicto de nombre, agrega el apellido: `feat/14-train-loop-cristian`.

---

## Regla común para todos

- Nadie hace push directo a `main`.
- Un issue → una rama → un PR.
- El autor no aprueba su propio PR.
- CI verde obligatorio.
- Cada PR deja el repo ejecutable.
- Cada integrante hace su primer commit en los días 1–2.
- Después del día 24 no se agregan features.
- Todos deben poder explicar **todo**, no solo su módulo.
