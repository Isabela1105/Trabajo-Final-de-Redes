# Registro de decisiones (ADR simplificado)

Cada decisión técnica importante se registra aquí. Formato: contexto, decisión, alternativas,
consecuencias. Esto alimenta el informe y la defensa.

---

## ADR-001 — Tema del proyecto

- **Fecha:** Día 1.
- **Contexto:** lista de temas del enunciado; 6 integrantes; ~4 semanas.
- **Decisión:** restauración de fotografías antiguas.
- **Alternativas:** super-resolución, bosquejo→foto, día→noche, pintura→foto.
- **Motivo:** pares fáciles de generar, métricas objetivas, demo vistosa, dificultad media.
- **Consecuencias:** el dataset se construye con degradaciones sintéticas; se necesita cuidar la
  ética si hay rostros.

---

## ADR-002 — Modelo

- **Fecha:** Día 1.
- **Contexto:** se necesita imagen→imagen con control sobre la entrada.
- **Decisión:** Pix2Pix (U-Net + PatchGAN).
- **Alternativas:** CycleGAN (no pareada), SRGAN (super-resolución), StyleGAN2 (no condicional).
- **Motivo:** datos pareados disponibles, arquitectura explicable, buen soporte en TF.
- **Consecuencias:** requiere pares; puede alucinar detalles en zonas muy dañadas.

---

## ADR-003 — Framework

- **Fecha:** Día 1.
- **Decisión:** TensorFlow 2.16 / Keras 3.
- **Alternativas:** PyTorch.
- **Motivo:** recomendado en el enunciado; tutorial oficial de Pix2Pix; el equipo puede usar
  Colab/Kaggle con TF preinstalado.
- **Consecuencias:** en Windows no hay GPU nativa para TF ≥ 2.11; se entrena en Colab/Kaggle o
  WSL2.

---

## ADR-004 — Normalización

- **Fecha:** semana 1.
- **Contexto:** batch size 1–4 por límite de GPU.
- **Decisión:** InstanceNorm en generador y discriminador.
- **Alternativas:** BatchNorm (paper original), GroupNorm.
- **Motivo:** BatchNorm es inestable con batch pequeño; InstanceNorm normaliza por imagen/canal.
- **Consecuencias:** se debe explicar en el informe que es una desviación deliberada del paper.

---

## ADR-005 — Pérdida

- **Fecha:** semana 1.
- **Decisión:** BCEWithLogits + λ·L1 con λ = 100.
- **Alternativas:** MSE (LSGAN), perceptual loss, gradient penalty.
- **Motivo:** es el esquema del paper/tutorial; L1 garantiza fidelidad, adversarial añade textura.
- **Consecuencias:** si el resultado es borroso, revisar λ y el discriminador; gradient penalty
  queda como experimental.

---

## ADR-006 — Splits del dataset

- **Fecha:** semana 1.
- **Decisión:** 80/10/10 por **foto fuente**, no por parche.
- **Motivo:** evitar fuga de datos; una misma escena no debe estar en train y test.
- **Consecuencias:** `check_dataset.py` y un test verifican que no haya fuentes repetidas entre
  splits.

---

## ADR-007 — Estrategia de ramas

- **Fecha:** día 1.
- **Decisión:** GitHub Flow (ramas cortas + PR + squash).
- **Alternativas:** GitFlow, trunk-based sin PR.
- **Motivo:** 6 personas, 4 semanas, se necesita revisión y evidencia de participación.
- **Consecuencias:** `main` protegida; PRs pequeños; una aprobación mínimo.

---

## ADR-008 — Alcance del CI

- **Fecha:** semana 1.
- **Decisión:** CI con lint, tests y smoke training de 2 pasos; sin entrenamiento completo.
- **Motivo:** no hay GPU en GitHub Actions; el objetivo es detectar rupturas, no entrenar.
- **Consecuencias:** el entrenamiento real se documenta con checkpoints y configuración.

---

## Plantilla para nuevas decisiones

```markdown
## ADR-00X — Título

- **Fecha:** ...
- **Contexto:** ...
- **Decisión:** ...
- **Alternativas:** ...
- **Motivo:** ...
- **Consecuencias:** ...
```
