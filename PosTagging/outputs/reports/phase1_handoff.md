# Handoff - Fase I

## Objetivo
La Fase I deja preparados los datasets para las siguientes etapas del proyecto de POS Tagging.

## Datasets procesados
- Ancora
- CoNLL2002

---

## Ancora
### Estado
Completado.

### Pipeline realizado
- carga y limpieza del CSV
- construcción de oraciones usando `Sentence #`
- split train / validation / test = 70 / 15 / 15
- vocabulario de palabras construido solo con train
- vocabulario de etiquetas POS construido solo con train
- codificación de palabras y etiquetas
- padding
- máscaras
- Dataset de PyTorch
- DataLoader

### Resumen técnico
- Oraciones totales: 17345
- Vocabulario de palabras: 36131
- Vocabulario de etiquetas POS: 17
- Max length: 148

### Artefactos
- `outputs/artifacts/ancora_word2idx.json`
- `outputs/artifacts/ancora_tag2idx.json`

---

## CoNLL2002
### Estado
Completado.

### Pipeline realizado
- carga de `train.txt`, `valid.txt`, `test.txt`
- construcción de oraciones usando líneas vacías
- vocabulario de palabras construido solo con train
- vocabulario POS construido solo con train
- soporte para `<UNK>` y `<UNK_TAG>`
- codificación
- padding
- máscaras
- Dataset de PyTorch
- DataLoader

### Resumen técnico
- Train: 8323
- Validation: 1915
- Test: 1517
- Vocabulario de palabras: 26101
- Vocabulario de etiquetas POS: 61
- Max length: 1238

### Observación importante
CoNLL2002 contiene secuencias muy largas. Antes del entrenamiento conviene evaluar:
- truncamiento
- uso de percentiles de longitud
- batching más eficiente

### Artefactos
- `outputs/artifacts/conll_word2idx.json`
- `outputs/artifacts/conll_tag2idx.json`

---

## Nota crítica sobre ambos datasets
No deben mezclarse directamente como si usaran el mismo esquema POS.

- Ancora usa etiquetas universales tipo `NOUN`, `VERB`, `DET`
- CoNLL2002 usa etiquetas tipo EAGLES como `NC`, `VMI`, `DA`, `AQ`

Si se quiere entrenar sobre ambos datasets juntos, primero hay que definir una estrategia de mapeo de etiquetas.

---

## Qué debe usar el equipo en la siguiente fase
Para entrenamiento o experimentación posterior, usar:

- `train_loader`
- `val_loader`
- `test_loader`
- `word2idx`
- `tag2idx`
- `max_len`

Los scripts principales de ejecución son:
- `src/preprocessing/run_phase1_ancora.py`
- `src/preprocessing/run_phase1_conll.py`
- `src/preprocessing/run_phase1_all.py`