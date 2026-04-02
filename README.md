# POS Tagging en Espanol

Proyecto del curso de **Procesamiento de Lenguaje Natural** (Univalle).
Implementa un sistema de **POS Tagging** utilizando modelos basados en
**BiLSTM** sobre datasets en espanol.

---

## Objetivo

Construir y evaluar modelos de etiquetado gramatical (Part-of-Speech Tagging)
para espanol utilizando los datasets **Ancora** y **CoNLL2002**.

---

## Modelos implementados

| Modelo | Descripcion |
|--------|-------------|
| **BiLSTM Base** | Embedding + BiLSTM + FC |
| **BiLSTM Deep** | Embedding + BiLSTM + Dense + ReLU + Dropout + FC |
| **BiLSTM-CRF** | Embedding + BiLSTM + FC + CRF (decodificacion Viterbi) |

Cada modelo se entrena con **grid search de 24 combinaciones** de hiperparametros
(batch_size, optimizer, embedding_dim, hidden_dim) con early stopping.

---

## Datasets

| Metrica | Ancora | CoNLL2002 |
|---------|--------|-----------|
| Train sentences | 12,141 | 8,323 |
| Val sentences | 2,602 | 1,915 |
| Test sentences | 2,602 | 1,517 |
| Vocabulario palabras | 36,131 | 26,101 |
| Etiquetas POS | 17 (universales) | 61 (EAGLES) |
| Max sequence length | 148 | 1,238 |

**Nota:** Los esquemas de etiquetas son incompatibles entre si.
Ancora usa etiquetas universales (`NOUN`, `VERB`, `DET`) mientras que
CoNLL2002 usa etiquetas EAGLES (`NC`, `VMI`, `DA`).

---

## Estructura del proyecto

```
Taller1PLN/
|-- main.py                 <- Menu principal
|-- config.py               <- Paths centralizados
|-- requirements.txt        <- Dependencias
|
|-- data/raw/               <- Datasets crudos
|   |-- ancora/             <- ancora_corpus_pos.csv
|   |-- conll2002/          <- train.txt, valid.txt, test.txt
|
|-- src/
|   |-- preprocessing/      <- Carga, limpieza, vocabularios, DataLoaders
|   |   |-- utils.py        <- Funciones compartidas (vocab, encode, pad, split)
|   |   |-- ancora.py       <- Pipeline completo Ancora
|   |   |-- conll.py        <- Pipeline completo CoNLL2002
|   |   |-- export.py       <- Exportar datos procesados a .pt
|   |
|   |-- training/           <- Entrenamiento de modelos
|   |   |-- bilstm_base/    <- model, train, predict, grid_search
|   |   |-- bilstm_deep/    <- model, train, predict, grid_search
|   |   |-- bilstm_crf/     <- model, train, predict, grid_search
|   |
|   |-- inference/          <- Uso de modelos para etiquetar texto
|       |-- tagger.py       <- Cargar modelos y etiquetar oraciones
|
|-- outputs/
    |-- artifacts/          <- Vocabularios JSON
    |-- models/             <- Modelos entrenados (.pt)
    |-- processed/          <- Datos serializados (.pt)
    |-- reports/            <- Reportes de evaluacion
```

---

## Como ejecutar

```bash
# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ejecutar el menu principal
python main.py
```

### Opciones del menu

| Opcion | Descripcion |
|--------|-------------|
| **1a-1c** | Preprocesar Ancora, CoNLL o ambos |
| **1d** | Exportar datos procesados (.pt) |
| **2a** | Entrenar BiLSTM Base (Ancora + CoNLL) |
| **2b** | Entrenar BiLSTM Deep (Ancora + CoNLL) |
| **2c** | Entrenar BiLSTM-CRF (Ancora + CoNLL) |
| **3a** | Etiquetar oraciones (modo interactivo) |
| **3b** | Ver modelos disponibles |

### Orden de ejecucion para un pipeline completo

1. `1c` - Preprocesar ambos datasets
2. `1d` - Exportar datos procesados
3. `2a`, `2b`, `2c` - Entrenar modelos
4. `3a` - Etiquetar oraciones

---

## Reportes generados

Al entrenar cada modelo se generan automaticamente:

- **Reporte individual** por modelo y dataset (`outputs/reports/bilstm_ancora_report.txt`)
  con metricas globales, metricas por etiqueta y detalle del grid search.
- **Tabla comparativa global** (`outputs/reports/tabla_comparativa_global.txt`)
  con todos los modelos entrenados.

Formato de la tabla:

```
Modelo       | Dataset   | Accuracy | Precision | Recall | F1-Score
BiLSTM       | Ancora    | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
BiLSTM+Dense | Ancora    | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
BiLSTM-CRF   | Ancora    | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
BiLSTM       | CoNLL2002 | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
BiLSTM+Dense | CoNLL2002 | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
BiLSTM-CRF   | CoNLL2002 | 0.XXXX   | 0.XXXX    | 0.XXXX | 0.XXXX
```
