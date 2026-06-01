# Taller 2 - Datasets, Tokenización y Embeddings

## Integrantes

- Cristian Camilo Llanos Alvarez - 1943852
- Anderson Johan Alban Angulo - 2310006
- Gustavo Adolfo Arango Nieves - 2310133
- Juan Esteban Clavijo García - 2225709
- Jhoan Felipe Leon Correa - 2228527

---

# Introducción

El objetivo de este taller es analizar diferentes estrategias modernas utilizadas en Procesamiento de Lenguaje Natural (
PLN), incluyendo:

- Tokenización con WordPiece y SentencePiece.
- Entrenamiento de embeddings distribucionales usando Word2Vec y FastText.
- Visualización semántica mediante PCA y t-SNE.
- Recuperación semántica basada en embeddings utilizando Sentence Transformers.
- Procesamiento y fragmentación de documentos PDF para sistemas de búsqueda semántica.

El desarrollo del proyecto fue dividido entre los integrantes del grupo para modularizar cada componente del pipeline.

---

# Estado actual del proyecto

Actualmente se encuentra implementada la fase de:

# Preprocesamiento y preparación de datos

Esta fase corresponde a la base del pipeline y prepara todos los recursos que serán utilizados posteriormente por los
demás integrantes del grupo.

---

# Estructura del proyecto

```plaintext
Proyecto/
│
├── requirements.txt
├── srcTaller2/
│
├── preprocessing/
│   ├── datasets.py
│   ├── text_cleaning.py
│   ├── pdf_loader.py
│   ├── chunking.py
│   └── main_preprocessing.py
│
├── outputs/
│   ├── processed/
│   ├── chunks/
│   └── reports/
│
└── data/
    └── raw/
        ├── ancora/
        ├── conll2002/
        └── pdfs/
```

---

# Funcionalidades implementadas

## 1. Carga de datasets

Se implementó la carga de:

- CoNLL2002
- Ancora

### CoNLL2002

Se utilizaron directamente los conjuntos originales:

- train
- validation
- test

### Ancora

Se realizó una división manual del dataset:

- 70% entrenamiento
- 15% validación
- 15% prueba

utilizando `train_test_split` de Scikit-Learn.

---

# 2. Exportación de datasets procesados

El sistema genera automáticamente:

```plaintext
srcTaller2/outputs/processed/
```

Archivos generados:

```plaintext
conll2002_splits.json
ancora_splits.json
conll2002_first_3.json
ancora_first_3.json
```

Estos archivos contienen:

- conjuntos train/validation/test,
- primeras 3 oraciones,
- estructuras listas para tokenización.

---

# 3. Preprocesamiento textual

Se implementó limpieza de texto para preparar sentencias destinadas a:

- Word2Vec
- FastText
- Gensim

## Limpieza aplicada

- Conversión a minúsculas
- Eliminación de puntuación
- Eliminación de números
- Eliminación de stopwords
- Eliminación de tokens vacíos

Ejemplo:

```python
["Hola", ",", "Mundo", "2025"]
```

Resultado:

```python
["hola", "mundo"]
```

---

# 4. Carga de Spanish Billion Words

Se implementó la carga del corpus:

```plaintext
crscardellino/spanish_billion_words
```

usando HuggingFace Datasets.

El dataset fue procesado para generar:

```python
list[list[str]]
```

listas de tokens compatibles con:

- Gensim
- Word2Vec
- FastText

Ejemplo:

```python
[
    ["familia", "dashwood", "llevaba"],
    ["propiedad", "buen", "tamaño"]
]
```

---

# 5. Procesamiento de PDFs

Se implementó la carga automática de documentos PDF desde:

```plaintext
data/raw/pdfs/
```

Para la extracción de texto se utilizó:

```python
PyMuPDF(fitz)
```

## Funcionalidades

- detección automática de archivos PDF,
- extracción de texto por página,
- lectura multipágina,
- consolidación del contenido textual de cada documento.

---

# 6. Chunking de documentos con LangChain

Se implementó la fragmentación de documentos para recuperación semántica utilizando `RecursiveCharacterTextSplitter` del
ecosistema LangChain, mediante el paquete:

```plaintext
langchain-text-splitters
```

Esta decisión permite cumplir con la recomendación del taller, manteniendo una instalación más ligera y estable que el
paquete completo de LangChain.

## Configuración utilizada

```python
chunk_size = 1000
chunk_overlap = 200
```

## Fragmentador utilizado

```python
RecursiveCharacterTextSplitter
```

## Objetivo

La fragmentación permite:

- mantener coherencia semántica,
- preservar parte del contexto entre fragmentos mediante solapamiento,
- mejorar la recuperación basada en embeddings,
- facilitar búsquedas semánticas posteriores con Sentence Transformers.

Cada chunk generado contiene:

```json
{
  "filename": "...",
  "chunk_id": 0,
  "text": "...",
  "chunk_size": 968,
  "splitter": "RecursiveCharacterTextSplitter",
  "chunk_size_config": 1000,
  "chunk_overlap_config": 200
}
```

En la última ejecución del pipeline se procesaron:

- 12 documentos PDF,
- 6796 chunks generados.

---

# Estado actual

## Completado

- Preprocesamiento datasets
- Limpieza textual
- Spanish Billion Words
- Carga automática de PDFs
- Extracción de texto con PyMuPDF
- Chunking con RecursiveCharacterTextSplitter
- Exportación JSON

## Pendiente

- WordPiece
- SentencePiece
- Word2Vec
- FastText
- PCA
- t-SNE
- Sentence Transformers
- Semantic Search
- Visualización embeddings

---

# Archivos generados

## Chunks

```plaintext
srcTaller2/outputs/chunks/pdf_chunks.json
```

## Reportes

```plaintext
srcTaller2/outputs/reports/preprocessing_summary.txt
```

---

# Dependencias utilizadas

Instalación:

```bash
pip install -r requirements.txt
```

Dependencias principales:

```plaintext
pandas
numpy
scikit-learn
nltk
datasets
pymupdf
sentence-transformers
gensim
matplotlib
```

---

# Cómo ejecutar el preprocesamiento

Desde la raíz del proyecto:

```bash
python -m srcTaller2.main_preprocessing

```
#cómo ejecutar tokenización con t5-small(sentecePiece)

Desde Taller1PLN\srcTaller2:

```bash
pip install transformers huggingface_hub pandas

python tokenization_t5-small.py

```
---

```

---

# Flujo actual del pipeline

```plaintext
Carga datasets
      ↓
Preprocesamiento texto
      ↓
Spanish Billion Words
      ↓
Carga PDFs
      ↓
Extracción texto
      ↓
Chunking
      ↓
Exportación JSON
```

---

# Salidas disponibles para los demás integrantes

Los siguientes archivos deben ser utilizados por los siguientes módulos del proyecto:

## Tokenización

Archivos:

```plaintext
conll2002_first_3.json
ancora_first_3.json
```

Uso:

- WordPiece
- SentencePiece

---

## Embeddings distribucionales

Archivo fuente:

```plaintext
Spanish Billion Words procesado
```

Uso:

- Word2Vec
- FastText
- PCA
- t-SNE

---

## Recuperación semántica

Archivo:

```plaintext
pdf_chunks.json
```

Uso:

- Sentence Transformers
- Embeddings semánticos
- Similaridad coseno
- Retrieval semántico