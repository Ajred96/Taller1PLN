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

Actualmente se encuentran implementadas las fases de:

- Preprocesamiento y preparación de datos
- Recuperación semántica con Sentence Transformers

---

# Estructura del proyecto

```plaintext
Proyecto/
│
├── README.md
├── requirements.txt
├── taller2_integrante5.ipynb
│
└── srcTaller2/
    │
    ├── preprocessing/
    │   ├── datasets.py
    │   ├── text_cleaning.py
    │   ├── pdf_loader.py
    │   ├── chunking.py
    │   └── main_preprocessing.py
    │
    ├── embeddings/
    │   ├── sentence_transformers_embeddings.py
    │   ├── semantic_search.py
    │   ├── visualization.py
    │   └── main_embeddings.py
    │
    └── outputs/
        ├── processed/
        │   ├── conll2002_splits.json
        │   ├── ancora_splits.json
        │   ├── conll2002_first_3.json
        │   └── ancora_first_3.json
        ├── chunks/
        │   └── pdf_chunks.json
        ├── embeddings/
        │   ├── embeddings_e5.npy
        │   ├── embeddings_mpnet.npy
        │   ├── embeddings_bge.npy
        │   ├── embeddings_minilm.npy
        │   └── visualizations/
        │       ├── comparacion_scores.png
        │       ├── pca_e5.png
        │       ├── pca_mpnet.png
        │       ├── pca_bge.png
        │       ├── pca_minilm.png
        │       ├── t-sne_e5.png
        │       ├── t-sne_mpnet.png
        │       ├── t-sne_bge.png
        │       └── t-sne_minilm.png
        └── reports/
            └── preprocessing_summary.txt
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

## 7. Recuperación semántica con Sentence Transformers

Se implementó un pipeline completo de recuperación semántica sobre los chunks generados en el paso anterior.

### Modelos evaluados

| ID     | Modelo                                | Dimensiones | Fortaleza                   |
| ------ | ------------------------------------- | ----------- | --------------------------- |
| e5     | intfloat/multilingual-e5-base         | 768         | Balance precisión/velocidad |
| mpnet  | paraphrase-multilingual-mpnet-base-v2 | 768         | Alta calidad semántica      |
| bge    | BAAI/bge-m3                           | 1024        | Mejor retrieval semántico   |
| minilm | paraphrase-multilingual-MiniLM-L12-v2 | 384         | Rápido y ligero             |

### Proceso

Para cada modelo se:

1. descargó el modelo desde HuggingFace Hub,
2. generaron embeddings para los 6796 chunks con `normalize_embeddings=True`,
3. almacenaron los vectores como archivos `.npy`,
4. calculó la similitud coseno entre una consulta y todos los fragmentos,
5. identificó el fragmento más similar.

### Consulta de ejemplo utilizada

```
"El coronel recordó su pasado mientras enfrentaba la muerte"
```

### Resultados de similitud coseno

| Modelo | Score  | Archivo recuperado |
| ------ | ------ | ------------------ |
| e5     | 0.8422 | coronel-no.pdf     |
| mpnet  | 0.7514 | coronel-no.pdf     |
| bge    | 0.5940 | coronel-no.pdf     |
| minilm | 0.6894 | coronel-no.pdf     |

### Visualización

Para cada modelo se generaron gráficas PCA y t-SNE mostrando:

- la posición de la consulta (rojo),
- el fragmento más similar (azul),
- una muestra de 500 chunks de fondo (gris).

Las visualizaciones están disponibles en:

```plaintext
srcTaller2/outputs/embeddings/visualizations/
```

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
- Generación de embeddings con 4 modelos de Sentence Transformers
- Recuperación semántica por similitud coseno
- Visualización PCA y t-SNE
- Comparación entre modelos de embeddings

## Pendiente

- WordPiece
- SentencePiece
- Word2Vec
- FastText

---

# Archivos generados

## Chunks

```plaintext
srcTaller2/outputs/chunks/pdf_chunks.json
```

## Embeddings

```plaintext
srcTaller2/outputs/embeddings/embeddings_e5.npy
srcTaller2/outputs/embeddings/embeddings_mpnet.npy
srcTaller2/outputs/embeddings/embeddings_bge.npy
srcTaller2/outputs/embeddings/embeddings_minilm.npy
```

## Visualizaciones

```plaintext
srcTaller2/outputs/embeddings/visualizations/comparacion_scores.png
srcTaller2/outputs/embeddings/visualizations/pca_e5.png
srcTaller2/outputs/embeddings/visualizations/pca_mpnet.png
srcTaller2/outputs/embeddings/visualizations/pca_bge.png
srcTaller2/outputs/embeddings/visualizations/pca_minilm.png
srcTaller2/outputs/embeddings/visualizations/t-sne_e5.png
srcTaller2/outputs/embeddings/visualizations/t-sne_mpnet.png
srcTaller2/outputs/embeddings/visualizations/t-sne_bge.png
srcTaller2/outputs/embeddings/visualizations/t-sne_minilm.png
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
seaborn
tqdm
```

---

# Cómo ejecutar el preprocesamiento

Desde la raíz del proyecto:

```bash
python -m srcTaller2.main_preprocessing
```

# Cómo ejecutar tokenización con t5-small (SentencePiece)

Desde `Taller1PLN\srcTaller2`:

```bash
pip install transformers huggingface_hub pandas

python tokenization_t5-small.py
```

# Cómo ejecutar el pipeline de embeddings semánticos

Desde la raíz del proyecto:

```bash
python -m srcTaller2.embeddings.main_embeddings
```

Con consulta personalizada:

```bash
python -m srcTaller2.embeddings.main_embeddings --consulta "tu frase aquí"
```

Para forzar la regeneración de embeddings aunque existan en disco:

```bash
python -m srcTaller2.embeddings.main_embeddings --regenerar
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
      ↓
Generación de embeddings (4 modelos)
      ↓
Recuperación semántica (similitud coseno)
      ↓
Visualización PCA y t-SNE
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

- Sentence Transformers ✔ (completado)
- Embeddings semánticos ✔ (completado)
- Similaridad coseno ✔ (completado)
- Retrieval semántico ✔ (completado)
