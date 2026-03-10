# POS Tagging en Español

Proyecto del curso de **Procesamiento de Lenguaje Natural** para
implementar un sistema de **POS Tagging** utilizando modelos basados en
**BiLSTM** sobre datasets en español.

El proyecto está dividido en varias fases que se desarrollarán
progresivamente por los integrantes del equipo.

------------------------------------------------------------------------

# Objetivo del proyecto

Construir y evaluar modelos de etiquetado gramatical (**Part-of-Speech
Tagging**) para español utilizando:

-   Dataset **Ancora**
-   Dataset **CoNLL2002**

El pipeline completo del proyecto incluirá:

1.  Preprocesamiento de datos\
2.  Construcción de vocabularios\
3.  Modelos de Deep Learning (BiLSTM)\
4.  Entrenamiento\
5.  Evaluación

------------------------------------------------------------------------

# Estado actual del repositorio

Actualmente se ha completado la **FASE I: Preprocesamiento de los
datasets**.

Las siguientes fases del proyecto serán implementadas posteriormente por
otros integrantes del equipo.

------------------------------------------------------------------------

# Fase I --- Preprocesamiento de datos

El código de esta fase se encuentra en:

    src/preprocessing/

Esta fase incluye:

-   carga de datasets\
-   limpieza de datos\
-   construcción de secuencias de oraciones\
-   creación de vocabularios\
-   indexación de palabras y etiquetas\
-   partición de datos\
-   padding de secuencias\
-   creación de máscaras\
-   construcción de `Dataset` y `DataLoader` para PyTorch

El objetivo de esta fase es dejar los datos **listos para ser utilizados
por los modelos de entrenamiento en las siguientes etapas del
proyecto**.

------------------------------------------------------------------------

# Datasets utilizados

## Ancora

Formato original del dataset:

    Sentence #
    Word
    POS

Pipeline aplicado:

-   limpieza de columnas
-   construcción de oraciones usando `Sentence #`
-   split train / validation / test = **70 / 15 / 15**
-   vocabulario de palabras construido solo con train
-   indexación de palabras y etiquetas
-   padding de secuencias
-   generación de máscaras
-   construcción de `Dataset` y `DataLoader`

Resumen del dataset procesado:

    Oraciones totales: 17345
    Vocabulario de palabras (train): 36131
    Etiquetas POS: 17
    Max sequence length: 148

------------------------------------------------------------------------

## CoNLL2002

Formato original del dataset:

    Word POS NER

Las oraciones están separadas por **líneas vacías**.

Pipeline aplicado:

-   lectura de `train.txt`, `valid.txt`, `test.txt`
-   construcción de oraciones por separación de líneas vacías
-   vocabulario de palabras construido solo con train
-   vocabulario POS construido solo con train
-   soporte para etiquetas desconocidas (`<UNK_TAG>`)
-   indexación
-   padding
-   generación de máscaras
-   construcción de `Dataset` y `DataLoader`

Resumen del dataset procesado:

    Train sentences: 8323
    Validation sentences: 1915
    Test sentences: 1517

    Vocabulario de palabras: 26101
    Vocabulario de etiquetas POS: 61

    Max sequence length: 1238

Nota:

CoNLL2002 contiene secuencias muy largas, por lo que en fases
posteriores puede ser necesario aplicar **truncamiento o limitar la
longitud máxima** para mejorar eficiencia durante el entrenamiento.

------------------------------------------------------------------------

# Estructura del proyecto

    data/
     ├── raw/
     │   ├── ancora/
     │   │   └── ancora_corpus_pos.csv
     │   └── conll2002/
     │       ├── train.txt
     │       ├── valid.txt
     │       └── test.txt

    src/
     ├── preprocessing/
     ├── models/
     └── training/

    outputs/
     ├── artifacts/
     └── reports/

------------------------------------------------------------------------

# Scripts principales

Para ejecutar el preprocesamiento completo:

    src/preprocessing/run_phase1_all.py

También es posible ejecutar cada dataset por separado:

    src/preprocessing/run_phase1_ancora.py
    src/preprocessing/run_phase1_conll.py

------------------------------------------------------------------------

# Cómo ejecutar la Fase I

Desde la raíz del proyecto:

``` bash
python src/preprocessing/run_phase1_all.py
```

Esto generará los artefactos de preprocesamiento y los reportes
correspondientes.

------------------------------------------------------------------------

# Artefactos generados

Los artefactos generados por la Fase I se guardan en:

    outputs/artifacts/

Incluyen:

-   vocabulario de palabras
-   vocabulario de etiquetas POS

Reportes generados:

    outputs/reports/

-   `phase1_ancora_summary.txt`
-   `phase1_conll_summary.txt`
-   `phase1_handoff.md`

------------------------------------------------------------------------

# Nota importante sobre los datasets

Ancora y CoNLL2002 **no utilizan el mismo esquema de etiquetas POS**.

Ancora usa etiquetas universales como:

    NOUN, VERB, DET, ADJ, ADV

Mientras que CoNLL2002 utiliza etiquetas tipo **EAGLES**, por ejemplo:

    NC, NP, DA, AQ, VMI, RG

Por lo tanto, **no deben mezclarse directamente sin definir previamente
un mapeo de etiquetas**.

------------------------------------------------------------------------

# Próximas fases del proyecto

Las siguientes etapas del proyecto incluirán:

-   implementación de modelos **BiLSTM**
-   entrenamiento del modelo
-   evaluación
-   comparación entre datasets

Estas fases serán desarrolladas por otros integrantes del equipo.

------------------------------------------------------------------------

# Integrantes del equipo

**Anderson Johan Alban Angulo** --- Preprocesamiento
