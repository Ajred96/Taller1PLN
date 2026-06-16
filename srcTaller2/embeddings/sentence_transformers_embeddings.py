"""
sentence_transformers_embeddings.py
====================================
Carga los 4 modelos de Sentence Transformers y genera embeddings
para todos los chunks extraídos de los PDFs.

Modelos evaluados:
    - intfloat/multilingual-e5-base         (768 dims)
    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768 dims)
    - BAAI/bge-m3                           (1024 dims)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Configuración ────────────────────────────────────────────────────────────

MODELOS = {
    "e5":     "intfloat/multilingual-e5-base",
    "mpnet":  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge":    "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

# Rutas (ajusta si tu estructura es diferente)
CHUNKS_PATH    = Path("srcTaller2/outputs/chunks/pdf_chunks.json")
EMBEDDINGS_DIR = Path("srcTaller2/outputs/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Funciones ────────────────────────────────────────────────────────────────

def cargar_chunks(path: Path) -> tuple[list[dict], list[str]]:
    """
    Lee el archivo pdf_chunks.json y devuelve:
        - chunks: lista completa de objetos con metadata
        - textos: lista de strings listos para encodear
    """
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    textos = [c["text"] for c in chunks]
    print(f"✔ Chunks cargados: {len(chunks)}")
    return chunks, textos


def generar_embeddings(
    textos: list[str],
    nombre_modelo: str,
    id_modelo: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Descarga el modelo, genera embeddings para todos los textos
    y devuelve un array numpy de shape (n_chunks, dimensiones).

    Args:
        textos:        lista de strings a encodear
        nombre_modelo: nombre en HuggingFace Hub
        id_modelo:     clave corta usada para guardar el archivo
        batch_size:    chunks procesados en paralelo (baja si hay OOM)

    Returns:
        embeddings: np.ndarray float32
    """
    print(f"\n── Cargando modelo: {nombre_modelo}")
    modelo = SentenceTransformer(nombre_modelo)

    print(f"   Generando embeddings para {len(textos)} chunks...")
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # necesario para similitud coseno directa
    )

    # Guardar en disco
    ruta_salida = EMBEDDINGS_DIR / f"embeddings_{id_modelo}.npy"
    np.save(ruta_salida, embeddings)
    print(f"   ✔ Guardado en {ruta_salida}  |  shape: {embeddings.shape}")

    return embeddings


def generar_todos_los_embeddings(
    textos: list[str],
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Itera sobre los 4 modelos y genera embeddings para cada uno.

    Returns:
        dict con claves {"e5", "mpnet", "bge", "minilm"} y valores np.ndarray
    """
    resultados = {}
    for id_modelo, nombre_modelo in MODELOS.items():
        embeddings = generar_embeddings(textos, nombre_modelo, id_modelo, batch_size)
        resultados[id_modelo] = embeddings
    print("\n✔ Embeddings generados para todos los modelos.")
    return resultados


def cargar_embeddings_desde_disco() -> dict[str, np.ndarray]:
    """
    Carga embeddings previamente guardados sin volver a encodear.
    Útil para reutilizar en búsqueda o visualización sin re-ejecutar.
    """
    resultados = {}
    for id_modelo in MODELOS:
        ruta = EMBEDDINGS_DIR / f"embeddings_{id_modelo}.npy"
        if ruta.exists():
            resultados[id_modelo] = np.load(ruta)
            print(f"✔ Cargado {id_modelo}: shape {resultados[id_modelo].shape}")
        else:
            print(f"✗ No encontrado: {ruta}")
    return resultados


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks, textos = cargar_chunks(CHUNKS_PATH)
    embeddings_por_modelo = generar_todos_los_embeddings(textos)
