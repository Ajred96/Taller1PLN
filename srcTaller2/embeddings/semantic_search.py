"""
semantic_search.py
====================================
Recuperación semántica: dado un texto de consulta, encuentra
el fragmento más similar en cada uno de los 4 modelos usando
similitud coseno.

Dado que los embeddings fueron normalizados en la generación
(normalize_embeddings=True), la similitud coseno se reduce a
un producto punto, lo que es más eficiente.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─── Configuración ────────────────────────────────────────────────────────────

MODELOS = {
    "e5":     "intfloat/multilingual-e5-base",
    "mpnet":  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge":    "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

EMBEDDINGS_DIR = Path("srcTaller2/outputs/embeddings")


# ─── Funciones ────────────────────────────────────────────────────────────────

def encodear_consulta(consulta: str, nombre_modelo: str) -> np.ndarray:
    """
    Genera el embedding de una oración de consulta usando el modelo indicado.

    Args:
        consulta:      texto ingresado por el usuario o definido en el código
        nombre_modelo: nombre en HuggingFace Hub

    Returns:
        vector normalizado de shape (dimensiones,)
    """
    modelo = SentenceTransformer(nombre_modelo)
    vector = modelo.encode(
        consulta,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector


def similitud_coseno(vec_consulta: np.ndarray, matriz_embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud coseno entre la consulta y todos los chunks.

    Como los vectores están normalizados, esto equivale al producto punto.
    El resultado es un array de scores en [-1, 1] donde 1 es idéntico.

    Args:
        vec_consulta:      shape (dimensiones,)
        matriz_embeddings: shape (n_chunks, dimensiones)

    Returns:
        scores: shape (n_chunks,)
    """
    scores = matriz_embeddings @ vec_consulta  # producto punto vectorizado
    return scores


def recuperar_top_k(
    consulta: str,
    chunks: list[dict],
    embeddings_por_modelo: dict[str, np.ndarray],
    k: int = 5,
) -> dict[str, list[dict]]:
    """
    Para cada modelo, genera el embedding de la consulta y recupera
    los k fragmentos más similares.

    Args:
        consulta:              texto de búsqueda
        chunks:                lista completa de objetos chunk con metadata
        embeddings_por_modelo: dict {id_modelo: np.ndarray}
        k:                     número de resultados a retornar por modelo

    Returns:
        dict con claves de modelo y listas de los top-k chunks con su score
    """
    resultados = {}

    for id_modelo, nombre_modelo in MODELOS.items():
        if id_modelo not in embeddings_por_modelo:
            print(f"✗ Embeddings no disponibles para: {id_modelo}")
            continue

        print(f"\n── Buscando con modelo: {id_modelo}")

        # Embedding de la consulta
        vec_consulta = encodear_consulta(consulta, nombre_modelo)

        # Similitud contra todos los chunks
        matriz = embeddings_por_modelo[id_modelo]
        scores = similitud_coseno(vec_consulta, matriz)

        # Índices ordenados de mayor a menor similitud
        indices_top = np.argsort(scores)[::-1][:k]

        top_chunks = []
        for idx in indices_top:
            chunk_info = {
                "rank":      len(top_chunks) + 1,
                "score":     float(scores[idx]),
                "chunk_id":  chunks[idx]["chunk_id"],
                "filename":  chunks[idx]["filename"],
                "text":      chunks[idx]["text"],
            }
            top_chunks.append(chunk_info)

        resultados[id_modelo] = top_chunks

        # Mostrar el resultado #1
        mejor = top_chunks[0]
        print(f"   Mejor resultado — score: {mejor['score']:.4f}")
        print(f"   Archivo: {mejor['filename']}  |  chunk_id: {mejor['chunk_id']}")
        print(f"   Texto: {mejor['text'][:150]}...")

    return resultados


def mostrar_comparacion(resultados: dict[str, list[dict]], consulta: str) -> None:
    """
    Imprime una tabla comparativa del resultado #1 de cada modelo
    para facilitar el análisis de diferencias.
    """
    print("\n" + "═" * 70)
    print(f"CONSULTA: \"{consulta}\"")
    print("═" * 70)
    print(f"{'Modelo':<10} {'Score':>7}  {'Archivo':<20}  Fragmento (primeros 100 chars)")
    print("─" * 70)

    for id_modelo, top_chunks in resultados.items():
        mejor = top_chunks[0]
        texto_corto = mejor["text"][:100].replace("\n", " ")
        print(f"{id_modelo:<10} {mejor['score']:>7.4f}  {mejor['filename']:<20}  {texto_corto}")

    print("═" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from sentence_transformers_embeddings import cargar_chunks, cargar_embeddings_desde_disco, CHUNKS_PATH

    # Cargar datos
    chunks, textos = cargar_chunks(CHUNKS_PATH)
    embeddings_por_modelo = cargar_embeddings_desde_disco()

    # Consulta de ejemplo (puede ser cualquier frase en español)
    CONSULTA = "El coronel recordó su pasado mientras enfrentaba la muerte"

    resultados = recuperar_top_k(CONSULTA, chunks, embeddings_por_modelo, k=5)
    mostrar_comparacion(resultados, CONSULTA)
