"""
visualization.py
====================================
Genera gráficas PCA y t-SNE para cada modelo de embeddings.

Para cada modelo se visualiza:
    - La oración de consulta (en rojo)
    - El fragmento más similar recuperado (en azul)
    - Una muestra de chunks de fondo para dar contexto (en gris)

Las gráficas se guardan en:
    srcTaller2/outputs/embeddings/visualizations/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─── Configuración ────────────────────────────────────────────────────────────

VISUALIZATIONS_DIR = Path("srcTaller2/outputs/embeddings/visualizations")
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

NOMBRES_MODELOS = {
    "e5":     "multilingual-e5-base",
    "mpnet":  "paraphrase-multilingual-mpnet-base-v2",
    "bge":    "BAAI/bge-m3",
    "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
}

# Número de chunks de fondo para dar contexto visual (no todos los 6796)
N_FONDO = 500


# ─── Funciones ────────────────────────────────────────────────────────────────

def reducir_pca(matriz: np.ndarray, n_componentes: int = 2) -> np.ndarray:
    """Reduce dimensionalidad con PCA."""
    pca = PCA(n_components=n_componentes, random_state=42)
    return pca.fit_transform(matriz)


def reducir_tsne(matriz: np.ndarray, n_componentes: int = 2) -> np.ndarray:
    """
    Reduce dimensionalidad con t-SNE.
    Perplexity se ajusta automáticamente si hay pocos puntos.
    """
    perplexity = min(30, max(5, len(matriz) // 5))
    tsne = TSNE(
        n_components=n_componentes,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(matriz)


def _graficar(
    coords_fondo:   np.ndarray,
    coord_consulta: np.ndarray,
    coord_similar:  np.ndarray,
    titulo:         str,
    nombre_modelo:  str,
    tecnica:        str,
    texto_consulta: str,
    texto_similar:  str,
    ruta_salida:    Path,
) -> None:
    """
    Dibuja una gráfica 2D con:
        - puntos de fondo en gris (muestra de chunks)
        - consulta en rojo
        - fragmento más similar en azul
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Fondo: muestra de chunks
    ax.scatter(
        coords_fondo[:, 0], coords_fondo[:, 1],
        c="lightgray", alpha=0.4, s=15, label="Chunks (muestra)"
    )

    # Fragmento más similar
    ax.scatter(
        coord_similar[0], coord_similar[1],
        c="steelblue", s=120, zorder=5, label="Fragmento más similar"
    )
    ax.annotate(
        f"  Similar\n  ({texto_similar[:60]}...)",
        xy=(coord_similar[0], coord_similar[1]),
        fontsize=7, color="steelblue",
        xytext=(8, 4), textcoords="offset points",
    )

    # Consulta
    ax.scatter(
        coord_consulta[0], coord_consulta[1],
        c="crimson", s=160, zorder=6, marker="*", label="Consulta"
    )
    ax.annotate(
        f"  Consulta\n  ({texto_consulta[:60]}...)",
        xy=(coord_consulta[0], coord_consulta[1]),
        fontsize=7, color="crimson",
        xytext=(8, 4), textcoords="offset points",
    )

    ax.set_title(f"{titulo}\nModelo: {nombre_modelo}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{tecnica} — Dimensión 1")
    ax.set_ylabel(f"{tecnica} — Dimensión 2")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✔ Gráfica guardada: {ruta_salida}")


def visualizar_modelo(
    id_modelo:         str,
    embeddings:        np.ndarray,
    idx_similar:       int,
    vec_consulta:      np.ndarray,
    texto_consulta:    str,
    texto_similar:     str,
    n_fondo:           int = N_FONDO,
) -> None:
    """
    Genera las gráficas PCA y t-SNE para un modelo dado.

    Proceso:
        1. Toma una muestra aleatoria de n_fondo chunks como fondo
        2. Agrega la consulta y el fragmento más similar a la matriz
        3. Reduce con PCA y t-SNE
        4. Grafica y guarda

    Args:
        id_modelo:      clave corta del modelo ("e5", "mpnet", etc.)
        embeddings:     matriz de embeddings de todos los chunks (n_chunks, dims)
        idx_similar:    índice del chunk más similar en la matriz
        vec_consulta:   embedding de la consulta (dims,)
        texto_consulta: texto de la consulta (para la etiqueta)
        texto_similar:  texto del chunk más similar (para la etiqueta)
        n_fondo:        cuántos chunks de fondo incluir en la visualización
    """
    nombre_modelo = NOMBRES_MODELOS.get(id_modelo, id_modelo)
    print(f"\n── Visualizando modelo: {id_modelo}")

    # Seleccionar muestra de fondo (excluyendo el chunk más similar para no duplicar)
    indices_disponibles = [i for i in range(len(embeddings)) if i != idx_similar]
    rng = np.random.default_rng(42)
    indices_fondo = rng.choice(indices_disponibles, size=min(n_fondo, len(indices_disponibles)), replace=False)

    # Construir matriz combinada: [fondo | similar | consulta]
    matriz_fondo    = embeddings[indices_fondo]          # (n_fondo, dims)
    vec_similar     = embeddings[idx_similar:idx_similar+1]  # (1, dims)
    vec_consulta_2d = vec_consulta.reshape(1, -1)        # (1, dims)

    matriz_total = np.vstack([matriz_fondo, vec_similar, vec_consulta_2d])
    # Índices dentro de matriz_total:
    idx_similar_local  = len(matriz_fondo)
    idx_consulta_local = len(matriz_fondo) + 1

    for tecnica, fn_reduccion in [("PCA", reducir_pca), ("t-SNE", reducir_tsne)]:
        print(f"   Aplicando {tecnica}...")
        coords = fn_reduccion(matriz_total)

        coords_fondo     = coords[:len(matriz_fondo)]
        coord_similar    = coords[idx_similar_local]
        coord_consulta   = coords[idx_consulta_local]

        ruta_salida = VISUALIZATIONS_DIR / f"{tecnica.lower()}_{id_modelo}.png"

        _graficar(
            coords_fondo   = coords_fondo,
            coord_consulta = coord_consulta,
            coord_similar  = coord_similar,
            titulo         = f"Recuperación semántica con {tecnica}",
            nombre_modelo  = nombre_modelo,
            tecnica        = tecnica,
            texto_consulta = texto_consulta,
            texto_similar  = texto_similar,
            ruta_salida    = ruta_salida,
        )


def visualizar_todos(
    embeddings_por_modelo: dict[str, np.ndarray],
    resultados_busqueda:   dict[str, list[dict]],
    vecs_consulta:         dict[str, np.ndarray],
    texto_consulta:        str,
    chunks:                list[dict],
    n_fondo:               int = N_FONDO,
) -> None:
    """
    Itera sobre los 4 modelos y genera PCA + t-SNE para cada uno.

    Args:
        embeddings_por_modelo: dict {id_modelo: np.ndarray}
        resultados_busqueda:   salida de recuperar_top_k()
        vecs_consulta:         dict {id_modelo: np.ndarray} con el embedding de la consulta
        texto_consulta:        texto original de la consulta
        chunks:                lista completa de chunks con metadata
        n_fondo:               chunks de fondo a mostrar
    """
    for id_modelo, embeddings in embeddings_por_modelo.items():
        if id_modelo not in resultados_busqueda:
            continue

        mejor = resultados_busqueda[id_modelo][0]

        # Encontrar el índice global del chunk más similar en la lista original
        idx_similar = next(
            i for i, c in enumerate(chunks)
            if c["chunk_id"] == mejor["chunk_id"] and c["filename"] == mejor["filename"]
        )

        visualizar_modelo(
            id_modelo      = id_modelo,
            embeddings     = embeddings,
            idx_similar    = idx_similar,
            vec_consulta   = vecs_consulta[id_modelo],
            texto_consulta = texto_consulta,
            texto_similar  = mejor["text"],
            n_fondo        = n_fondo,
        )

    print("\n✔ Todas las visualizaciones generadas.")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from sentence_transformers_embeddings import cargar_chunks, cargar_embeddings_desde_disco, CHUNKS_PATH
    from semantic_search import recuperar_top_k, encodear_consulta, MODELOS

    chunks, textos = cargar_chunks(CHUNKS_PATH)
    embeddings_por_modelo = cargar_embeddings_desde_disco()

    CONSULTA = "El coronel recordó su pasado mientras enfrentaba la muerte"

    resultados = recuperar_top_k(CONSULTA, chunks, embeddings_por_modelo, k=5)

    # Embeddings de la consulta por modelo (necesarios para la visualización)
    vecs_consulta = {
        id_modelo: encodear_consulta(CONSULTA, nombre_modelo)
        for id_modelo, nombre_modelo in MODELOS.items()
        if id_modelo in embeddings_por_modelo
    }

    visualizar_todos(embeddings_por_modelo, resultados, vecs_consulta, CONSULTA, chunks)
