"""
main_embeddings.py
====================================
Orquesta el pipeline completo del módulo de embeddings semánticos:

    1. Carga los chunks del preprocesamiento (pdf_chunks.json)
    2. Genera embeddings con los 4 modelos de Sentence Transformers
    3. Realiza recuperación semántica con una consulta de ejemplo
    4. Genera visualizaciones PCA y t-SNE para cada modelo

Uso:
    python -m srcTaller2.embeddings.main_embeddings

    O con consulta personalizada:
    python -m srcTaller2.embeddings.main_embeddings --consulta "tu frase aquí"
"""

import argparse
from pathlib import Path

from sentence_transformers_embeddings import (
    cargar_chunks,
    generar_todos_los_embeddings,
    cargar_embeddings_desde_disco,
    CHUNKS_PATH,
    EMBEDDINGS_DIR,
)
from semantic_search import recuperar_top_k, mostrar_comparacion, encodear_consulta, MODELOS
from visualization import visualizar_todos

# ─── Consulta por defecto ─────────────────────────────────────────────────────

CONSULTA_DEFAULT = "El coronel recordó su pasado mientras enfrentaba la muerte"


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def ejecutar_pipeline(consulta: str, forzar_regenerar: bool = False) -> None:
    """
    Ejecuta el pipeline completo.

    Args:
        consulta:          texto de búsqueda semántica
        forzar_regenerar:  si True, regenera embeddings aunque existan en disco
    """
    print("=" * 65)
    print("  PIPELINE: EMBEDDINGS SEMÁNTICOS CON SENTENCE TRANSFORMERS")
    print("=" * 65)

    # ── Paso 1: Cargar chunks ─────────────────────────────────────────────────
    print("\n[1/4] Cargando chunks...")
    chunks, textos = cargar_chunks(CHUNKS_PATH)

    # ── Paso 2: Embeddings ───────────────────────────────────────────────────
    archivos_existentes = list(EMBEDDINGS_DIR.glob("embeddings_*.npy"))
    if archivos_existentes and not forzar_regenerar:
        print(f"\n[2/4] Embeddings encontrados en disco ({len(archivos_existentes)} modelos). Cargando...")
        embeddings_por_modelo = cargar_embeddings_desde_disco()
    else:
        print("\n[2/4] Generando embeddings (esto puede tardar varios minutos)...")
        embeddings_por_modelo = generar_todos_los_embeddings(textos)

    # ── Paso 3: Recuperación semántica ────────────────────────────────────────
    print(f"\n[3/4] Búsqueda semántica para: \"{consulta}\"")
    resultados = recuperar_top_k(consulta, chunks, embeddings_por_modelo, k=5)
    mostrar_comparacion(resultados, consulta)

    # Embeddings de la consulta (necesarios para la visualización)
    print("\n   Generando embeddings de la consulta para visualización...")
    vecs_consulta = {
        id_modelo: encodear_consulta(consulta, nombre_modelo)
        for id_modelo, nombre_modelo in MODELOS.items()
        if id_modelo in embeddings_por_modelo
    }

    # ── Paso 4: Visualización ─────────────────────────────────────────────────
    print("\n[4/4] Generando visualizaciones PCA y t-SNE...")
    visualizar_todos(
        embeddings_por_modelo = embeddings_por_modelo,
        resultados_busqueda   = resultados,
        vecs_consulta         = vecs_consulta,
        texto_consulta        = consulta,
        chunks                = chunks,
        n_fondo               = 500,
    )

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETADO")
    print(f"  Embeddings: {EMBEDDINGS_DIR}")
    print(f"  Gráficas:   {EMBEDDINGS_DIR / 'visualizations'}")
    print("=" * 65)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de embeddings semánticos")
    parser.add_argument(
        "--consulta",
        type=str,
        default=CONSULTA_DEFAULT,
        help="Oración de consulta para la búsqueda semántica",
    )
    parser.add_argument(
        "--regenerar",
        action="store_true",
        help="Fuerza la regeneración de embeddings aunque existan en disco",
    )
    args = parser.parse_args()

    ejecutar_pipeline(consulta=args.consulta, forzar_regenerar=args.regenerar)
