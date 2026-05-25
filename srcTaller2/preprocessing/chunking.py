import json
import os

from srcTaller2.config import (
    CHUNKS_DIR,
    ensure_output_dirs,
)


# ============================================================
# Splitter manual
# ============================================================

def split_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
):
    """
    Divide texto en chunks con overlap.
    """

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


# ============================================================
# Chunking documento
# ============================================================

def chunk_document(
        document,
        chunk_size=1000,
        chunk_overlap=200,
):
    chunks = split_text(
        document["text"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = []

    for i, chunk in enumerate(chunks):
        chunked_docs.append({
            "filename": document["filename"],
            "chunk_id": i,
            "text": chunk,
            "chunk_size": len(chunk),
        })

    return chunked_docs


# ============================================================
# Chunking múltiples documentos
# ============================================================

def chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
):
    all_chunks = []

    for document in documents:
        print(f"Fragmentando: {document['filename']}")

        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# Guardar chunks
# ============================================================

def save_chunks(
        chunks,
        filename="pdf_chunks.json",
):
    ensure_output_dirs()

    path = os.path.join(CHUNKS_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Chunks guardados: {path}")


# ============================================================
# Ejemplos
# ============================================================

def print_chunk_examples(
        chunks,
        n=3,
):
    print("\nEjemplos de chunks")
    print("=" * 50)

    for chunk in chunks[:n]:
        print(f"\nArchivo: {chunk['filename']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Tamaño: {chunk['chunk_size']}")

        print(chunk["text"][:500])

        print("-" * 50)


# ============================================================
# Pipeline
# ============================================================

def run_chunking_pipeline(documents):
    chunks = chunk_documents(documents)

    save_chunks(chunks)

    print(f"\nTotal chunks generados: {len(chunks)}")

    print_chunk_examples(chunks)

    return chunks
