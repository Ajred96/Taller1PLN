import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from srcTaller2.config import (
    CHUNKS_DIR,
    ensure_output_dirs,
)


def create_text_splitter(
        chunk_size=1000,
        chunk_overlap=200,
):
    """
    Crea un RecursiveCharacterTextSplitter de LangChain.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def chunk_document(
        document,
        chunk_size=1000,
        chunk_overlap=200,
):
    """
    Fragmenta un documento usando RecursiveCharacterTextSplitter.
    """

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_text(document["text"])

    chunked_docs = []

    for i, chunk in enumerate(chunks):
        chunked_docs.append({
            "filename": document["filename"],
            "chunk_id": i,
            "text": chunk,
            "chunk_size": len(chunk),
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size_config": chunk_size,
            "chunk_overlap_config": chunk_overlap,
        })

    return chunked_docs


def chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
):
    """
    Fragmenta todos los documentos PDF.
    """

    all_chunks = []

    for document in documents:
        print(f"Fragmentando con LangChain: {document['filename']}")

        chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(
        chunks,
        filename="pdf_chunks.json",
):
    """
    Guarda los chunks en JSON.
    """

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


def print_chunk_examples(
        chunks,
        n=3,
):
    """
    Muestra ejemplos de chunks generados.
    """

    print("\nEjemplos de chunks")
    print("=" * 50)

    for chunk in chunks[:n]:
        print(f"\nArchivo: {chunk['filename']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Tamaño: {chunk['chunk_size']}")
        print(f"Splitter: {chunk['splitter']}")
        print(chunk["text"][:500])
        print("-" * 50)


def run_chunking_pipeline(documents):
    """
    Pipeline completo de chunking.
    """

    chunks = chunk_documents(
        documents=documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    save_chunks(chunks)

    print(f"\nTotal chunks generados: {len(chunks)}")

    print_chunk_examples(chunks)

    return chunks