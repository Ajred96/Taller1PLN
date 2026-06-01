from srcTaller2.preprocessing.chunking import (
    run_chunking_pipeline,
)
from srcTaller2.preprocessing.datasets import (
    run_dataset_preprocessing,
)
from srcTaller2.preprocessing.pdf_loader import (
    load_all_pdfs,
    print_pdf_examples,
)
from srcTaller2.preprocessing.text_cleaning import (
    load_text_dataset,
    preprocess_streaming_dataset,
    print_preprocessed_examples,
)


def print_header():
    print("=" * 60)
    print("TALLER 2 - PREPROCESAMIENTO")
    print("Datasets, tokenización y embeddings")
    print("=" * 60)


def run_spanish_text_preprocessing():
    print("\nCargando Spanish Billion Words...")

    dataset = load_text_dataset(
        dataset_name="crscardellino/spanish_billion_words",
        streaming=True,
    )

    processed_sentences = preprocess_streaming_dataset(
        dataset,
        text_column="text",
        max_sentences=300,
    )

    print(f"\nTotal oraciones procesadas: {len(processed_sentences)}")
    print_preprocessed_examples(processed_sentences)


def run_pdf_loading():
    print("\nCargando PDFs...")

    documents = load_all_pdfs()

    print(f"\nPDFs cargados: {len(documents)}")

    print_pdf_examples(documents)

    print("\nIniciando chunking...")

    chunks = run_chunking_pipeline(documents)

    print(f"\nChunks generados: {len(chunks)}")


def main():
    print_header()

    try:
        run_dataset_preprocessing()
        run_spanish_text_preprocessing()
        run_pdf_loading()

    except Exception as error:
        print("\nError durante el preprocesamiento:")
        print(error)

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
