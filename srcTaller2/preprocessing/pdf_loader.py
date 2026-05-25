import os

import fitz  # pymupdf

from srcTaller2.config import PDF_DIR


# ============================================================
# Obtener PDFs
# ============================================================

def get_pdf_paths():

    pdf_files = []

    for filename in os.listdir(PDF_DIR):

        if filename.lower().endswith(".pdf"):

            full_path = os.path.join(PDF_DIR, filename)
            pdf_files.append(full_path)

    return pdf_files


# ============================================================
# Extraer texto PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    full_text = []

    for page in document:

        text = page.get_text()

        if text.strip():
            full_text.append(text)

    document.close()

    return "\n".join(full_text)


# ============================================================
# Cargar todos los PDFs
# ============================================================

def load_all_pdfs():

    pdf_paths = get_pdf_paths()

    documents = []

    for pdf_path in pdf_paths:

        print(f"Cargando PDF: {os.path.basename(pdf_path)}")

        text = extract_text_from_pdf(pdf_path)

        documents.append({
            "filename": os.path.basename(pdf_path),
            "text": text,
        })

    return documents


# ============================================================
# Ejemplos
# ============================================================

def print_pdf_examples(documents, n=2):

    print("\nEjemplos PDFs")
    print("=" * 50)

    for doc in documents[:n]:

        print(f"\nArchivo: {doc['filename']}")

        preview = doc["text"][:500]

        print(preview)
        print("\n" + "-" * 50)