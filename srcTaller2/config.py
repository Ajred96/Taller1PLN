import os

# srcTaller2/config.py -> sube un nivel para llegar a la raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

ANCORA_PATH = os.path.join(DATA_DIR, "ancora", "ancora_corpus_pos.csv")
CONLL_TRAIN = os.path.join(DATA_DIR, "conll2002", "train.txt")
CONLL_VALID = os.path.join(DATA_DIR, "conll2002", "valid.txt")
CONLL_TEST = os.path.join(DATA_DIR, "conll2002", "test.txt")

PDF_DIR = os.path.join(DATA_DIR, "pdfs")

# --- Output paths ---
TALLER2_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUTS_DIR = os.path.join(TALLER2_DIR, "outputs")
PROCESSED_DIR = os.path.join(OUTPUTS_DIR, "processed")
CHUNKS_DIR = os.path.join(OUTPUTS_DIR, "chunks")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")


def ensure_output_dirs():
    for folder in [OUTPUTS_DIR, PROCESSED_DIR, CHUNKS_DIR, REPORTS_DIR]:
        os.makedirs(folder, exist_ok=True)
