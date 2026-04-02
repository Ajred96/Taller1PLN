import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Data paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ANCORA_PATH = os.path.join(DATA_DIR, "ancora", "ancora_corpus_pos.csv")
CONLL_TRAIN = os.path.join(DATA_DIR, "conll2002", "train.txt")
CONLL_VALID = os.path.join(DATA_DIR, "conll2002", "valid.txt")
CONLL_TEST = os.path.join(DATA_DIR, "conll2002", "test.txt")

# --- Output paths ---
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
ARTIFACTS_DIR = os.path.join(OUTPUTS_DIR, "artifacts")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PROCESSED_DIR = os.path.join(OUTPUTS_DIR, "processed")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")


def ensure_output_dirs():
    for d in [ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
