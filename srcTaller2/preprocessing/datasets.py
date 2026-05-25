import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from srcTaller2.config import (
    ANCORA_PATH,
    CONLL_TRAIN,
    CONLL_VALID,
    CONLL_TEST,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)


# ============================================================
# CoNLL2002
# ============================================================

def load_conll_sentences(path):
    """
    Carga un archivo CoNLL2002 y retorna
    una lista de oraciones.
    """

    sentences = []
    current_sentence = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            # Fin de oración
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split()

            # palabra POS NER
            if len(parts) >= 1:
                word = parts[0]
                current_sentence.append(word)

    # última oración
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def load_conll2002():
    """
    Carga train/validation/test originales.
    """

    train = load_conll_sentences(CONLL_TRAIN)
    validation = load_conll_sentences(CONLL_VALID)
    test = load_conll_sentences(CONLL_TEST)

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


# ============================================================
# Ancora
# ============================================================

def load_ancora():
    """
    Carga Ancora y crea split 70/15/15.
    """

    df = pd.read_csv(ANCORA_PATH)

    expected_columns = ["Sentence #", "Word", "POS"]

    missing_columns = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en Ancora: {missing_columns}"
        )

    df = df[expected_columns]

    sentences = []

    grouped = df.groupby("Sentence #")

    for _, group in grouped:
        sentence = (
            group["Word"]
            .astype(str)
            .tolist()
        )

        sentences.append(sentence)

    # 70 train
    # 15 validation
    # 15 test

    train, temp = train_test_split(
        sentences,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


# ============================================================
# Utilidades
# ============================================================

def get_first_sentences(dataset, n=3):
    """
    Obtiene las primeras n oraciones
    de cada split.
    """

    return {
        split_name: sentences[:n]
        for split_name, sentences in dataset.items()
    }


def save_json(data, filename):
    ensure_output_dirs()

    path = os.path.join(PROCESSED_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Guardado: {path}")


def save_report(conll, ancora):
    ensure_output_dirs()

    report_path = os.path.join(
        REPORTS_DIR,
        "preprocessing_summary.txt"
    )

    lines = [
        "TALLER 2 - RESUMEN PREPROCESAMIENTO",
        "=" * 50,
        "",

        "CoNLL2002",
        f"Train: {len(conll['train'])}",
        f"Validation: {len(conll['validation'])}",
        f"Test: {len(conll['test'])}",
        "",

        "Ancora",
        f"Train: {len(ancora['train'])}",
        f"Validation: {len(ancora['validation'])}",
        f"Test: {len(ancora['test'])}",
        "",

        "Archivos generados:",
        "- conll2002_splits.json",
        "- ancora_splits.json",
        "- conll2002_first_3.json",
        "- ancora_first_3.json",
    ]

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"Reporte guardado: {report_path}")


# ============================================================
# Pipeline principal
# ============================================================

def run_dataset_preprocessing():
    print("\nCargando CoNLL2002...")
    conll = load_conll2002()

    print("Cargando Ancora...")
    ancora = load_ancora()

    print("\nGuardando datasets completos...")
    save_json(conll, "conll2002_splits.json")
    save_json(ancora, "ancora_splits.json")

    print("\nGuardando primeras 3 oraciones...")
    save_json(
        get_first_sentences(conll),
        "conll2002_first_3.json"
    )

    save_json(
        get_first_sentences(ancora),
        "ancora_first_3.json"
    )

    print("\nGenerando reporte...")
    save_report(conll, ancora)

    print("\nPreprocesamiento completado correctamente.")
