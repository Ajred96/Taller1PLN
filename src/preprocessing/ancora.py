"""
Pipeline completo de preprocesamiento para el dataset Ancora.

Uso desde main.py:
    from src.preprocessing.ancora import run_ancora_pipeline
    data = run_ancora_pipeline()  # retorna dict con DataLoaders, vocabs, etc.
"""

import json
import os
import pandas as pd
from torch.utils.data import DataLoader

from .utils import (
    build_sentences,
    build_word_vocab,
    build_tag_vocab,
    encode_sentences,
    encode_tags,
    split_dataset,
    pad_sequences,
    POSDataset,
)


# ── Carga ────────────────────────────────────────────────────

def load_ancora(path):
    """
    Carga y limpia el dataset Ancora.
    Conserva únicamente las columnas necesarias para POS tagging.
    """
    df = pd.read_csv(path)

    # Eliminar columnas innecesarias si existen
    columns_to_drop = [col for col in ["Unnamed: 0", "Tag"] if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Verificar columnas esperadas
    expected_columns = ["Sentence #", "Word", "POS"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Faltan columnas esperadas. Columnas actuales: {df.columns.tolist()}")

    # Reordenar por seguridad
    df = df[expected_columns]

    print("\nColumnas finales:")
    print(df.columns.tolist())

    print("\nPrimeras filas limpias:\n")
    print(df.head())

    print("\nNúmero total de filas:", len(df))
    print("Número de oraciones:", df["Sentence #"].nunique())
    print("Número de etiquetas POS únicas:", df["POS"].nunique())

    print("\nEtiquetas POS únicas:")
    print(sorted(df["POS"].unique()))

    return df


# ── Pipeline completo ────────────────────────────────────────

def run_ancora_pipeline(path, batch_size=32):
    """
    Pipeline completo de Ancora:
    1. Cargar y limpiar CSV
    2. Construir oraciones
    3. Split train/val/test (70/15/15)
    4. Vocabularios SOLO con train
    5. Codificar
    6. Padding
    7. Dataset + DataLoader

    Retorna dict con: train_loader, val_loader, test_loader,
                      word2idx, tag2idx, max_len
    """

    # 1. Cargar y construir secuencias
    df = load_ancora(path)
    sentences, pos_tags = build_sentences(df)

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(sentences, pos_tags)

    # 3. Vocabularios SOLO con train
    word2idx = build_word_vocab(X_train)
    tag2idx = build_tag_vocab(y_train)

    # 4. Codificación
    X_train_enc = encode_sentences(X_train, word2idx)
    X_val_enc = encode_sentences(X_val, word2idx)
    X_test_enc = encode_sentences(X_test, word2idx)

    y_train_enc = encode_tags(y_train, tag2idx)
    y_val_enc = encode_tags(y_val, tag2idx)
    y_test_enc = encode_tags(y_test, tag2idx)

    # 5. Padding usando max_len de train
    pad_word = word2idx["<PAD>"]
    pad_tag = tag2idx["<PAD>"]

    X_train_pad, train_masks, max_len = pad_sequences(X_train_enc, pad_value=pad_word)
    y_train_pad, _, _ = pad_sequences(y_train_enc, pad_value=pad_tag, max_len=max_len)

    X_val_pad, val_masks, _ = pad_sequences(X_val_enc, pad_value=pad_word, max_len=max_len)
    y_val_pad, _, _ = pad_sequences(y_val_enc, pad_value=pad_tag, max_len=max_len)

    X_test_pad, test_masks, _ = pad_sequences(X_test_enc, pad_value=pad_word, max_len=max_len)
    y_test_pad, _, _ = pad_sequences(y_test_enc, pad_value=pad_tag, max_len=max_len)

    # 6. Dataset + DataLoader
    train_dataset = POSDataset(X_train_pad, y_train_pad, train_masks)
    val_dataset = POSDataset(X_val_pad, y_val_pad, val_masks)
    test_dataset = POSDataset(X_test_pad, y_test_pad, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "word2idx": word2idx,
        "tag2idx": tag2idx,
        "max_len": max_len,
    }


# ── Guardar artefactos y reporte ─────────────────────────────

def save_ancora_artifacts(data):
    """Guarda vocabularios JSON y reporte de resumen."""
    from config import ARTIFACTS_DIR, REPORTS_DIR, ensure_output_dirs
    ensure_output_dirs()

    # Vocabularios
    _save_json(data["word2idx"], os.path.join(ARTIFACTS_DIR, "ancora_word2idx.json"))
    _save_json(data["tag2idx"], os.path.join(ARTIFACTS_DIR, "ancora_tag2idx.json"))

    # Reporte
    summary = [
        "FASE I - RESUMEN ANCORA",
        "=" * 40,
        "Dataset: Ancora",
        f"Word vocab size: {len(data['word2idx'])}",
        f"Tag vocab size: {len(data['tag2idx'])}",
        f"Max length: {data['max_len']}",
        "Batch size: 32",
        "",
        "Artefactos generados:",
        "- outputs/artifacts/ancora_word2idx.json",
        "- outputs/artifacts/ancora_tag2idx.json",
        "",
        "Objetos disponibles para entrenamiento:",
        "- train_loader",
        "- val_loader",
        "- test_loader",
        "- word2idx",
        "- tag2idx",
        "- max_len",
    ]

    with open(os.path.join(REPORTS_DIR, "phase1_ancora_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\nAncora procesado correctamente.")
    print("Se guardaron vocabularios y resumen.")


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
