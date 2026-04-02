"""
Pipeline completo de preprocesamiento para el dataset CoNLL2002.

Uso desde main.py:
    from src.preprocessing.conll import run_conll_pipeline
    data = run_conll_pipeline()  # retorna dict con DataLoaders, vocabs, etc.
"""

import json
import os
from torch.utils.data import DataLoader

from .utils import (
    encode_sentences,
    pad_sequences,
    POSDataset,
)


# ── Carga ────────────────────────────────────────────────────

def load_conll_sentences(path):
    """
    Carga un archivo CoNLL2002 y devuelve:
    - sentences: lista de oraciones, cada una como lista de palabras
    - pos_tags: lista de etiquetas POS por oración

    Formato esperado por línea:
    palabra POS NER

    Las oraciones están separadas por líneas vacías.
    """
    sentences = []
    pos_tags = []

    current_words = []
    current_pos = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Fin de oración
            if not line:
                if current_words:
                    sentences.append(current_words)
                    pos_tags.append(current_pos)
                    current_words = []
                    current_pos = []
                continue

            parts = line.split()

            # Esperamos al menos: word, POS, NER
            if len(parts) < 3:
                continue

            word = parts[0]
            pos = parts[1]

            current_words.append(word)
            current_pos.append(pos)

    # Por si el archivo no termina con línea vacía
    if current_words:
        sentences.append(current_words)
        pos_tags.append(current_pos)

    return sentences, pos_tags


# ── Vocabularios especificos CoNLL ───────────────────────────

def build_word_vocab_conll(sentences):
    """
    Vocabulario de palabras para CoNLL2002.
    Solo con train.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def build_tag_vocab_conll(pos_tags):
    """
    Vocabulario de etiquetas POS para CoNLL2002.
    Solo con train, pero con soporte para etiquetas desconocidas.
    """
    tag_vocab = {"<PAD>": 0, "<UNK_TAG>": 1}

    for tag_sequence in pos_tags:
        for tag in tag_sequence:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return tag_vocab


def encode_tags_conll(pos_tags, tag2idx):
    """
    Codifica etiquetas POS.
    Si aparece una etiqueta no vista en train, va a <UNK_TAG>.
    """
    unk_tag = tag2idx["<UNK_TAG>"]
    encoded_tags = []

    for tag_sequence in pos_tags:
        encoded = [tag2idx.get(tag, unk_tag) for tag in tag_sequence]
        encoded_tags.append(encoded)

    return encoded_tags


# ── Pipeline completo ────────────────────────────────────────

def run_conll_pipeline(train_path, valid_path, test_path, batch_size=32):
    """
    Pipeline completo de CoNLL2002:
    1. cargar train/valid/test ya dados
    2. construir vocabularios solo con train
    3. codificar
    4. padding
    5. dataset
    6. dataloaders

    Retorna dict con: train_loader, val_loader, test_loader,
                      word2idx, tag2idx, max_len,
                      X_train, X_val, X_test, y_train, y_val, y_test
    """

    # 1. Cargar
    X_train, y_train = load_conll_sentences(train_path)
    X_val, y_val = load_conll_sentences(valid_path)
    X_test, y_test = load_conll_sentences(test_path)

    # 2. Vocabularios solo con train
    word2idx = build_word_vocab_conll(X_train)
    tag2idx = build_tag_vocab_conll(y_train)

    # 3. Codificar
    X_train_enc = encode_sentences(X_train, word2idx)
    X_val_enc = encode_sentences(X_val, word2idx)
    X_test_enc = encode_sentences(X_test, word2idx)

    y_train_enc = encode_tags_conll(y_train, tag2idx)
    y_val_enc = encode_tags_conll(y_val, tag2idx)
    y_test_enc = encode_tags_conll(y_test, tag2idx)

    # 4. Padding usando max_len de train
    pad_word = word2idx["<PAD>"]
    pad_tag = tag2idx["<PAD>"]

    X_train_pad, train_masks, max_len = pad_sequences(X_train_enc, pad_value=pad_word)
    y_train_pad, _, _ = pad_sequences(y_train_enc, pad_value=pad_tag, max_len=max_len)

    X_val_pad, val_masks, _ = pad_sequences(X_val_enc, pad_value=pad_word, max_len=max_len)
    y_val_pad, _, _ = pad_sequences(y_val_enc, pad_value=pad_tag, max_len=max_len)

    X_test_pad, test_masks, _ = pad_sequences(X_test_enc, pad_value=pad_word, max_len=max_len)
    y_test_pad, _, _ = pad_sequences(y_test_enc, pad_value=pad_tag, max_len=max_len)

    # 5. Dataset
    train_dataset = POSDataset(X_train_pad, y_train_pad, train_masks)
    val_dataset = POSDataset(X_val_pad, y_val_pad, val_masks)
    test_dataset = POSDataset(X_test_pad, y_test_pad, test_masks)

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "word2idx": word2idx,
        "tag2idx": tag2idx,
        "max_len": max_len,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


# ── Guardar artefactos y reporte ─────────────────────────────

def save_conll_artifacts(data):
    """Guarda vocabularios JSON y reporte de resumen."""
    from config import ARTIFACTS_DIR, REPORTS_DIR, ensure_output_dirs
    ensure_output_dirs()

    # Vocabularios
    _save_json(data["word2idx"], os.path.join(ARTIFACTS_DIR, "conll_word2idx.json"))
    _save_json(data["tag2idx"], os.path.join(ARTIFACTS_DIR, "conll_tag2idx.json"))

    # Reporte
    summary = [
        "FASE I - RESUMEN CONLL2002",
        "=" * 40,
        "Dataset: CoNLL2002",
        f"Train sentences: {len(data['X_train'])}",
        f"Validation sentences: {len(data['X_val'])}",
        f"Test sentences: {len(data['X_test'])}",
        f"Word vocab size: {len(data['word2idx'])}",
        f"Tag vocab size: {len(data['tag2idx'])}",
        f"Max length: {data['max_len']}",
        "Batch size: 32",
        "",
        "Observacion importante:",
        "- CoNLL2002 tiene oraciones extremadamente largas.",
        f"- max_len observado en train: {data['max_len']}",
        "- Conviene revisar truncamiento o percentiles antes del entrenamiento.",
        "",
        "Artefactos generados:",
        "- outputs/artifacts/conll_word2idx.json",
        "- outputs/artifacts/conll_tag2idx.json",
        "",
        "Objetos disponibles para entrenamiento:",
        "- train_loader",
        "- val_loader",
        "- test_loader",
        "- word2idx",
        "- tag2idx",
        "- max_len",
    ]

    with open(os.path.join(REPORTS_DIR, "phase1_conll_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\nCoNLL2002 procesado correctamente.")
    print("Se guardaron vocabularios y resumen.")


def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
