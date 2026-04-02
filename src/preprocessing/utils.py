"""
Funciones compartidas de preprocesamiento.
Incluye: construccion de oraciones, vocabularios, codificacion,
particion de datos, padding y POSDataset de PyTorch.
"""

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# ── Construccion de oraciones ────────────────────────────────

def build_sentences(df):
    """
    Convierte el dataframe en listas de oraciones
    (palabras) y etiquetas POS.
    """

    sentences = []
    pos_tags = []

    grouped = df.groupby("Sentence #")

    for _, group in grouped:
        words = group["Word"].tolist()
        tags = group["POS"].tolist()

        sentences.append(words)
        pos_tags.append(tags)

    return sentences, pos_tags


# ── Vocabularios ─────────────────────────────────────────────

def build_word_vocab(sentences):
    """
    Construye el vocabulario de palabras.
    Incluye tokens especiales para padding y desconocidos.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def build_tag_vocab(pos_tags):
    """
    Construye el vocabulario de etiquetas POS.
    Incluye token especial para padding.
    """
    tag_vocab = {"<PAD>": 0}

    for tag_sequence in pos_tags:
        for tag in tag_sequence:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return tag_vocab


# ── Codificacion ─────────────────────────────────────────────

def encode_sentences(sentences, word2idx):
    """
    Convierte cada oración en una lista de índices.
    Las palabras desconocidas se mapean a <UNK>.
    """
    encoded_sentences = []

    for sentence in sentences:
        encoded_sentence = [word2idx.get(word, word2idx["<UNK>"]) for word in sentence]
        encoded_sentences.append(encoded_sentence)

    return encoded_sentences


def encode_tags(pos_tags, tag2idx):
    """
    Convierte cada secuencia de etiquetas POS en índices.
    """
    unk_tag = tag2idx.get("<UNK_TAG>", 0)
    encoded_tags = []

    for tag_sequence in pos_tags:
        encoded_tag_sequence = [tag2idx.get(tag, unk_tag) for tag in tag_sequence]
        encoded_tags.append(encoded_tag_sequence)

    return encoded_tags


# ── Particion de datos ───────────────────────────────────────

def split_dataset(sentences, pos_tags, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divide las secuencias en train, validation y test.
    """

    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("Las proporciones train/val/test deben sumar 1.")

    # Primer split: train vs temp(30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        sentences,
        pos_tags,
        test_size=(1 - train_size),
        random_state=random_state,
        shuffle=True
    )

    # Segundo split: temp(30%) -> val(15%) y test(15%)
    relative_test_size = test_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Padding ──────────────────────────────────────────────────

def pad_sequences(sequences, pad_value=0, max_len=None):
    """
    Aplica padding a una lista de secuencias.
    Devuelve:
    - secuencias padded
    - máscaras (1=token real, 0=padding)
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    masks = []

    for seq in sequences:
        seq_len = len(seq)
        padding_needed = max_len - seq_len

        padded_seq = seq + [pad_value] * padding_needed
        mask = [1] * seq_len + [0] * padding_needed

        padded_sequences.append(padded_seq)
        masks.append(mask)

    return padded_sequences, masks, max_len


# ── PyTorch Dataset ──────────────────────────────────────────

class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.long)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.masks = masks if isinstance(masks, torch.Tensor) else torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
            "attention_mask": self.masks[idx]
        }
