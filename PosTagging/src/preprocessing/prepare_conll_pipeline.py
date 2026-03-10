from load_conll import load_conll_sentences
from dataset_utils import encode_sentences
from pad_sequences import pad_sequences
from torch_dataset import POSDataset

from torch.utils.data import DataLoader


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


def prepare_conll_pipeline(train_path, valid_path, test_path, batch_size=32):
    """
    Pipeline completo de CoNLL2002:
    1. cargar train/valid/test ya dados
    2. construir vocabularios solo con train
    3. codificar
    4. padding
    5. dataset
    6. dataloaders
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


if __name__ == "__main__":
    train_path = "../../data/raw/conll2002/train.txt"
    valid_path = "../../data/raw/conll2002/valid.txt"
    test_path = "../../data/raw/conll2002/test.txt"

    data = prepare_conll_pipeline(train_path, valid_path, test_path, batch_size=32)

    print("\nResumen CoNLL2002")
    print("=" * 40)
    print("Train oraciones:", len(data["X_train"]))
    print("Validation oraciones:", len(data["X_val"]))
    print("Test oraciones:", len(data["X_test"]))
    print("Tamaño vocabulario palabras:", len(data["word2idx"]))
    print("Tamaño vocabulario tags:", len(data["tag2idx"]))
    print("Max length:", data["max_len"])

    batch = next(iter(data["train_loader"]))

    print("\nShape input_ids:", batch["input_ids"].shape)
    print("Shape labels:", batch["labels"].shape)
    print("Shape attention_mask:", batch["attention_mask"].shape)

    print("\nPrimeras 15 etiquetas del vocabulario:")
    print(list(data["tag2idx"].items())[:15])