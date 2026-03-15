from load_ancora import load_ancora
from build_sentences import build_sentences
from split_data import split_dataset
from vocab import build_word_vocab, build_tag_vocab
from dataset_utils import encode_sentences, encode_tags


def prepare_ancora_pipeline(path):
    """
    Pipeline completo de preprocesamiento para Ancora:
    1. Cargar
    2. Construir oraciones
    3. Dividir train/val/test
    4. Construir vocabularios SOLO con train
    5. Codificar train/val/test
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

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_enc": X_train_enc,
        "X_val_enc": X_val_enc,
        "X_test_enc": X_test_enc,
        "y_train_enc": y_train_enc,
        "y_val_enc": y_val_enc,
        "y_test_enc": y_test_enc,
        "word2idx": word2idx,
        "tag2idx": tag2idx,
    }


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    data = prepare_ancora_pipeline(path)

    print("\nTamaño vocabulario palabras (solo train):", len(data["word2idx"]))
    print("Tamaño vocabulario POS (solo train):", len(data["tag2idx"]))

    print("\nTamaños datasets codificados:")
    print("Train:", len(data["X_train_enc"]), len(data["y_train_enc"]))
    print("Validation:", len(data["X_val_enc"]), len(data["y_val_enc"]))
    print("Test:", len(data["X_test_enc"]), len(data["y_test_enc"]))

    print("\nEjemplo oración train original:")
    print(data["X_train"][0])

    print("\nEjemplo oración train codificada:")
    print(data["X_train_enc"][0])

    print("\nEjemplo tags train originales:")
    print(data["y_train"][0])

    print("\nEjemplo tags train codificados:")
    print(data["y_train_enc"][0])