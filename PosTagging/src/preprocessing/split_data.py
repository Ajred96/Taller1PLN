from sklearn.model_selection import train_test_split

from build_sentences import build_sentences
from load_ancora import load_ancora


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


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    df = load_ancora(path)
    sentences, pos_tags = build_sentences(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(sentences, pos_tags)

    print("\nNúmero total de oraciones:", len(sentences))

    print("\nTamaños de los conjuntos:")
    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    print("\nVerificación de alineación:")
    print("Train:", len(X_train), len(y_train))
    print("Validation:", len(X_val), len(y_val))
    print("Test:", len(X_test), len(y_test))

    print("\nEjemplo de oración en train:")
    print(X_train[0])
    print(y_train[0])
