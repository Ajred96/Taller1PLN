from prepare_ancora_pipeline import prepare_ancora_pipeline


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


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    data = prepare_ancora_pipeline(path)

    pad_word = data["word2idx"]["<PAD>"]
    pad_tag = data["tag2idx"]["<PAD>"]

    # Usamos el max_len del train para todo el pipeline
    X_train_pad, train_masks, max_len = pad_sequences(data["X_train_enc"], pad_value=pad_word)
    y_train_pad, _, _ = pad_sequences(data["y_train_enc"], pad_value=pad_tag, max_len=max_len)

    X_val_pad, val_masks, _ = pad_sequences(data["X_val_enc"], pad_value=pad_word, max_len=max_len)
    y_val_pad, _, _ = pad_sequences(data["y_val_enc"], pad_value=pad_tag, max_len=max_len)

    X_test_pad, test_masks, _ = pad_sequences(data["X_test_enc"], pad_value=pad_word, max_len=max_len)
    y_test_pad, _, _ = pad_sequences(data["y_test_enc"], pad_value=pad_tag, max_len=max_len)

    print("\nLongitud máxima usada:", max_len)

    print("\nTamaño train padded:", len(X_train_pad), "x", len(X_train_pad[0]))
    print("Tamaño val padded:", len(X_val_pad), "x", len(X_val_pad[0]))
    print("Tamaño test padded:", len(X_test_pad), "x", len(X_test_pad[0]))

    print("\nPrimera oración train codificada sin padding:")
    print(data["X_train_enc"][0])

    print("\nPrimera oración train con padding:")
    print(X_train_pad[0])

    print("\nPrimera máscara train:")
    print(train_masks[0])

    print("\nPrimera secuencia de tags padded:")
    print(y_train_pad[0])

    print("\nChequeo de longitudes:")
    print(len(X_train_pad[0]), len(train_masks[0]), len(y_train_pad[0]))