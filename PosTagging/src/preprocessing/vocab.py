from load_ancora import load_ancora
from build_sentences import build_sentences


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


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    df = load_ancora(path)
    sentences, pos_tags = build_sentences(df)

    word2idx = build_word_vocab(sentences)
    tag2idx = build_tag_vocab(pos_tags)

    print("\nTamaño del vocabulario de palabras:", len(word2idx))
    print("Tamaño del vocabulario POS:", len(tag2idx))

    print("\nPrimeras 20 palabras del vocabulario:")
    print(list(word2idx.items())[:20])

    print("\nVocabulario de etiquetas POS:")
    print(tag2idx)