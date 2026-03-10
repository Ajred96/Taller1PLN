from load_ancora import load_ancora
from build_sentences import build_sentences
from vocab import build_word_vocab, build_tag_vocab


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
    encoded_tags = []

    for tag_sequence in pos_tags:
        encoded_tag_sequence = [tag2idx[tag] for tag in tag_sequence]
        encoded_tags.append(encoded_tag_sequence)

    return encoded_tags


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    df = load_ancora(path)
    sentences, pos_tags = build_sentences(df)

    word2idx = build_word_vocab(sentences)
    tag2idx = build_tag_vocab(pos_tags)

    encoded_sentences = encode_sentences(sentences, word2idx)
    encoded_tags = encode_tags(pos_tags, tag2idx)

    print("\nPrimera oración original:")
    print(sentences[0])

    print("\nPrimera oración codificada:")
    print(encoded_sentences[0])

    print("\nPrimeros tags originales:")
    print(pos_tags[0])

    print("\nPrimeros tags codificados:")
    print(encoded_tags[0])

    print("\nVerificación de longitudes:")
    print("Palabras:", len(sentences[0]), "->", len(encoded_sentences[0]))
    print("Tags:", len(pos_tags[0]), "->", len(encoded_tags[0]))