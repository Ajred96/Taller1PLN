from load_ancora import load_ancora


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


if __name__ == "__main__":

    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    df = load_ancora(path)

    sentences, pos_tags = build_sentences(df)

    print("\nNúmero de oraciones:", len(sentences))

    print("\nPrimera oración:")
    print(sentences[0])

    print("\nPOS de la primera oración:")
    print(pos_tags[0])

    print("\nLongitud primera oración:", len(sentences[0]))