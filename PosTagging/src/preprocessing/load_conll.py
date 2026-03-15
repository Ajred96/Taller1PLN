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


def inspect_loaded_conll(path):
    sentences, pos_tags = load_conll_sentences(path)

    print(f"\nArchivo: {path}")
    print("Número de oraciones:", len(sentences))

    if sentences:
        print("\nPrimera oración:")
        print(sentences[0])

        print("\nPOS de la primera oración:")
        print(pos_tags[0])

        print("\nLongitud primera oración:", len(sentences[0]))

        unique_tags = sorted(set(tag for seq in pos_tags for tag in seq))
        print("\nNúmero de etiquetas POS únicas:", len(unique_tags))
        print("Etiquetas POS únicas:")
        print(unique_tags)

    return sentences, pos_tags


if __name__ == "__main__":
    train_path = "../../data/raw/conll2002/train.txt"
    valid_path = "../../data/raw/conll2002/valid.txt"
    test_path = "../../data/raw/conll2002/test.txt"

    print("=== TRAIN ===")
    inspect_loaded_conll(train_path)

    print("\n=== VALID ===")
    inspect_loaded_conll(valid_path)

    print("\n=== TEST ===")
    inspect_loaded_conll(test_path)