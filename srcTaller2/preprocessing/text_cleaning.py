import nltk
import re
import string
from nltk.corpus import stopwords

# ============================================================
# Descargar recursos NLTK
# ============================================================

try:
    STOPWORDS_ES = set(stopwords.words("spanish"))

except LookupError:

    nltk.download("stopwords")
    STOPWORDS_ES = set(stopwords.words("spanish"))


# ============================================================
# Limpieza básica
# ============================================================

def clean_token(token):
    """
    Limpia un token individual.
    """

    # minúsculas
    token = token.lower()

    # eliminar espacios
    token = token.strip()

    return token


def is_valid_token(token):
    """
    Verifica si el token debe conservarse.
    """

    # vacío
    if not token:
        return False

    # puntuación pura
    if token in string.punctuation:
        return False

    # números
    if token.isdigit():
        return False

    # stopwords
    if token in STOPWORDS_ES:
        return False

    # tokens vacíos tipo ''
    if token.strip() == "":
        return False

    return True


# ============================================================
# Limpieza de oraciones
# ============================================================

def clean_sentence(sentence):
    """
    Limpia una oración completa.

    Entrada:
        ["Hola", ",", "Mundo"]

    Salida:
        ["hola", "mundo"]
    """

    cleaned_tokens = []

    for token in sentence:

        token = clean_token(token)

        # quitar caracteres raros repetidos
        token = re.sub(r"\s+", "", token)

        if is_valid_token(token):
            cleaned_tokens.append(token)

    return cleaned_tokens


def clean_sentences(sentences):
    """
    Limpia lista de oraciones.
    """

    cleaned = []

    for sentence in sentences:

        cleaned_sentence = clean_sentence(sentence)

        # evitar oraciones vacías
        if cleaned_sentence:
            cleaned.append(cleaned_sentence)

    return cleaned


# ============================================================
# Reporte
# ============================================================

def print_cleaning_example(sentences, n=3):
    """
    Muestra ejemplos antes/después.
    """

    print("\nEjemplos de limpieza")
    print("=" * 50)

    for i, sentence in enumerate(sentences[:n]):
        cleaned = clean_sentence(sentence)

        print(f"\nOración {i + 1}")
        print("Original:")
        print(sentence)

        print("\nLimpia:")
        print(cleaned)


# ============================================================
# Spanish Billion Words / datasets streaming
# ============================================================

def load_text_dataset(dataset_name, split="train", streaming=False, trust_remote_code=True, ):
    """
    Carga un dataset de texto desde Hugging Face.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )

    return dataset


def preprocess_streaming_dataset(dataset, text_column="text", max_sentences=1000, ):
    """
    Preprocesa un dataset en streaming y lo convierte en list[list[str]]
    para Gensim, Word2Vec y FastText.
    """

    processed_sentences = []

    for i, sample in enumerate(dataset):

        if i >= max_sentences:
            break

        text = sample.get(text_column, "")

        if not isinstance(text, str):
            continue

        tokens = text.split()
        cleaned_tokens = clean_sentence(tokens)

        if cleaned_tokens:
            processed_sentences.append(cleaned_tokens)

        if (i + 1) % 100 == 0:
            print(f"Procesadas: {i + 1}")

    return processed_sentences


def print_preprocessed_examples(sentences, n=3):
    """
    Muestra ejemplos de oraciones ya limpias.
    """

    print("\nEjemplos preprocesados")
    print("=" * 50)

    for i, sentence in enumerate(sentences[:n]):
        print(f"\nOración {i + 1}:")
        print(sentence)
