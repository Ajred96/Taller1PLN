import pandas as pd


def load_ancora(path):
    """
    Carga y limpia el dataset Ancora.
    Conserva únicamente las columnas necesarias para POS tagging.
    """
    df = pd.read_csv(path)

    # Eliminar columnas innecesarias si existen
    columns_to_drop = [col for col in ["Unnamed: 0", "Tag"] if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Verificar columnas esperadas
    expected_columns = ["Sentence #", "Word", "POS"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Faltan columnas esperadas. Columnas actuales: {df.columns.tolist()}")

    # Reordenar por seguridad
    df = df[expected_columns]

    print("\nColumnas finales:")
    print(df.columns.tolist())

    print("\nPrimeras filas limpias:\n")
    print(df.head())

    print("\nNúmero total de filas:", len(df))
    print("Número de oraciones:", df["Sentence #"].nunique())
    print("Número de etiquetas POS únicas:", df["POS"].nunique())

    print("\nEtiquetas POS únicas:")
    print(sorted(df["POS"].unique()))

    return df


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"
    df = load_ancora(path)
