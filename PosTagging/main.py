"""
POS Tagging en Español — Menú principal

Uso:
    cd PosTagging
    python main.py

Opciones:
    1. Preprocesar datasets (Fase I)
    2. Entrenar BiLSTM Base (Fase II - BiLSTM Base)
    3. POS Tagging de oración (Modelos entrenados)
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(BASE_DIR, "src", "preprocessing")
BILSTM_DIR = os.path.join(BASE_DIR, "src", "training", "BiLSTMBase")
PROCESSED_DIR = os.path.join(BASE_DIR, "outputs", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


def check_processed():
    """Verifica si los datos procesados existen."""
    ancora = os.path.exists(os.path.join(PROCESSED_DIR, "ancora_data.pt"))
    conll = os.path.exists(os.path.join(PROCESSED_DIR, "conll_data.pt"))
    return ancora and conll


def check_models():
    """Verifica si los modelos entrenados existen."""
    ancora = os.path.exists(os.path.join(MODELS_DIR, "bilstm_ancora.pt"))
    conll = os.path.exists(os.path.join(MODELS_DIR, "bilstm_conll.pt"))
    return ancora, conll


def run_preprocessing():
    """Ejecuta Fase I"""
    print("\n" + "=" * 60)
    print("FASE I — Preprocesamiento")
    print("=" * 60)

    # Fase I original
    print("\n--- Ejecutando preprocesamiento ---")
    subprocess.run([sys.executable, "run_phase1_all.py"], cwd=PREPROCESSING_DIR, check=True)

    # Exportar .pt
    print("\n--- Exportando datos procesados (.pt) ---")
    subprocess.run([sys.executable, "export_processed.py"], cwd=PREPROCESSING_DIR, check=True)

    print("\nFase I completada. Datos listos en outputs/processed/")


def run_training():
    """Ejecuta Fase II — BiLSTM Base"""
    if not check_processed():
        print("\nNo se encontraron datos procesados en outputs/processed/")
        print("Ejecuta primero la opción 1 (Preprocesar).")
        return

    print("\n" + "=" * 60)
    print("FASE II — Entrenamiento BiLSTM Base")
    print("=" * 60)

    subprocess.run([sys.executable, "run_phase2.py"], cwd=BILSTM_DIR, check=True)


def run_prediction():
    """Carga modelos guardados y realiza POS Tagging de oraciones."""
    ancora_exists, conll_exists = check_models()

    if not ancora_exists and not conll_exists:
        print("\nNo se encontraron modelos entrenados en outputs/models/")
        print("Ejecuta primero la opción 2 (Entrenar).")
        return

    import torch
    sys.path.insert(0, BILSTM_DIR)
    from model import BiLSTMTagger
    from predict import print_prediction

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {device}")

    # Cargar datos procesados (para vocabs y max_len)
    if not check_processed():
        print("\nNo se encontraron datos procesados. Ejecuta primero la opción 1.")
        return

    models = {}

    if ancora_exists:
        ancora_data = torch.load(os.path.join(PROCESSED_DIR, "ancora_data.pt"), map_location=device, weights_only=False)
        ckpt_a = torch.load(os.path.join(MODELS_DIR, "bilstm_ancora.pt"), map_location=device, weights_only=False)
        cfg_a = ckpt_a["best_config"]

        model_a = BiLSTMTagger(
            vocab_size=ckpt_a["vocab_size"],
            tagset_size=ckpt_a["tagset_size"],
            embedding_dim=cfg_a["embedding_dim"],
            hidden_dim=cfg_a["hidden_dim"]
        )
        model_a.load_state_dict(ckpt_a["model_state_dict"])
        model_a = model_a.to(device)

        models["Ancora"] = {
            "model": model_a,
            "word2idx": ancora_data["word2idx"],
            "tag2idx": ancora_data["tag2idx"],
            "max_len": ancora_data["max_len"],
            "config": cfg_a
        }
        print(f"  Ancora cargado: {cfg_a}")
    else:
        print("  Ancora: modelo no encontrado, se omite.")

    if conll_exists:
        conll_data = torch.load(os.path.join(PROCESSED_DIR, "conll_data.pt"), map_location=device, weights_only=False)
        ckpt_c = torch.load(os.path.join(MODELS_DIR, "bilstm_conll.pt"), map_location=device, weights_only=False)
        cfg_c = ckpt_c["best_config"]

        model_c = BiLSTMTagger(
            vocab_size=ckpt_c["vocab_size"],
            tagset_size=ckpt_c["tagset_size"],
            embedding_dim=cfg_c["embedding_dim"],
            hidden_dim=cfg_c["hidden_dim"]
        )
        model_c.load_state_dict(ckpt_c["model_state_dict"])
        model_c = model_c.to(device)

        models["CoNLL2002"] = {
            "model": model_c,
            "word2idx": conll_data["word2idx"],
            "tag2idx": conll_data["tag2idx"],
            "max_len": conll_data["max_len"],
            "config": cfg_c
        }
        print(f"  CoNLL2002 cargado: {cfg_c}")
    else:
        print("  CoNLL2002: modelo no encontrado, se omite.")

    print("\nModelos listos. Escribe 'salir' para volver al menú.\n")

    while True:
        oracion = input("Oración: ").strip()
        if oracion.lower() in ("salir", "exit", "q", ""):
            break

        for name, m in models.items():
            print_prediction(
                oracion, m["model"], m["word2idx"], m["tag2idx"],
                device, f"BiLSTM — {name}"
            )
        print()


def main():
    while True:
        print("\n" + "=" * 60)
        print("POS TAGGING EN ESPAÑOL — MENÚ PRINCIPAL")
        print("=" * 60)
        print()
        print("  1. Preprocesar datasets (Fase I)")
        print("  2. Entrenar BiLSTM Base (Fase II)")
        print("  3. POS Tagging de oración (Modelos entrenados)")
        print("  0. Salir")
        print()

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            run_preprocessing()
        elif opcion == "2":
            run_training()
        elif opcion == "3":
            run_prediction()
        elif opcion == "0":
            print("\nHasta luego.")
            break
        else:
            print("\nOpción no válida.")


if __name__ == "__main__":
    main()
