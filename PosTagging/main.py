"""
POS Tagging en Español — Menú principal

Uso:
    cd PosTagging
    python main.py

Opciones:
    1. Preprocesar datasets (Fase I)
    2. Entrenar BiLSTM Base (Fase II)
    3. Entrenar BiLSTM Deep (Fase II)
    4. Entrenar BiLSTM-CRF (Fase III)
    5. POS Tagging de oración (BiLSTM Base)
    6. POS Tagging de oración (BiLSTM Deep)
    7. POS Tagging de oración (BiLSTM-CRF)
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(BASE_DIR, "src", "preprocessing")
BILSTM_DIR = os.path.join(BASE_DIR, "src", "training", "BiLSTMBase")
BILSTM_DEEP_DIR = os.path.join(BASE_DIR, "src", "training", "BiLSTMDeep")
BILSTM_CRF_DIR = os.path.join(BASE_DIR, "src", "training", "BiLSTMCRF")
PROCESSED_DIR = os.path.join(BASE_DIR, "outputs", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


def _require_torchcrf():
    """
    BiLSTM-CRF depende del paquete PyPI 'pytorch-crf' (se importa como torchcrf).
    Debe estar instalado en el mismo intérprete que ejecuta main.py.
    """
    try:
        import torchcrf  # noqa: F401
    except ModuleNotFoundError:
        print("\nFalta la dependencia para BiLSTM-CRF: instala el paquete 'pytorch-crf'.")
        print(f"  {sys.executable} -m pip install pytorch-crf")
        return False
    return True


def check_processed():
    """Verifica si los datos procesados existen."""
    ancora = os.path.exists(os.path.join(PROCESSED_DIR, "ancora_data.pt"))
    conll = os.path.exists(os.path.join(PROCESSED_DIR, "conll_data.pt"))
    return ancora and conll


def check_models_base():
    """Checkpoints del modelo BiLSTM Base."""
    ancora = os.path.exists(os.path.join(MODELS_DIR, "bilstm_ancora.pt"))
    conll = os.path.exists(os.path.join(MODELS_DIR, "bilstm_conll.pt"))
    return ancora, conll


def check_models_deep():
    """Checkpoints del modelo BiLSTM Deep."""
    ancora = os.path.exists(os.path.join(MODELS_DIR, "bilstm_deep_ancora.pt"))
    conll = os.path.exists(os.path.join(MODELS_DIR, "bilstm_deep_conll.pt"))
    return ancora, conll


def check_models_crf():
    """Checkpoints del modelo BiLSTM-CRF."""
    ancora = os.path.exists(os.path.join(MODELS_DIR, "bilstm_crf_ancora.pt"))
    conll = os.path.exists(os.path.join(MODELS_DIR, "bilstm_crf_conll.pt"))
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


def run_training_deep():
    """Ejecuta Fase II — BiLSTM Deep"""
    if not check_processed():
        print("\nNo se encontraron datos procesados en outputs/processed/")
        print("Ejecuta primero la opción 1 (Preprocesar).")
        return

    print("\n" + "=" * 60)
    print("FASE II — Entrenamiento BiLSTM Deep")
    print("=" * 60)

    subprocess.run([sys.executable, "run_phase2.py"], cwd=BILSTM_DEEP_DIR, check=True)


def run_training_crf():
    """Ejecuta Fase III — BiLSTM-CRF"""
    if not check_processed():
        print("\nNo se encontraron datos procesados en outputs/processed/")
        print("Ejecuta primero la opción 1 (Preprocesar).")
        return

    if not _require_torchcrf():
        return

    print("\n" + "=" * 60)
    print("FASE III — Entrenamiento BiLSTM-CRF")
    print("=" * 60)

    subprocess.run([sys.executable, "run_phase3.py"], cwd=BILSTM_CRF_DIR, check=True)


def run_prediction(variant):
    """
    Carga modelos guardados y realiza POS Tagging.
    variant: 'base' | 'deep'
    """
    if variant == "base":
        ancora_exists, conll_exists = check_models_base()
        ckpt_ancora, ckpt_conll = "bilstm_ancora.pt", "bilstm_conll.pt"
        train_dir = BILSTM_DIR
        family_label = "BiLSTM Base"
    elif variant == "deep":
        ancora_exists, conll_exists = check_models_deep()
        ckpt_ancora, ckpt_conll = "bilstm_deep_ancora.pt", "bilstm_deep_conll.pt"
        train_dir = BILSTM_DEEP_DIR
        family_label = "BiLSTM Deep"
    else:
        raise ValueError("variant debe ser 'base' o 'deep'")

    if not ancora_exists and not conll_exists:
        print("\nNo se encontraron modelos entrenados en outputs/models/ para esta variante.")
        print("Ejecuta primero el entrenamiento correspondiente (opción 2 o 3).")
        return

    # Importar modelo y predict según la carpeta de entrenamiento.
    # Quitar módulos cacheados: Python reutilizaría `model`/`predict` del otro
    for mod in ("model", "predict"):
        sys.modules.pop(mod, None)
    for p in (BILSTM_DIR, BILSTM_DEEP_DIR, BILSTM_CRF_DIR):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, train_dir)

    if variant == "base":
        from model import BiLSTMTagger as TaggerClass
    else:
        from model import BiLSTMDeepTagger as TaggerClass
    from predict import print_prediction

    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {device}")

    if not check_processed():
        print("\nNo se encontraron datos procesados. Ejecuta primero la opción 1.")
        return

    models = {}

    def build_model(ckpt):
        cfg = ckpt["best_config"]
        if variant == "base":
            return TaggerClass(
                vocab_size=ckpt["vocab_size"],
                tagset_size=ckpt["tagset_size"],
                embedding_dim=cfg["embedding_dim"],
                hidden_dim=cfg["hidden_dim"],
            )
        return TaggerClass(
            vocab_size=ckpt["vocab_size"],
            tagset_size=ckpt["tagset_size"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dense_dim=cfg["dense_dim"],
        )

    if ancora_exists:
        ancora_data = torch.load(
            os.path.join(PROCESSED_DIR, "ancora_data.pt"), map_location=device, weights_only=False
        )
        ckpt_a = torch.load(
            os.path.join(MODELS_DIR, ckpt_ancora), map_location=device, weights_only=False
        )
        cfg_a = ckpt_a["best_config"]

        model_a = build_model(ckpt_a)
        model_a.load_state_dict(ckpt_a["model_state_dict"])
        model_a = model_a.to(device)

        models["Ancora"] = {
            "model": model_a,
            "word2idx": ancora_data["word2idx"],
            "tag2idx": ancora_data["tag2idx"],
            "max_len": ancora_data["max_len"],
            "config": cfg_a,
        }
        print(f"  Ancora cargado: {cfg_a}")
    else:
        print("  Ancora: modelo no encontrado, se omite.")

    if conll_exists:
        conll_data = torch.load(
            os.path.join(PROCESSED_DIR, "conll_data.pt"), map_location=device, weights_only=False
        )
        ckpt_c = torch.load(
            os.path.join(MODELS_DIR, ckpt_conll), map_location=device, weights_only=False
        )
        cfg_c = ckpt_c["best_config"]

        model_c = build_model(ckpt_c)
        model_c.load_state_dict(ckpt_c["model_state_dict"])
        model_c = model_c.to(device)

        models["CoNLL2002"] = {
            "model": model_c,
            "word2idx": conll_data["word2idx"],
            "tag2idx": conll_data["tag2idx"],
            "max_len": conll_data["max_len"],
            "config": cfg_c,
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
                device, f"{family_label} — {name}"
            )
        print()


def run_prediction_crf():
    """Carga modelos BiLSTM-CRF guardados y realiza POS Tagging."""
    ancora_exists, conll_exists = check_models_crf()

    if not ancora_exists and not conll_exists:
        print("\nNo se encontraron modelos CRF entrenados en outputs/models/")
        print("Ejecuta primero la opción 4 (Entrenar BiLSTM-CRF).")
        return

    if not _require_torchcrf():
        return

    import torch
    # Evitar reutilizar `model`/`predict` de BiLSTM Base/Deep (mismo nombre de módulo).
    for mod in ("model", "predict"):
        sys.modules.pop(mod, None)
    for p in (BILSTM_DIR, BILSTM_DEEP_DIR, BILSTM_CRF_DIR):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, BILSTM_CRF_DIR)
    from model import BiLSTMCRFTagger
    from predict import print_prediction as print_prediction_crf

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {device}")

    if not check_processed():
        print("\nNo se encontraron datos procesados. Ejecuta primero la opción 1.")
        return

    models = {}

    if ancora_exists:
        ancora_data = torch.load(os.path.join(PROCESSED_DIR, "ancora_data.pt"), map_location=device, weights_only=False)
        ckpt_a = torch.load(os.path.join(MODELS_DIR, "bilstm_crf_ancora.pt"), map_location=device, weights_only=False)
        cfg_a = ckpt_a["best_config"]

        model_a = BiLSTMCRFTagger(
            vocab_size=ckpt_a["vocab_size"],
            tagset_size=ckpt_a["tagset_size"],
            embedding_dim=cfg_a["embedding_dim"],
            hidden_dim=cfg_a["hidden_dim"]
        )
        model_a.load_state_dict(ckpt_a["model_state_dict"])
        model_a = model_a.to(device)

        models["Ancora"] = {
            "model":    model_a,
            "word2idx": ancora_data["word2idx"],
            "tag2idx":  ancora_data["tag2idx"],
        }
        print(f"  BiLSTM-CRF Ancora cargado: {cfg_a}")
    else:
        print("  Ancora: modelo CRF no encontrado, se omite.")

    if conll_exists:
        conll_data = torch.load(os.path.join(PROCESSED_DIR, "conll_data.pt"), map_location=device, weights_only=False)
        ckpt_c = torch.load(os.path.join(MODELS_DIR, "bilstm_crf_conll.pt"), map_location=device, weights_only=False)
        cfg_c = ckpt_c["best_config"]

        model_c = BiLSTMCRFTagger(
            vocab_size=ckpt_c["vocab_size"],
            tagset_size=ckpt_c["tagset_size"],
            embedding_dim=cfg_c["embedding_dim"],
            hidden_dim=cfg_c["hidden_dim"]
        )
        model_c.load_state_dict(ckpt_c["model_state_dict"])
        model_c = model_c.to(device)

        models["CoNLL2002"] = {
            "model":    model_c,
            "word2idx": conll_data["word2idx"],
            "tag2idx":  conll_data["tag2idx"],
        }
        print(f"  BiLSTM-CRF CoNLL2002 cargado: {cfg_c}")
    else:
        print("  CoNLL2002: modelo CRF no encontrado, se omite.")

    print("\nModelos listos. Escribe 'salir' para volver al menú.\n")

    while True:
        oracion = input("Oración: ").strip()
        if oracion.lower() in ("salir", "exit", "q", ""):
            break
        for name, m in models.items():
            print_prediction_crf(oracion, m["model"], m["word2idx"], m["tag2idx"], device, f"BiLSTM-CRF — {name}")
        print()


def main():
    while True:
        print("\n" + "=" * 60)
        print("POS TAGGING EN ESPAÑOL — MENÚ PRINCIPAL")
        print("=" * 60)
        print()
        print("  1. Preprocesar datasets (Fase I)")
        print("  2. Entrenar BiLSTM Base (Fase II)")
        print("  3. Entrenar BiLSTM Deep (Fase II)")
        print("  4. Entrenar BiLSTM-CRF (Fase III)")
        print("  5. POS Tagging de oración (BiLSTM Base)")
        print("  6. POS Tagging de oración (BiLSTM Deep)")
        print("  7. POS Tagging de oración (BiLSTM-CRF)")
        print("  0. Salir")
        print()

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            run_preprocessing()
        elif opcion == "2":
            run_training()
        elif opcion == "3":
            run_training_deep()
        elif opcion == "4":
            run_training_crf()
        elif opcion == "5":
            run_prediction("base")
        elif opcion == "6":
            run_prediction("deep")
        elif opcion == "7":
            run_prediction_crf()
        elif opcion == "0":
            print("\nHasta luego.")
            break
        else:
            print("\nOpción no válida.")


if __name__ == "__main__":
    main()
