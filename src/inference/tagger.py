"""
Modulo de inferencia: carga modelos entrenados y etiqueta oraciones.
"""

import os
import torch
from config import MODELS_DIR, PROCESSED_DIR

from src.training.bilstm_base.model import BiLSTMTagger
from src.training.bilstm_base.predict import predict_sentence as predict_base
from src.training.bilstm_deep.model import BiLSTMDeepTagger
from src.training.bilstm_deep.predict import predict_sentence as predict_deep
from src.training.bilstm_crf.model import BiLSTMCRFTagger
from src.training.bilstm_crf.predict import predict_sentence as predict_crf


# Mapa de modelos disponibles
MODEL_REGISTRY = {
    "bilstm_base": {
        "ancora": "bilstm_ancora.pt",
        "conll": "bilstm_conll.pt",
    },
    "bilstm_deep": {
        "ancora": "bilstm_deep_ancora.pt",
        "conll": "bilstm_deep_conll.pt",
    },
    "bilstm_crf": {
        "ancora": "bilstm_crf_ancora.pt",
        "conll": "bilstm_crf_conll.pt",
    },
}


def list_available_models():
    """Lista los modelos entrenados que existen en disco."""
    available = []
    for model_type, datasets in MODEL_REGISTRY.items():
        for dataset_name, filename in datasets.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                available.append((model_type, dataset_name, filename))
    return available


def load_model(model_type, dataset_name, device=None):
    """
    Carga un modelo entrenado desde disco.
    Retorna (model, word2idx, tag2idx, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = MODEL_REGISTRY[model_type][dataset_name]
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    vocab_size = checkpoint["vocab_size"]
    tagset_size = checkpoint["tagset_size"]
    cfg = checkpoint["best_config"]

    if model_type == "bilstm_base":
        model = BiLSTMTagger(vocab_size, tagset_size, cfg["embedding_dim"], cfg["hidden_dim"])
    elif model_type == "bilstm_deep":
        model = BiLSTMDeepTagger(vocab_size, tagset_size, cfg["embedding_dim"], cfg["hidden_dim"])
    elif model_type == "bilstm_crf":
        model = BiLSTMCRFTagger(vocab_size, tagset_size, cfg["embedding_dim"], cfg["hidden_dim"])
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Cargar vocabularios desde datos procesados
    data_file = "ancora_data.pt" if dataset_name == "ancora" else "conll_data.pt"
    data = torch.load(os.path.join(PROCESSED_DIR, data_file), map_location=device, weights_only=False)
    word2idx = data["word2idx"]
    tag2idx = data["tag2idx"]

    return model, word2idx, tag2idx, device


def tag_sentence(sentence, model_type, dataset_name, device=None):
    """
    Etiqueta una oracion usando un modelo entrenado.
    Retorna lista de tuplas (palabra, etiqueta).
    """
    model, word2idx, tag2idx, device = load_model(model_type, dataset_name, device)

    if model_type == "bilstm_crf":
        return predict_crf(sentence, model, word2idx, tag2idx, device)
    else:
        return predict_base(sentence, model, word2idx, tag2idx, device)


def interactive_tagging():
    """Sesion interactiva de etiquetado POS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        available = list_available_models()

        if not available:
            print("\nNo hay modelos entrenados disponibles.")
            print("Primero entrena un modelo desde el menu principal.")
            return

        print("\n" + "=" * 50)
        print("MODELOS DISPONIBLES")
        print("=" * 50)
        for i, (mtype, dset, fname) in enumerate(available, 1):
            print(f"  {i}. {mtype} ({dset})")
        print(f"  0. Volver al menu principal")

        try:
            choice = int(input("\nSelecciona un modelo: "))
        except (ValueError, EOFError, KeyboardInterrupt):
            return

        if choice == 0:
            return

        choice -= 1
        if choice < 0 or choice >= len(available):
            print("Opcion invalida.")
            continue

        model_type, dataset_name, _ = available[choice]

        print(f"\nCargando {model_type} ({dataset_name})...")
        model, word2idx, tag2idx, device = load_model(model_type, dataset_name, device)

        print("Modelo cargado. Escribe oraciones para etiquetar (vacio para cambiar modelo).\n")

        while True:
            try:
                sentence = input("Oracion: ").strip()
            except (EOFError, KeyboardInterrupt):
                return

            if not sentence:
                break

            if model_type == "bilstm_crf":
                results = predict_crf(sentence, model, word2idx, tag2idx, device)
            else:
                results = predict_base(sentence, model, word2idx, tag2idx, device)

            print()
            for word, tag in results:
                print(f"  {word:20s} -> {tag}")
            print()
