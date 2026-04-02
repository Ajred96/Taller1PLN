"""
Exporta los datasets procesados (tensores + vocabularios + max_len)
como archivos .pt para que las fases de entrenamiento los carguen
directamente sin reprocesar.
"""

import os
import torch


def export_dataset(data, name, output_dir):
    """Exporta un dataset procesado a .pt"""
    payload = {
        "train_inputs": data["train_loader"].dataset.inputs,
        "train_labels": data["train_loader"].dataset.labels,
        "train_masks":  data["train_loader"].dataset.masks,
        "val_inputs":   data["val_loader"].dataset.inputs,
        "val_labels":   data["val_loader"].dataset.labels,
        "val_masks":    data["val_loader"].dataset.masks,
        "test_inputs":  data["test_loader"].dataset.inputs,
        "test_labels":  data["test_loader"].dataset.labels,
        "test_masks":   data["test_loader"].dataset.masks,
        "word2idx":     data["word2idx"],
        "tag2idx":      data["tag2idx"],
        "max_len":      data["max_len"],
    }

    path = os.path.join(output_dir, f"{name}_data.pt")
    torch.save(payload, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {name}_data.pt: {size_mb:.1f} MB")


def main():
    from config import PROCESSED_DIR, ANCORA_PATH, CONLL_TRAIN, CONLL_VALID, CONLL_TEST
    from .ancora import run_ancora_pipeline
    from .conll import run_conll_pipeline

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Ancora
    print("Procesando Ancora...")
    ancora_data = run_ancora_pipeline(ANCORA_PATH, batch_size=32)
    export_dataset(ancora_data, "ancora", PROCESSED_DIR)

    # CoNLL2002
    print("Procesando CoNLL2002...")
    conll_data = run_conll_pipeline(CONLL_TRAIN, CONLL_VALID, CONLL_TEST, batch_size=32)
    export_dataset(conll_data, "conll", PROCESSED_DIR)

    print("\nDatasets exportados a outputs/processed/")
