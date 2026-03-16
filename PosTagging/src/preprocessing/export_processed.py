"""
Exporta los datasets procesados (tensores + vocabularios + max_len)
como archivos .pt para que las fases de entrenamiento los carguen
directamente sin reprocesar.

Uso:
    cd PosTagging/src/preprocessing
    python export_processed.py
"""

import os
import torch

from build_dataloaders import build_ancora_dataloaders
from prepare_conll_pipeline import prepare_conll_pipeline


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
    output_dir = "../../outputs/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Ancora
    print("Procesando Ancora...")
    ancora_path = "../../data/raw/ancora/ancora_corpus_pos.csv"
    ancora_data = build_ancora_dataloaders(ancora_path, batch_size=32)
    export_dataset(ancora_data, "ancora", output_dir)

    # CoNLL2002
    print("Procesando CoNLL2002...")
    conll_train = "../../data/raw/conll2002/train.txt"
    conll_valid = "../../data/raw/conll2002/valid.txt"
    conll_test  = "../../data/raw/conll2002/test.txt"
    conll_data = prepare_conll_pipeline(conll_train, conll_valid, conll_test, batch_size=32)
    export_dataset(conll_data, "conll", output_dir)

    print("\nDatasets exportados a outputs/processed/")


if __name__ == "__main__":
    main()
