"""
Fase II — Entrenamiento BiLSTM Base

Ejecuta grid search de 24 combinaciones por dataset,
evalúa el mejor modelo en test, guarda resultados.

Uso:
    cd PosTagging/src/training/BiLSTMBase
    python run_phase2.py
"""

import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from model import BiLSTMTagger
from train import evaluate_model
from grid_search import run_grid_search
from predict import print_prediction


# ── Device ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── POSDataset ──────────────────────────────────────────
class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.long)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.masks = masks if isinstance(masks, torch.Tensor) else torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
            "attention_mask": self.masks[idx]
        }


# ── Cargar datos procesados ────────────────────────────
processed_dir = os.path.join("..", "..", "..", "outputs", "processed")
models_dir = os.path.join("..", "..", "..", "outputs", "models")
os.makedirs(models_dir, exist_ok=True)

print("\nCargando datos procesados...")

ancora = torch.load(os.path.join(processed_dir, "ancora_data.pt"), weights_only=False)
word2idx_a = ancora["word2idx"]
tag2idx_a  = ancora["tag2idx"]
max_len_a  = ancora["max_len"]
train_ds_a = POSDataset(ancora["train_inputs"], ancora["train_labels"], ancora["train_masks"])
val_ds_a   = POSDataset(ancora["val_inputs"],   ancora["val_labels"],   ancora["val_masks"])
test_ds_a  = POSDataset(ancora["test_inputs"],  ancora["test_labels"],  ancora["test_masks"])
print(f"  Ancora: vocab={len(word2idx_a)}, tags={len(tag2idx_a)}, max_len={max_len_a}")

conll = torch.load(os.path.join(processed_dir, "conll_data.pt"), weights_only=False)
word2idx_c = conll["word2idx"]
tag2idx_c  = conll["tag2idx"]
max_len_c  = conll["max_len"]
train_ds_c = POSDataset(conll["train_inputs"], conll["train_labels"], conll["train_masks"])
val_ds_c   = POSDataset(conll["val_inputs"],   conll["val_labels"],   conll["val_masks"])
test_ds_c  = POSDataset(conll["test_inputs"],  conll["test_labels"],  conll["test_masks"])
print(f"  CoNLL2002: vocab={len(word2idx_c)}, tags={len(tag2idx_c)}, max_len={max_len_c}")


# ── Grid Search Ancora ──────────────────────────────────
print("\n" + "=" * 60)
print("GRID SEARCH — ANCORA (24 combinaciones)")
print("=" * 60)

best_model_a, best_config_a, results_a = run_grid_search(
    train_ds_a, val_ds_a,
    vocab_size=len(word2idx_a),
    tagset_size=len(tag2idx_a),
    dataset_name="Ancora",
    device=device
)

test_loader_a = DataLoader(test_ds_a, batch_size=64, shuffle=False)
eval_a = evaluate_model(best_model_a, test_loader_a, tag2idx_a, device)

print(f"\n{'='*60}")
print("RESULTADOS TEST — ANCORA")
print(f"{'='*60}")
print(f"Config: {best_config_a}")
print(f"\nAccuracy:    {eval_a['accuracy']:.4f}")
print(f"Macro F1:    {eval_a['macro_f1']:.4f}")
print(f"Weighted F1: {eval_a['weighted_f1']:.4f}")
print(f"\nClassification Report:\n{eval_a['report']}")

torch.save({
    "model_state_dict": best_model_a.state_dict(),
    "best_config": best_config_a,
    "eval_results": eval_a,
    "grid_results": results_a,
    "vocab_size": len(word2idx_a),
    "tagset_size": len(tag2idx_a),
}, os.path.join(models_dir, "bilstm_ancora.pt"))
print("Modelo Ancora guardado.")


# ── Grid Search CoNLL2002 ───────────────────────────────
print("\n" + "=" * 60)
print("GRID SEARCH — CoNLL2002 (24 combinaciones)")
print("=" * 60)

best_model_c, best_config_c, results_c = run_grid_search(
    train_ds_c, val_ds_c,
    vocab_size=len(word2idx_c),
    tagset_size=len(tag2idx_c),
    dataset_name="CoNLL2002",
    device=device
)

test_loader_c = DataLoader(test_ds_c, batch_size=64, shuffle=False)
eval_c = evaluate_model(best_model_c, test_loader_c, tag2idx_c, device)

print(f"\n{'='*60}")
print("RESULTADOS TEST — CoNLL2002")
print(f"{'='*60}")
print(f"Config: {best_config_c}")
print(f"\nAccuracy:    {eval_c['accuracy']:.4f}")
print(f"Macro F1:    {eval_c['macro_f1']:.4f}")
print(f"Weighted F1: {eval_c['weighted_f1']:.4f}")
print(f"\nClassification Report:\n{eval_c['report']}")

torch.save({
    "model_state_dict": best_model_c.state_dict(),
    "best_config": best_config_c,
    "eval_results": eval_c,
    "grid_results": results_c,
    "vocab_size": len(word2idx_c),
    "tagset_size": len(tag2idx_c),
}, os.path.join(models_dir, "bilstm_conll.pt"))
print("Modelo CoNLL2002 guardado.")


# ── Tabla de resultados ─────────────────────────────────
print("\n" + "=" * 80)
print("TABLA DE RESULTADOS — BiLSTM Base")
print("=" * 80)

results_table = pd.DataFrame([
    {
        "Modelo": "BiLSTM",
        "Dataset": "Ancora",
        "Accuracy": f"{eval_a['accuracy']:.4f}",
        "Macro F1": f"{eval_a['macro_f1']:.4f}",
        "Weighted F1": f"{eval_a['weighted_f1']:.4f}",
        "Mejor Config": f"bs={best_config_a['batch_size']}, opt={best_config_a['optimizer']}, emb={best_config_a['embedding_dim']}, hid={best_config_a['hidden_dim']}"
    },
    {
        "Modelo": "BiLSTM",
        "Dataset": "CoNLL2002",
        "Accuracy": f"{eval_c['accuracy']:.4f}",
        "Macro F1": f"{eval_c['macro_f1']:.4f}",
        "Weighted F1": f"{eval_c['weighted_f1']:.4f}",
        "Mejor Config": f"bs={best_config_c['batch_size']}, opt={best_config_c['optimizer']}, emb={best_config_c['embedding_dim']}, hid={best_config_c['hidden_dim']}"
    }
])
print(results_table.to_string(index=False))

print(f"\n\nDETALLE GRID SEARCH — Ancora (ordenado por val_loss):")
df_a = pd.DataFrame(results_a).sort_values("best_val_loss")
print(df_a.to_string(index=False))

print(f"\n\nDETALLE GRID SEARCH — CoNLL2002 (ordenado por val_loss):")
df_c = pd.DataFrame(results_c).sort_values("best_val_loss")
print(df_c.to_string(index=False))


# ── Predicción de ejemplo ───────────────────────────────
print("\n" + "=" * 60)
print("PREDICCIÓN DE EJEMPLO")
print("=" * 60)

test_sentence = "El hombre bajo toca el bajo bajo la escalera"
print(f'Oración: "{test_sentence}"')

print_prediction(test_sentence, best_model_a, word2idx_a, tag2idx_a, device, "BiLSTM — Ancora")
print_prediction(test_sentence, best_model_c, word2idx_c, tag2idx_c, device, "BiLSTM — CoNLL2002")

# Predicción interactiva
print("\n" + "=" * 60)
mi_oracion = input("Escribe una oración en español: ")
print(f'\nOración: "{mi_oracion}"')
print_prediction(mi_oracion, best_model_a, word2idx_a, tag2idx_a, device, "BiLSTM — Ancora")
print_prediction(mi_oracion, best_model_c, word2idx_c, tag2idx_c, device, "BiLSTM — CoNLL2002")
