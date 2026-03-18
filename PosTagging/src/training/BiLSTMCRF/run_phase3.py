# run_phase3.py

"""
Fase III — Entrenamiento BiLSTM-CRF

Ejecuta grid search de 24 combinaciones por dataset,
evalúa el mejor modelo en test, guarda resultados.

Uso:
    cd PosTagging/src/training/BiLSTMCRF
    python run_phase3.py
"""

import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

from model import BiLSTMCRFTagger
from train import evaluate_model
from grid_search import run_grid_search
from predict import print_prediction


# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — Ajusta estos valores según tus necesidades.
# Los valores por defecto reproducen el comportamiento completo
# sin ninguna modificación sobre los datos.
# ════════════════════════════════════════════════════════════════

# SMOKE_TEST
# Si es True, ignora el grid search completo y corre una sola
# combinación con parámetros mínimos (2 epochs, patience=1)
# para verificar que el pipeline funciona de principio a fin.
# Combínalo con TRAIN_SUBSET_* = 0.05 para que dure segundos.
# Ponlo en False para el entrenamiento real.
SMOKE_TEST = False

# MAX_LEN_CONLL
# Trunca las secuencias de CoNLL a esta longitud máxima.
# CoNLL tiene oraciones de hasta 1238 tokens, lo que hace
# que el algoritmo de Viterbi del CRF sea extremadamente lento.
# Por defecto es None: se usa el max_len original sin truncamiento.
# Ejemplo para acelerar: MAX_LEN_CONLL = 150
MAX_LEN_CONLL = None

# TRAIN_SUBSET_ANCORA (valor entre 0.0 y 1.0)
# Fracción del conjunto de entrenamiento de Ancora a usar.
# 1.0 = usar el 100% de los datos (comportamiento original).
# Ejemplo para smoke test o acelerar: TRAIN_SUBSET_ANCORA = 0.05
TRAIN_SUBSET_ANCORA = 1.0

# TRAIN_SUBSET_CONLL (valor entre 0.0 y 1.0)
# Fracción del conjunto de entrenamiento de CoNLL a usar.
# 1.0 = usar el 100% de los datos (comportamiento original).
# Ejemplo para smoke test o acelerar: TRAIN_SUBSET_CONLL = 0.05
TRAIN_SUBSET_CONLL = 1.0

# ════════════════════════════════════════════════════════════════


# ── Device ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── POSDataset ───────────────────────────────────────────────────
class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.long)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.masks  = masks  if isinstance(masks,  torch.Tensor) else torch.tensor(masks,  dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.inputs[idx],
            "labels":         self.labels[idx],
            "attention_mask": self.masks[idx]
        }


def apply_subset(dataset, fraction):
    """
    Devuelve un subconjunto aleatorio del dataset.
    Si fraction es 1.0 devuelve el dataset original sin cambios.
    """
    if fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    indices = torch.randperm(len(dataset))[:n].tolist()
    print(f"  Subset aplicado: {n}/{len(dataset)} ejemplos ({fraction*100:.0f}%)")
    return Subset(dataset, indices)


def truncate_tensors(inputs, labels, masks, max_len):
    """
    Recorta inputs, labels y masks a max_len columnas.
    Si max_len es None devuelve los tensores sin cambios.
    """
    if max_len is None:
        return inputs, labels, masks

    original_len = inputs.shape[1]
    if max_len >= original_len:
        print(f"  max_len={max_len} >= longitud original={original_len}, no se trunca.")
        return inputs, labels, masks

    inputs_t = inputs[:, :max_len]
    labels_t = labels[:, :max_len]
    masks_t = masks[:, :max_len]
    print(f"  Truncamiento aplicado: {original_len} → {max_len} tokens por secuencia.")
    return inputs_t, labels_t, masks_t


# ── Cargar datos procesados ──────────────────────────────────────
processed_dir = os.path.join("..", "..", "..", "outputs", "processed")
models_dir = os.path.join("..", "..", "..", "outputs", "models")
os.makedirs(models_dir, exist_ok=True)

print("\nCargando datos procesados...")

# ── Ancora ───────────────────────────────────────────────────────
ancora = torch.load(os.path.join(processed_dir, "ancora_data.pt"), weights_only=False)
word2idx_a = ancora["word2idx"]
tag2idx_a = ancora["tag2idx"]
max_len_a = ancora["max_len"]

train_ds_a = POSDataset(ancora["train_inputs"], ancora["train_labels"], ancora["train_masks"])
val_ds_a = POSDataset(ancora["val_inputs"],   ancora["val_labels"],   ancora["val_masks"])
test_ds_a = POSDataset(ancora["test_inputs"],  ancora["test_labels"],  ancora["test_masks"])

train_ds_a = apply_subset(train_ds_a, TRAIN_SUBSET_ANCORA)
print(f"  Ancora: vocab={len(word2idx_a)}, tags={len(tag2idx_a)}, max_len={max_len_a}")

# ── CoNLL2002 ────────────────────────────────────────────────────
conll      = torch.load(os.path.join(processed_dir, "conll_data.pt"), weights_only=False)
word2idx_c = conll["word2idx"]
tag2idx_c = conll["tag2idx"]
max_len_c = conll["max_len"]

# Truncamiento aplicado antes de construir los datasets
train_in_c, train_lb_c, train_mk_c = truncate_tensors(
    conll["train_inputs"], conll["train_labels"], conll["train_masks"], MAX_LEN_CONLL)
val_in_c,   val_lb_c,   val_mk_c = truncate_tensors(
    conll["val_inputs"],   conll["val_labels"],   conll["val_masks"],   MAX_LEN_CONLL)
test_in_c,  test_lb_c,  test_mk_c = truncate_tensors(
    conll["test_inputs"],  conll["test_labels"],  conll["test_masks"],  MAX_LEN_CONLL)

train_ds_c = POSDataset(train_in_c, train_lb_c, train_mk_c)
val_ds_c = POSDataset(val_in_c,   val_lb_c,   val_mk_c)
test_ds_c = POSDataset(test_in_c,  test_lb_c,  test_mk_c)

train_ds_c = apply_subset(train_ds_c, TRAIN_SUBSET_CONLL)
print(f"  CoNLL2002: vocab={len(word2idx_c)}, tags={len(tag2idx_c)}, max_len_original={max_len_c}")


# ── Grid Search Ancora ───────────────────────────────────────────
print("\n" + "=" * 60)
print("GRID SEARCH — ANCORA (24 combinaciones)")
print("=" * 60)

best_model_a, best_config_a, results_a = run_grid_search(
    train_ds_a, val_ds_a,
    vocab_size=len(word2idx_a),
    tagset_size=len(tag2idx_a),
    dataset_name="Ancora",
    device=device,
    smoke_test=SMOKE_TEST
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
    "best_config":      best_config_a,
    "eval_results":     eval_a,
    "grid_results":     results_a,
    "vocab_size":       len(word2idx_a),
    "tagset_size":      len(tag2idx_a),
}, os.path.join(models_dir, "bilstm_crf_ancora.pt"))
print("Modelo CRF Ancora guardado.")


# ── Grid Search CoNLL2002 ────────────────────────────────────────
print("\n" + "=" * 60)
print("GRID SEARCH — CoNLL2002 (24 combinaciones)")
print("=" * 60)

best_model_c, best_config_c, results_c = run_grid_search(
    train_ds_c, val_ds_c,
    vocab_size=len(word2idx_c),
    tagset_size=len(tag2idx_c),
    dataset_name="CoNLL2002",
    device=device,
    smoke_test=SMOKE_TEST
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
    "best_config":      best_config_c,
    "eval_results":     eval_c,
    "grid_results":     results_c,
    "vocab_size":       len(word2idx_c),
    "tagset_size":      len(tag2idx_c),
}, os.path.join(models_dir, "bilstm_crf_conll.pt"))
print("Modelo CRF CoNLL2002 guardado.")


# ── Tabla de resultados ──────────────────────────────────────────
print("\n" + "=" * 80)
print("TABLA DE RESULTADOS — BiLSTM-CRF")
print("=" * 80)

results_table = pd.DataFrame([
    {
        "Modelo":       "BiLSTM-CRF",
        "Dataset":      "Ancora",
        "Accuracy":     f"{eval_a['accuracy']:.4f}",
        "Macro F1":     f"{eval_a['macro_f1']:.4f}",
        "Weighted F1":  f"{eval_a['weighted_f1']:.4f}",
        "Mejor Config": f"bs={best_config_a['batch_size']}, opt={best_config_a['optimizer']}, emb={best_config_a['embedding_dim']}, hid={best_config_a['hidden_dim']}"
    },
    {
        "Modelo":       "BiLSTM-CRF",
        "Dataset":      "CoNLL2002",
        "Accuracy":     f"{eval_c['accuracy']:.4f}",
        "Macro F1":     f"{eval_c['macro_f1']:.4f}",
        "Weighted F1":  f"{eval_c['weighted_f1']:.4f}",
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


# ── Predicción de ejemplo ────────────────────────────────────────
print("\n" + "=" * 60)
print("PREDICCIÓN DE EJEMPLO")
print("=" * 60)

test_sentence = "El hombre bajo toca el bajo bajo la escalera"
print(f'Oración: "{test_sentence}"')

print_prediction(test_sentence, best_model_a, word2idx_a, tag2idx_a, device, "BiLSTM-CRF — Ancora")
print_prediction(test_sentence, best_model_c, word2idx_c, tag2idx_c, device, "BiLSTM-CRF — CoNLL2002")

# Predicción interactiva
print("\n" + "=" * 60)
mi_oracion = input("Escribe una oración en español: ")
print(f'\nOración: "{mi_oracion}"')
print_prediction(mi_oracion, best_model_a, word2idx_a, tag2idx_a, device, "BiLSTM-CRF — Ancora")
print_prediction(mi_oracion, best_model_c, word2idx_c, tag2idx_c, device, "BiLSTM-CRF — CoNLL2002")
