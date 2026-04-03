import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from model import BiLSTMCRFTagger
from train import evaluate_model
from grid_search import run_grid_search
from predict import print_prediction

# --- CONFIGURACIÓN ---
SMOKE_TEST = False # Cambiar a True para probar rápidamente sin esperar todo el Grid Search
# ---------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.long)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.masks  = masks  if isinstance(masks,  torch.Tensor) else torch.tensor(masks,  dtype=torch.long)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return {"input_ids": self.inputs[idx], "labels": self.labels[idx], "attention_mask": self.masks[idx]}

# -- Cargar Datos --
processed_dir = os.path.join("..", "..", "..", "outputs", "processed")
models_dir = os.path.join("..", "..", "..", "outputs", "models")
os.makedirs(models_dir, exist_ok=True)

print("Cargando Ancora...")
ancora = torch.load(os.path.join(processed_dir, "ancora_data.pt"), weights_only=False)
train_ds_a = POSDataset(ancora["train_inputs"], ancora["train_labels"], ancora["train_masks"])
val_ds_a = POSDataset(ancora["val_inputs"], ancora["val_labels"], ancora["val_masks"])
test_ds_a = POSDataset(ancora["test_inputs"], ancora["test_labels"], ancora["test_masks"])

print("Cargando CoNLL2002...")
conll = torch.load(os.path.join(processed_dir, "conll_data.pt"), weights_only=False)
train_ds_c = POSDataset(conll["train_inputs"], conll["train_labels"], conll["train_masks"])
val_ds_c = POSDataset(conll["val_inputs"], conll["val_labels"], conll["val_masks"])
test_ds_c = POSDataset(conll["test_inputs"], conll["test_labels"], conll["test_masks"])

# -- Ejecución Ancora --
print("\n" + "="*40 + "\nENTRENANDO ANCORA (BiLSTM-CRF)\n" + "="*40)
best_model_a, best_config_a, results_a = run_grid_search(
    train_ds_a, val_ds_a, len(ancora["word2idx"]), len(ancora["tag2idx"]), "Ancora", device, smoke_test=SMOKE_TEST
)
eval_a = evaluate_model(best_model_a, DataLoader(test_ds_a, batch_size=64), ancora["tag2idx"], device)
torch.save({"model_state_dict": best_model_a.state_dict(), "best_config": best_config_a}, os.path.join(models_dir, "bilstm_crf_ancora.pt"))

# -- Ejecución CoNLL --
print("\n" + "="*40 + "\nENTRENANDO CONLL (BiLSTM-CRF)\n" + "="*40)
best_model_c, best_config_c, results_c = run_grid_search(
    train_ds_c, val_ds_c, len(conll["word2idx"]), len(conll["tag2idx"]), "CoNLL2002", device, smoke_test=SMOKE_TEST
)
eval_c = evaluate_model(best_model_c, DataLoader(test_ds_c, batch_size=64), conll["tag2idx"], device)
torch.save({"model_state_dict": best_model_c.state_dict(), "best_config": best_config_c}, os.path.join(models_dir, "bilstm_crf_conll.pt"))

# -- Resumen --
print("\n" + "=" * 50 + "\nRESUMEN FINAL BiLSTM-CRF\n" + "=" * 50)
print(f"ANCORA -> Accuracy: {eval_a['accuracy']:.4f} | F1 Macro: {eval_a['macro_f1']:.4f}")
print(f"CONLL  -> Accuracy: {eval_c['accuracy']:.4f} | F1 Macro: {eval_c['macro_f1']:.4f}")

test_sentence = "El modelo CRF funciona muy bien."
print_prediction(test_sentence, best_model_a, ancora["word2idx"], ancora["tag2idx"], device, "Predicción Ancora")