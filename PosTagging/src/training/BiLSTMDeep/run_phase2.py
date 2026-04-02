"""
Fase II — Entrenamiento BiLSTM Deep (Modelo 2)


Uso:
    cd PosTagging/src/training/BiLSTMDeep
    python run_phase2.py
"""

import os
import sys
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Importaciones locales (Carpeta BiLSTMDeep)
from model import BiLSTMDeepTagger
from train import train_model, evaluate_model
from grid_search import run_grid_search_deep
from predict import print_prediction

# ── Configuración de Rutas Dinámicas ───────────────────────
# Detecta la raíz del proyecto basándose en la ubicación de este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

processed_dir = os.path.join(PROJECT_ROOT, "outputs", "processed")
models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(models_dir, exist_ok=True)

# ── Device ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Ejecutando en: {device}")

# ── POSDataset (Clase necesaria para cargar los .pt) ──────
class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.long)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        self.masks = masks if isinstance(masks, torch.Tensor) else torch.tensor(masks, dtype=torch.long)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx], "attention_mask": self.masks[idx]}

# ── Cargar datos ────────────────────────────────────────
print("\n📦 Cargando datos preprocesados...")
ancora = torch.load(os.path.join(processed_dir, "ancora_data.pt"), weights_only=False)
conll = torch.load(os.path.join(processed_dir, "conll_data.pt"), weights_only=False)

train_ds_a = POSDataset(ancora["train_inputs"], ancora["train_labels"], ancora["train_masks"])
val_ds_a   = POSDataset(ancora["val_inputs"],   ancora["val_labels"],   ancora["val_masks"])
test_ds_a  = POSDataset(ancora["test_inputs"],  ancora["test_labels"],  ancora["test_masks"])

train_ds_c = POSDataset(conll["train_inputs"], conll["train_labels"], conll["train_masks"])
val_ds_c   = POSDataset(conll["val_inputs"],   conll["val_labels"],   conll["val_masks"])
test_ds_c  = POSDataset(conll["test_inputs"],  conll["test_labels"],  conll["test_masks"])

# ── Función Auxiliar para Reconstruir Modelos ───────────
def reconstruct_best_model(results, vocab_size, tagset_size, train_ds, val_ds):
    best_cfg = min(results, key=lambda x: x['best_val_loss'])
    print(f"🔧 Reconstruyendo mejor modelo (Loss: {best_cfg['best_val_loss']})...")
    
    model = BiLSTMDeepTagger(
        vocab_size=vocab_size, tagset_size=tagset_size,
        embedding_dim=best_cfg['embedding_dim'], hidden_dim=best_cfg['hidden_dim'],
        dense_dim=best_cfg['dense_dim']
    ).to(device)
    
    t_loader = DataLoader(train_ds, batch_size=best_cfg['batch_size'], shuffle=True)
    v_loader = DataLoader(val_ds, batch_size=best_cfg['batch_size'], shuffle=False)
    
    opt = optim.Adam(model.parameters(), lr=0.001) if best_cfg['optimizer'] == "Adam" else \
          optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model, _, _, _ = train_model(model, t_loader, v_loader, opt, device, epochs=50, patience=5)
    return model, best_cfg

# ── PROCESAMIENTO ANCORA ────────────────────────────────
print("\n" + "="*60 + "\nGRID SEARCH — ANCORA (DEEP)\n" + "="*60)
best_model_a, best_config_a, results_a = run_grid_search_deep(
    train_ds_a, val_ds_a, len(ancora["word2idx"]), len(ancora["tag2idx"]), "Ancora", device
)

if best_model_a is None and results_a:
    best_model_a, best_config_a = reconstruct_best_model(
        results_a, len(ancora["word2idx"]), len(ancora["tag2idx"]), train_ds_a, val_ds_a
    )

eval_a = evaluate_model(best_model_a, DataLoader(test_ds_a, batch_size=64), ancora["tag2idx"], device)

torch.save({
    "model_state_dict": best_model_a.state_dict(),
    "best_config": best_config_a,
    "eval_results": eval_a,
    "vocab_size": len(ancora["word2idx"]),
    "tagset_size": len(ancora["tag2idx"]),
}, os.path.join(models_dir, "bilstm_deep_ancora.pt"))

# ── PROCESAMIENTO CONLL ─────────────────────────────────
print("\n" + "="*60 + "\nGRID SEARCH — CoNLL2002 (DEEP)\n" + "="*60)
best_model_c, best_config_c, results_c = run_grid_search_deep(
    train_ds_c, val_ds_c, len(conll["word2idx"]), len(conll["tag2idx"]), "CoNLL2002", device
)

if best_model_c is None and results_c:
    best_model_c, best_config_c = reconstruct_best_model(
        results_c, len(conll["word2idx"]), len(conll["tag2idx"]), train_ds_c, val_ds_c
    )

eval_c = evaluate_model(best_model_c, DataLoader(test_ds_c, batch_size=64), conll["tag2idx"], device)

torch.save({
    "model_state_dict": best_model_c.state_dict(),
    "best_config": best_config_c,
    "eval_results": eval_c,
    "vocab_size": len(conll["word2idx"]),
    "tagset_size": len(conll["tag2idx"]),
}, os.path.join(models_dir, "bilstm_deep_conll.pt"))

# ── RESULTADOS FINALES ──────────────────────────────────
print("\n" + "="*80 + "\nTABLA COMPARATIVA FINAL — BiLSTM DEEP\n" + "="*80)
final_df = pd.DataFrame([
    {"Dataset": "Ancora", "Acc": eval_a['accuracy'], "F1": eval_a['weighted_f1'], "Config": best_config_a},
    {"Dataset": "CoNLL", "Acc": eval_c['accuracy'], "F1": eval_c['weighted_f1'], "Config": best_config_c}
])
print(final_df.to_string(index=False))

# Ejemplo de predicción rápida
test_sent = "El hombre bajo toca el bajo bajo la escalera"
print(f"\n📝 Test: {test_sent}")
print_prediction(test_sent, best_model_a, ancora["word2idx"], ancora["tag2idx"], ancora["max_len"], device, "Deep — Ancora")