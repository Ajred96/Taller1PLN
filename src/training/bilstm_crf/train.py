# train.py

import copy
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


def train_model(model, train_loader, val_loader, optimizer, device, epochs=50, patience=5):
    """
    Entrena el BiLSTM-CRF con Early Stopping monitoreando Val Loss.

    Diferencias respecto al modelo base:
    - El loss lo devuelve model.forward() directamente (log-likelihood negativo del CRF)
    - La máscara se castea a bool antes de pasarla al modelo
    - No se usa CrossEntropyLoss
    """
    model = model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # ── Entrenamiento ──────────────────────────────────────
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device).bool()  # cast a bool

            optimizer.zero_grad()
            loss = model(input_ids, labels, mask)  # forward() devuelve el loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train = total_train_loss / num_batches
        train_losses.append(avg_train)

        # ── Validación ─────────────────────────────────────────
        model.eval()
        total_val_loss = 0
        num_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device).bool()

                loss = model(input_ids, labels, mask)
                total_val_loss += loss.item()
                num_val += 1

        avg_val = total_val_loss / num_val
        val_losses.append(avg_val)

        # ── Early Stopping ──────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            marker = " *"
        else:
            patience_counter += 1
            marker = f" (patience {patience_counter}/{patience})"
            if patience_counter >= patience:
                print(f"  Epoch {epoch+1:02d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}")
                print(f"  Early stopping en epoch {epoch+1}")
                break

        print(f"  Epoch {epoch+1:02d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, best_val_loss


def evaluate_model(model, test_loader, tag2idx, device):
    """
    Evalúa en test usando decode() del CRF (Viterbi).

    Diferencia respecto al modelo base:
    - En lugar de argmax sobre logits, se llama a model.decode()
    - decode() devuelve listas de listas de Python, no tensores
    """
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device).bool()

            # decode() devuelve lista de listas (una por oración en el batch)
            # cada lista tiene exactamente len = número de tokens reales (sin padding)
            pred_sequences = model.decode(input_ids, mask)

            for i in range(len(pred_sequences)):
                # Las predicciones ya vienen sin padding gracias a la máscara
                pred_tags = pred_sequences[i]

                # Para las etiquetas reales, usamos la máscara para quitar padding
                true_tags = labels[i][mask[i]].cpu().numpy()

                all_preds.extend([idx2tag.get(p, "<UNK>") for p in pred_tags])
                all_labels.extend([idx2tag.get(t, "<UNK>") for t in true_tags])

    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report
    }
