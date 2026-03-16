import copy
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_model(model, train_loader, val_loader, optimizer, device, epochs=50, patience=5):
    """Entrena el modelo con Early Stopping monitoreando Val Loss."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train = total_train_loss / num_batches
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        num_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids)
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)

                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                num_val += 1

        avg_val = total_val_loss / num_val
        val_losses.append(avg_val)

        # --- Early Stopping ---
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
    """Evalúa en test. Retorna accuracy, macro_f1, weighted_f1 y classification_report."""
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["attention_mask"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            for i in range(len(masks)):
                mask = masks[i].bool()
                pred_tags = preds[i][mask].cpu().numpy()
                true_tags = labels[i][mask].cpu().numpy()

                all_preds.extend([idx2tag.get(p, "<UNK>") for p in pred_tags])
                all_labels.extend([idx2tag.get(t, "<UNK>") for t in true_tags])

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report
    }
