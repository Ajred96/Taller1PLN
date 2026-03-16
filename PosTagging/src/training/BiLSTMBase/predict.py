import torch


def predict_sentence(sentence, model, word2idx, tag2idx, device):
    """Predice las etiquetas POS para una oración en texto plano."""
    idx2tag = {v: k for k, v in tag2idx.items()}

    words = sentence.strip().split()
    if not words:
        return []

    encoded = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in words]

    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.argmax(logits, dim=-1)[0]

    pred_tags = [idx2tag.get(preds[i].item(), "<UNK>") for i in range(len(words))]
    return list(zip(words, pred_tags))


def print_prediction(sentence, model, word2idx, tag2idx, device, label):
    """Imprime las predicciones de forma formateada."""
    preds = predict_sentence(sentence, model, word2idx, tag2idx, device)
    print(f"\n--- {label} ---")
    for word, tag in preds:
        print(f"  {word:20s} -> {tag}")
