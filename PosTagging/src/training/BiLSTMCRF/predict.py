import torch

def predict_sentence(sentence, model, word2idx, tag2idx, device):
    """Predice usando Viterbi."""
    idx2tag = {v: k for k, v in tag2idx.items()}
    words = sentence.strip().split()
    if not words: return []

    encoded = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in words]
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    mask = torch.ones(1, len(words), dtype=torch.bool).to(device)

    model.eval()
    with torch.no_grad():
        pred_indices = model.decode(input_tensor, mask)[0]

    pred_tags = [idx2tag.get(idx, "<UNK>") for idx in pred_indices]
    return list(zip(words, pred_tags))

def print_prediction(sentence, model, word2idx, tag2idx, device, label):
    preds = predict_sentence(sentence, model, word2idx, tag2idx, device)
    print(f"\n--- {label} ---")
    for word, tag in preds:
        print(f"  {word:20s} -> {tag}")