# predict.py

import torch


def predict_sentence(sentence, model, word2idx, tag2idx, device):
    """
    Predice las etiquetas POS para una oración en texto plano.
    Usa el algoritmo de Viterbi del CRF para la decodificación.
    """
    idx2tag = {v: k for k, v in tag2idx.items()}

    words = sentence.strip().split()
    if not words:
        return []

    # Codificar palabras (las desconocidas van a <UNK>)
    encoded = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in words]

    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    # La máscara para una oración sin padding es todo 1s
    mask = torch.ones(1, len(words), dtype=torch.bool).to(device)

    model.eval()
    with torch.no_grad():
        # decode() devuelve lista de listas, tomamos la primera (único batch)
        pred_indices = model.decode(input_tensor, mask)[0]

    pred_tags = [idx2tag.get(idx, "<UNK>") for idx in pred_indices]
    return list(zip(words, pred_tags))


def print_prediction(sentence, model, word2idx, tag2idx, device, label):
    """Imprime las predicciones de forma formateada."""
    preds = predict_sentence(sentence, model, word2idx, tag2idx, device)
    print(f"\n--- {label} ---")
    for word, tag in preds:
        print(f"  {word:20s} -> {tag}")
