from torch.utils.data import DataLoader

from prepare_ancora_pipeline import prepare_ancora_pipeline
from pad_sequences import pad_sequences
from torch_dataset import POSDataset


def build_ancora_dataloaders(path, batch_size=32):
    data = prepare_ancora_pipeline(path)

    pad_word = data["word2idx"]["<PAD>"]
    pad_tag = data["tag2idx"]["<PAD>"]

    # Padding usando max_len de train
    X_train_pad, train_masks, max_len = pad_sequences(data["X_train_enc"], pad_value=pad_word)
    y_train_pad, _, _ = pad_sequences(data["y_train_enc"], pad_value=pad_tag, max_len=max_len)

    X_val_pad, val_masks, _ = pad_sequences(data["X_val_enc"], pad_value=pad_word, max_len=max_len)
    y_val_pad, _, _ = pad_sequences(data["y_val_enc"], pad_value=pad_tag, max_len=max_len)

    X_test_pad, test_masks, _ = pad_sequences(data["X_test_enc"], pad_value=pad_word, max_len=max_len)
    y_test_pad, _, _ = pad_sequences(data["y_test_enc"], pad_value=pad_tag, max_len=max_len)

    train_dataset = POSDataset(X_train_pad, y_train_pad, train_masks)
    val_dataset = POSDataset(X_val_pad, y_val_pad, val_masks)
    test_dataset = POSDataset(X_test_pad, y_test_pad, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "word2idx": data["word2idx"],
        "tag2idx": data["tag2idx"],
        "max_len": max_len
    }


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

    loaders = build_ancora_dataloaders(path, batch_size=32)

    train_loader = loaders["train_loader"]

    print("\nMáxima longitud:", loaders["max_len"])
    print("Tamaño vocabulario palabras:", len(loaders["word2idx"]))
    print("Tamaño vocabulario tags:", len(loaders["tag2idx"]))

    batch = next(iter(train_loader))

    print("\nClaves del batch:")
    print(batch.keys())

    print("\nShape input_ids:", batch["input_ids"].shape)
    print("Shape labels:", batch["labels"].shape)
    print("Shape attention_mask:", batch["attention_mask"].shape)