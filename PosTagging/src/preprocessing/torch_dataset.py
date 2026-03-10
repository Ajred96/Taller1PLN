import torch
from torch.utils.data import Dataset

from prepare_ancora_pipeline import prepare_ancora_pipeline
from pad_sequences import pad_sequences


class POSDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
            "attention_mask": self.masks[idx]
        }


if __name__ == "__main__":
    path = "../../data/raw/ancora/ancora_corpus_pos.csv"

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

    print("\nTamaño train_dataset:", len(train_dataset))
    print("Tamaño val_dataset:", len(val_dataset))
    print("Tamaño test_dataset:", len(test_dataset))

    sample = train_dataset[0]

    print("\nClaves del sample:")
    print(sample.keys())

    print("\nShape input_ids:", sample["input_ids"].shape)
    print("Shape labels:", sample["labels"].shape)
    print("Shape attention_mask:", sample["attention_mask"].shape)

    print("\nPrimer sample input_ids:")
    print(sample["input_ids"])

    print("\nPrimer sample labels:")
    print(sample["labels"])

    print("\nPrimer sample attention_mask:")
    print(sample["attention_mask"])