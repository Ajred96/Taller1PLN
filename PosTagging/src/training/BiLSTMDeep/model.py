import torch
import torch.nn as nn

class BiLSTMDeepTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, dense_dim=128, pad_idx=0):
        super(BiLSTMDeepTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Capas Densas Intermedias
        self.intermediate_dense = nn.Linear(hidden_dim * 2, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Capa de salida final
        self.fc = nn.Linear(dense_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)

        # Paso por capas intermedias
        out = self.intermediate_dense(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)

        # Salida final
        logits = self.fc(out)
        return logits
