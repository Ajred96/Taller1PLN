# model.py

import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRFTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, pad_idx=0):
        super(BiLSTMCRFTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def _get_emissions(self, x):
        """Paso compartido: entrada → emission scores."""
        embeds = self.embedding(x)         # (batch, seq_len, emb_dim)
        lstm_out, _ = self.lstm(embeds)    # (batch, seq_len, hidden*2)
        emissions = self.fc(lstm_out)      # (batch, seq_len, tagset_size)
        return emissions

    def forward(self, x, tags, mask):
        """
        Usado durante entrenamiento.
        Devuelve el loss (log-likelihood negativo).
        """
        emissions = self._get_emissions(x)
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, x, mask):
        """
        Usado durante evaluación e inferencia.
        Devuelve la secuencia de tags más probable (algoritmo de Viterbi).
        """
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions, mask=mask)
