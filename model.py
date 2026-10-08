import torch
import torch.nn as nn


class MiniAI(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        hidden_size=256
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        x, hidden = self.gru(
            x,
            hidden
        )

        x = self.output(x)

        return x, hidden
