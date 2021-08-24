from torch._C import device
import torch.nn as nn


class AptamerLSTM(nn.Module):
    def __init__(self):
        super(AptamerLSTM, self).__init__()

        self.embedding = nn.Embedding(5, 128)
        self.encoder = nn.LSTM(128, 512, 3)
        self.decoder = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(1, 0)  # weird input shape requirement
        embeds = self.embedding(x)
        y = self.encoder(embeds)[0]
        return self.decoder(y[-1, :, :])

