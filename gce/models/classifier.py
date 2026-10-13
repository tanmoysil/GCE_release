

from abc import ABC

import torch.nn as nn
import torch


class DummyClassifier(nn.Module):
    def __init__(self):
        super(DummyClassifier, self).__init__()
        self.dummyLayer = nn.Linear(1, 2)

    def forward(self, x):
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        x = torch.reshape(x, (x.shape[0], 1))
        out = self.dummyLayer(x)
        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs


class linClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linClassifier, self).__init__()

        self.hidden_dim1 = 256
        self.hidden_dim2 = 128
        self.hidden_dim3 = 64

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=self.hidden_dim2, out_features=output_dim),
        )

    def forward(self, x):
        out = self.model(x)
        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs
