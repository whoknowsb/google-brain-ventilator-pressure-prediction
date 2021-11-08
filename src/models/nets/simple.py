from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class lstms(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4, norm=False):
        super().__init__()

        self.norm = nn.BatchNorm1d(input_size) if norm else nn.Identity()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
        )

        self.pressure_in = nn.Linear(hidden_size, 1)
        self.pressure_out = nn.Linear(hidden_size, 1)

        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[slice((n // 4), (n // 2))].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # print(name,m)
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        bs = x.shape[0]
        x = self.norm(torch.transpose(x, -2, -1))
        x = torch.transpose(x, -2, -1)
        x, _ = self.lstm(x, None)
        x = self.head(x)
        x_in = self.pressure_in(x).view(bs, -1)
        x_out = self.pressure_out(x).view(bs, -1)
        return torch.stack([x_in, x_out])

