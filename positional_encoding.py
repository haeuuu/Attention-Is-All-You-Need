import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model = 512, max_len = 2048):
        """Positional Encoding Layer
            PE(pos, 2i) = sin( pos / 10000**(2i/d_model) )
            PE(pos, 2i+1) = cos( pos / 10000**(2i/d_model) )
        """
        super().__init__()

        positional_embedding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        denominator = torch.exp( - torch.arange(0, d_model, 2).float() * math.log(10000) / d_model)

        positional_embedding[:, 0::2] = torch.sin(torch.mul(pos, denominator))
        positional_embedding[:, 1::2] = torch.cos(torch.mul(pos, denominator))

        # (max_len, d_model) to (1, max_len, d_model)
        self.positional_embedding = positional_embedding.unsqueeze(0)

    def forward(self, x):
        """
        Paramters
        ---------
        x : torch.FloatTensor
            (batch_size, length, d_hidden)
        """
        length = x.shape[1]
        x = x + self.positional_embedding[:length, :]

        return x

