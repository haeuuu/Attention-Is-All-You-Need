import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        """Layer Normalization
        paper : https://arxiv.org/pdf/1607.06450.pdf

        Parameters
        ----------
        dim : tuple
        gamma : torch.FloatTensor
            scale factor
        beta : torch.FloatTensor
            shift factor
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
            
        self.gamma = nn.Parameter(torch.Tensor(*dim))
        self.beta = nn.Parameter(torch.Tensor(*dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1.)
        self.beta.data.fill_(0.)

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = ((x- mean) ** 2).mean(dim = -1, keepdim = True).sqrt()
        normalized = (x - mean) / std

        return self.gamma * normalized + self.beta