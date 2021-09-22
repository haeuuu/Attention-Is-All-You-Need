"""
Transformer
    - preprocessing
        - positional embedding
    - Encoder
        - self attention
        - feedforward
    - Decoder
        - masked self attention
            - input : masked Q,K,V (이전 step 예측값)
        - encoder decoder attention
        - feedforward

(+)
- residual connection
- layer normalization
"""

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask = None):
        """
        Parameters
        ----------
        Q, K : torch.Tensor
            query, key (batch_size, n_heads, seq_len, d_model)
        V : torch.Tensor
            value (batch_size, n_heads, seq_len, d_value)

        Returns
        -------
        output : torch.Tensor
            (batch_size, n_heads, seq_len, d_value)
        """
        scale = K.shape[-1] ** (-2)
        dot_product = Q @ K.transpose(-1, -2)
        scaled_dot_product = dot_product / scale

        if mask:
            scaled_dot_product.masked_fill_(mask, float('inf'))

        attention_score = scaled_dot_product.softmax(dim = -1)

        output = attention_score @ V

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_query, d_value):
        super().__init__()
        self.n_heads = n_heads
        self.d_query = d_query
        self.d_value = d_value

        self.w = nn.ModuleDict()
        self.w['w_q'] = nn.Linear(d_model, n_heads * d_query)
        self.w['w_k'] = nn.Linear(d_model, n_heads * d_query)
        self.w['w_v'] = nn.Linear(d_model, n_heads * d_value)
        
        self.attention = ScaledDotProductAttention()

        self.w['w_out'] = nn.Linear(n_heads * d_value, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.w.values():
            torch.nn.init.xavier_normal_(linear.weight)
            torch.nn.init.xavier_normal_(linear.bias)

    def transform(self, Q, K, V):
        Q_concat = self.w['w_q'](Q)
        K_concat = self.w['w_k'](K)
        V_concat = self.w['w_v'](V)

        return Q_concat, K_concat, V_concat

    def split(self, X, dim):
        batch_size, seq_len, d_concat = X.shape

        assert self.n_heads * dim == d_concat, 'DimensionError'

        return X.view(batch_size, self.n_heads, seq_len, dim)

    def concat(self, X):
        batch_size, n_heads, seq_len, d_value = X.shape
        return X.view(batch_size, seq_len, n_heads * d_value)

    def forward(self, Q, K, V, mask = None):
        Q_concat, K_concat, V_concat = self.transform(Q, K, V)

        Q = self.split(Q_concat, dim = self.d_query)
        K = self.split(K_concat, dim = self.d_query)
        V = self.split(V_concat, dim = self.d_value)

        output = self.attention(Q, K, V, mask)
        output_concat = self.concat(output)

        aggregated = self.w['w_out'](output_concat)

        return aggregated

class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x