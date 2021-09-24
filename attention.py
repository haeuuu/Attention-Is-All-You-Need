import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, drop_prob = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

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
        scale = K.shape[-1] ** (1/2)
        dot_product = Q @ K.transpose(-1, -2)
        scaled_dot_product = dot_product / scale

        if mask is not None:
            scaled_dot_product.masked_fill_(mask, float('-inf'))

        attention_score = scaled_dot_product.softmax(dim = -1)
        attention_score = self.dropout(attention_score)

        output = attention_score @ V

        return output, attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_query, d_value, drop_prob = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_query = d_query
        self.d_value = d_value

        self.w = nn.ModuleDict()
        self.w['w_q'] = nn.Linear(d_model, n_heads * d_query)
        self.w['w_k'] = nn.Linear(d_model, n_heads * d_query)
        self.w['w_v'] = nn.Linear(d_model, n_heads * d_value)
        
        self.attention = ScaledDotProductAttention(drop_prob = drop_prob)

        self.w['w_out'] = nn.Linear(n_heads * d_value, d_model)
        self.dropout = nn.Dropout(drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.w.values():
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)

    def transform(self, Q, K, V):
        Q_concat = self.w['w_q'](Q)
        K_concat = self.w['w_k'](K)
        V_concat = self.w['w_v'](V)

        return Q_concat, K_concat, V_concat

    def split(self, X, dim):
        batch_size, seq_len, d_concat = X.shape

        assert self.n_heads * dim == d_concat, 'DimensionError'

        return X.view(batch_size, seq_len, self.n_heads, dim).transpose(1,2)

    def concat(self, X):
        batch_size, n_heads, seq_len, d_value = X.shape
        return X.view(batch_size, seq_len, n_heads * d_value)

    def forward(self, Q, K, V, mask = None):
        Q_concat, K_concat, V_concat = self.transform(Q, K, V)

        Q = self.split(Q_concat, dim = self.d_query)
        K = self.split(K_concat, dim = self.d_query)
        V = self.split(V_concat, dim = self.d_value)

        if mask is not None:
            # (batch_size, seq_len, d_model) to (batch_size, n_heads, seq_len, d_model)
            mask = mask.unsqueeze(1)

        output, attn_score = self.attention(Q, K, V, mask)
        output_concat = self.concat(output)

        aggregated = self.w['w_out'](output_concat)
        aggregated = self.dropout(aggregated)

        return aggregated, attn_score

class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_hidden, drop_prob = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

if __name__ == '__main__':
    attention = ScaledDotProductAttention(drop_prob = 0.1)
    attention.eval() # deactivate dropout
    
    batch_size, seq_len, d_model = 2, 3, 4

    inputs = torch.rand((batch_size, seq_len, d_model))
    mask = torch.tensor([[[0, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0]],

                        [[0, 0, 1],
                        [0, 0, 1],
                        [0, 1, 1]]])

    output, attn_score = attention(inputs, inputs, inputs, mask)

    print('inputs : ', inputs)
    print('lm mask : ', mask)
    print('output : ', output)
    print('attn_score : ',attn_score)