import sys
sys.path.append('..')

import torch
from torch import nn

from transformer import Transformer
from positional_encoding import PositionalEncoding
from utils import generate_pad_mask, generate_masked_attn_mask


class TransformerforSeq2Seq(nn.Module):
    def __init__(self,
                n_tokens_enc,
                n_tokens_dec,
                n_encoders = 6,
                n_decoders = 6,
                n_heads = 8,
                d_model = 512,
                max_len = 5000,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1,
                pad_idx = -1):
        super().__init__()
        """Transformer for Machine Translation
        """
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        self.enc_emb = nn.Embedding(n_tokens_enc, d_model)
        self.dec_emb = nn.Embedding(n_tokens_dec, d_model)

        self.enc_emb_dropout = nn.Dropout(drop_prob)
        self.dec_emb_dropout = nn.Dropout(drop_prob)

        self.pos_encoder = PositionalEncoding(d_model = d_model, max_len = max_len)

        self.transformer = Transformer(n_encoders = n_encoders,
                                        n_decoders = n_decoders,
                                        n_heads = n_heads,
                                        d_model = d_model,
                                        d_hidden_ffnn = d_hidden_ffnn,
                                        drop_prob = drop_prob)
        
        self.linear = nn.Linear(d_model, n_tokens_dec)
        self.dropout = nn.Dropout(drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.enc_emb.weight, mean = 0, std = self.d_model**(-0.5))
        nn.init.normal_(self.dec_emb.weight, mean = 0, std = self.d_model**(-0.5))
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def get_enc_input(self, inputs):
        mask = generate_pad_mask(inputs, self.pad_idx)

        inputs = self.enc_emb(inputs)
        inputs_with_pos = self.pos_encoder(inputs)
        inputs_dropout = self.enc_emb_dropout(inputs_with_pos)

        return inputs_dropout, mask

    def get_dec_input(self, targets):
        mask = generate_masked_attn_mask(targets, self.pad_idx, fill = 1)

        targets = self.dec_emb(targets)
        targets_with_pos = self.pos_encoder(targets)
        targets_dropout = self.dec_emb_dropout(targets_with_pos)

        return targets_dropout, mask

    def forward(self, inputs, targets):
        inputs, inputs_mask = self.get_enc_input(inputs)
        targets, targets_mask = self.get_dec_input(targets)

        output = self.transformer(inputs, inputs_mask, targets, targets_mask)

        output = self.linear(output)
        output = self.dropout(output)

        return output

if __name__ == '__main__':
    inputs = torch.LongTensor([[5,2,1],[9,1,1]])
    targets = torch.LongTensor([[9,6],[6,4]])

    model = TransformerforSeq2Seq(n_tokens_enc = 10,
                                n_tokens_dec = 20)
    out = model(inputs, targets)

    print(out)
    print(out.shape)