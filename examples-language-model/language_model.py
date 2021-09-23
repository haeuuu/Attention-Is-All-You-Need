import sys
sys.path.append('..')

import torch
from torch import nn

from utils import generate_lm_mask
from transformer import TransformerEncoder
from positional_encoding import PositionalEncoding


class TransformerforLM(nn.Module):
    def __init__(self,
                n_tokens,
                n_encoders = 6,
                n_heads = 8,
                d_model = 512,
                max_len = 5000,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
        super().__init__()
        """Transformer for Language Modeling
        """

        self.emb = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model = d_model, max_len = max_len)
        self.encoder  = TransformerEncoder(n_layers = n_encoders,
                                            n_heads = n_heads,
                                            d_model = d_model,
                                            d_hidden_ffnn = d_hidden_ffnn,
                                            drop_prob = drop_prob)
        self.decoder = nn.Linear(d_model, n_tokens)
        self.log_softmax = nn.LogSoftmax(dim = -1)

    def get_input_features(self, tokens):
        emb = self.emb(tokens)
        emb_with_pos = self.pos_encoder(emb)

        return emb_with_pos

    def forward(self, src):
        inputs = self.get_input_features(src)
        mask = generate_lm_mask(inputs, fill = 1)

        encoder_out = self.encoder(inputs, mask)
        decoder_out = self.decoder(encoder_out)

        pred = self.log_softmax(decoder_out)

        return pred