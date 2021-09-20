import torch
from torch import nn

from attention import MultiHeadAttention, FeedForwardNN
from layer_normalization import LayerNormalization

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048):
        """Encoder Layer for Transformer
        Parameters
        ----------
        n_heads : int
            the number of heads in multi head attention.
        d_model : int
            the dimension of features.
        d_hidden_ffnn : int
            the hidden diemnsion of the feedforward network.
        """
        super().__init__()

        assert d_model%n_heads == 0, 'DimensionError. d_model must be a multiple of n_heads.'

        self.d_hidden = d_model // n_heads

        self.multi_head_attention = MultiHeadAttention(n_heads = n_heads,
                                                        d_model = d_model,
                                                        d_query = self.d_hidden,
                                                        d_value = self.d_hidden)
        self.layer_norm_for_attention = LayerNormalization(d_model)

        self.ffnn = FeedForwardNN(d_model = d_model,
                                  d_hidden = d_hidden_ffnn)
        self.layer_norm_for_ffnn = LayerNormalization(d_model)

    def self_attention(self, qkv, mask = None):
        attention = self.multi_head_attention(Q = qkv,
                                              K = qkv,
                                              V = qkv,
                                              mask = mask)
        attention_normalized = self.layer_norm_for_attention(qkv + attention)

        return attention_normalized

    def feed_forward(self, x):
        ffnn = self.ffnn(x)
        ffnn_normalized = self.layer_norm_for_ffnn(x + ffnn)

        return ffnn_normalized

    def forward(self, x, mask = None):
        """
        Notes
        -----
        1. self attention
            x' = multi_head_attention(x)
            x'' = layer_normalization(x + x')

        2. feed forward
            x* = feed_forward(x'')
            x** = layer_normalization(x'' + x*)

        return x**
        """
        attn_norm = self.self_attention(qkv = x, mask = mask)
        ffnn_norm = self.feed_forward(attn_norm)

        return ffnn_norm

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048):
        """Decoder Layer for Transformer
        Parameters
        ----------
        n_heads : int
            the number of heads in multi head attention.
        d_model : int
            the dimension of features.
        d_hidden_ffnn : int
            the hidden diemnsion of the feedforward network.
        """
        super().__init__()

        assert d_model%n_heads == 0, 'DimensionError. d_model must be a multiple of n_heads.'

        self.d_hidden = d_model // n_heads

        self.masked_attention = MultiHeadAttention(n_heads = n_heads,
                                                    d_model = d_model,
                                                    d_query = self.d_hidden,
                                                    d_value = self.d_hidden)
        self.layer_norm_for_masked_attn = LayerNormalization(d_model)

        self.encoder_decoder_attention = MultiHeadAttention(n_heads = n_heads,
                                                            d_model = d_model,
                                                            d_query = self.d_hidden,
                                                            d_value = self.d_hidden)
        self.layer_norm_for_ed_attn = LayerNormalization(d_model)

        self.ffnn = FeedForwardNN(d_model = d_model,
                                  d_hidden = d_hidden_ffnn)
        self.layer_norm_for_ffnn = LayerNormalization(d_model)

    def masked_self_attention(self, qkv, mask):
        attention = self.masked_attention(Q = qkv,
                                          K = qkv,
                                          V = qkv,
                                          mask = mask)
        attention_normalized = self.layer_norm_for_masked_attn(qkv + attention)

        return attention_normalized

    def self_attention(self, q, kv, mask):
        attention = self.multi_head_attention(Q = q,
                                              K = kv,
                                              V = kv,
                                              mask = mask)
        attention_normalized = self.layer_norm_for_ed_attn(q + attention)

        return attention_normalized

    def feed_forward(self, x):
        ffnn = self.ffnn(x)
        ffnn_normalized = self.layer_norm_for_ffnn(x + ffnn)

        return ffnn_normalized

    def forward(self, x, encoder_out, target_mask = None, encoder_pad_mask = None):
        """
        Paramters
        ---------
        x : torch.FloatTensor
            masked target features

        target_mask : torch.FloatTensor
            padding or masked toekn (0) otherwise (1)

        encoder_out : torch.FloatTensor
            encoder output

        encoder_pad_mask : torch.FloatTensor
            padding (0) otherwise (1)

        Notes
        -----
        1. masked self attention
            x' = multi_head_attention(x, mask)
            x'' = layer_normalization(x + x')

        2. encoder-decoder attention
            x* = multi_head_attention(x'')
            x** = layer_normalization(x'' + x*)

        2. feed forward
            x' = feed_forward(x**)
            x'' = layer_normalization(x** + x')

        return x*''
        """
        masked_attn_norm = self.masked_self_attention(qkv = x, mask = target_mask)
        attn_norm = self.self_attention(q = masked_attn_norm,
                                        kv = encoder_out,
                                        mask = encoder_pad_mask)
        ffnn_norm = self.feed_forward(attn_norm)

        return ffnn_norm

class TransformerEncoder(nn.Module):
    def __init__(self,
                n_layers = 6,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048):
        super().__init__()

        self.encoders = nn.Modulelist([TransformerEncoderLayer(n_heads = n_heads,
                                                                d_model = d_model,
                                                                d_hidden_ffnn = d_hidden_ffnn) \
                                                                for _ in range(n_layers)])

    def forward(self, x, mask = None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                n_layers = 6,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048):
        super().__init__()

        self.decoders = nn.Modulelist([TransformerDecoderLayer(n_heads = n_heads,
                                                                d_model = d_model,
                                                                d_hidden_ffnn = d_hidden_ffnn) \
                                                                for _ in range(n_layers)])

    def forward(self, x, encoder_out, target_mask = None, encoder_pad_mask = None):
        for decoder in self.decoders:
            x = decoder(x = x,
                        encoder_out = encoder_out,
                        target_mask = target_mask,
                        encoder_pad_mask = encoder_pad_mask)
        
        return x

class Transformer(nn.Module):
    def __init__(self,
                n_encoders = 6,
                n_decoders = 6,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048):
        """Transformer
        Attention is all you need (https://arxiv.org/pdf/1706.03762.pdf)

        Parameters
        ----------
        n_encoders, n_decoders : int
            the number of encoders and decoders
        n_heads : int
            the number of heads in multi head attention.
        d_model : int
            the dimension of features.
        d_hidden_ffnn : int
            the hidden diemnsion of the feedforward network.
        """
        super().__init__()

        self.encoder = TransformerEncoder(n_layers = n_encoders,
                                          n_heads = n_heads,
                                          d_model = d_model,
                                          d_hidden_ffnn = d_hidden_ffnn)
        
        self.decoder = TransformerDecoder(n_layers = n_decoders,
                                          n_heads = n_heads,
                                          d_model = d_model,
                                          d_hidden_ffnn = d_hidden_ffnn)

    def forward(self, inputs, inputs_pad_mask, masked_targets, target_mask):
        """
        Parameters
        ----------
        inputs : torch.FloatTensor
            shape : (batch_size, max_length, d_model)

        inputs_pad_mask : torch.FloatTensor
            shape : (batch_size, max_length, d_model)
            padding (0) otherwise (1)

        masked_targets : torch.FloatTensor
            shape : (batch_size, max_length, d_model)
            masked target features

        inputs_pad_mask : torch.FloatTensor
            shape : (batch_size, max_length, d_model)
            padding (0) otherwise (1)
        """
        encoder_out = self.encoder(x = inputs,
                                    mask = inputs_pad_mask)
        decoder_out = self.decoder(x = masked_targets,
                                    encoder_out = encoder_out,
                                    target_mask = target_mask,
                                    encoder_pad_mask = inputs_pad_mask)

        return decoder_out