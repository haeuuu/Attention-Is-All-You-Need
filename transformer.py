import torch
from torch import nn

from attention import MultiHeadAttention, FeedForwardNN
from layer_normalization import LayerNormalization

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
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

        self.self_attn = MultiHeadAttention(n_heads = n_heads,
                                            d_model = d_model,
                                            d_query = self.d_hidden,
                                            d_value = self.d_hidden,
                                            drop_prob = drop_prob)
        self.layer_norm_for_self_attn = LayerNormalization(d_model)

        self.ffnn = FeedForwardNN(d_model = d_model,
                                  d_hidden = d_hidden_ffnn,
                                  drop_prob = drop_prob)

        self.layer_norm_for_ffnn = LayerNormalization(d_model)

    def self_attention(self, qkv, mask = None):
        output = self.self_attn(Q = qkv,
                                K = qkv,
                                V = qkv,
                                mask = mask)
        output_normalized = self.layer_norm_for_self_attn(qkv + output)

        return output_normalized

    def feed_forward(self, x):
        output = self.ffnn(x)
        output_normalized = self.layer_norm_for_ffnn(x + output)

        return output_normalized

    def forward(self, x, mask = None):
        """
        Notes
        -----
        1. self attention
            x' = multi_head_attention(x, mask)
            x'' = layer_normalization(x + x')

        2. feed forward
            x* = feed_forward(x'')
            x** = layer_normalization(x'' + x*)

        return x**
        """
        output_attn = self.self_attention(qkv = x, mask = mask)
        output_ffnn = self.feed_forward(output_attn)

        return output_ffnn

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
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

        self.masked_self_attn = MultiHeadAttention(n_heads = n_heads,
                                                    d_model = d_model,
                                                    d_query = self.d_hidden,
                                                    d_value = self.d_hidden,
                                                    drop_prob = drop_prob)
        self.layer_norm_for_self_attn = LayerNormalization(d_model)

        self.ed_attn = MultiHeadAttention(n_heads = n_heads,
                                        d_model = d_model,
                                        d_query = self.d_hidden,
                                        d_value = self.d_hidden,
                                        drop_prob = drop_prob)
        self.layer_norm_for_ed_attn = LayerNormalization(d_model)

        self.ffnn = FeedForwardNN(d_model = d_model,
                                  d_hidden = d_hidden_ffnn,
                                  drop_prob = drop_prob)
        self.layer_norm_for_ffnn = LayerNormalization(d_model)

    def masked_self_attention(self, qkv, mask):
        output = self.masked_self_attn(Q = qkv,
                                        K = qkv,
                                        V = qkv,
                                        mask = mask)
        output_normalized = self.layer_norm_for_self_attn(qkv + output)

        return output_normalized

    def encoder_decoder_attention(self, q, kv, mask):
        output = self.ed_attn(Q = q,
                            K = kv,
                            V = kv,
                            mask = mask)
        output_normalized = self.layer_norm_for_ed_attn(q + output)

        return output_normalized

    def feed_forward(self, x):
        output = self.ffnn(x)
        output_normalized = self.layer_norm_for_ffnn(x + output)

        return output_normalized

    def forward(self, x, encoder_out, targets_mask = None, inputs_mask = None):
        """
        Paramters
        ---------
        x : torch.FloatTensor
            masked target features

        target_mask : torch.FloatTensor
            padding or masked token (0) otherwise (1)

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
            x* = multi_head_attention(x'', y, mask) where y ; encoder output
            x** = layer_normalization(x'' + x*)

        2. feed forward
            x' = feed_forward(x**)
            x'' = layer_normalization(x** + x')

        return x''
        """
        output_masked_attn = self.masked_self_attention(qkv = x, mask = targets_mask)
        output_ed_attn = self.encoder_decoder_attention(q = output_masked_attn,
                                                        kv = encoder_out,
                                                        mask = inputs_mask)
        output_ffnn = self.feed_forward(output_ed_attn)

        return output_ffnn

class TransformerEncoder(nn.Module):
    def __init__(self,
                n_layers = 6,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
        super().__init__()

        self.encoders = nn.ModuleList([TransformerEncoderLayer(n_heads = n_heads,
                                                                d_model = d_model,
                                                                d_hidden_ffnn = d_hidden_ffnn,
                                                                drop_prob = drop_prob) \
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
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
        super().__init__()

        self.decoders = nn.ModuleList([TransformerDecoderLayer(n_heads = n_heads,
                                                                d_model = d_model,
                                                                d_hidden_ffnn = d_hidden_ffnn,
                                                                drop_prob = drop_prob) \
                                                                for _ in range(n_layers)])

    def forward(self, x, encoder_out, targets_mask = None, inputs_mask = None):
        for decoder in self.decoders:
            x = decoder(x = x,
                        encoder_out = encoder_out,
                        targets_mask = targets_mask,
                        inputs_mask = inputs_mask)
        
        return x

class Transformer(nn.Module):
    def __init__(self,
                n_encoders = 6,
                n_decoders = 6,
                n_heads = 8,
                d_model = 512,
                d_hidden_ffnn = 2048,
                drop_prob = 0.1):
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
                                          d_hidden_ffnn = d_hidden_ffnn,
                                          drop_prob = drop_prob)
        
        self.decoder = TransformerDecoder(n_layers = n_decoders,
                                          n_heads = n_heads,
                                          d_model = d_model,
                                          d_hidden_ffnn = d_hidden_ffnn,
                                          drop_prob = drop_prob)

    def forward(self, inputs, inputs_mask, targets, targets_mask):
        """
        Parameters
        ----------
        inputs : torch.FloatTensor
            shape : (batch_size, seq_len, d_model)

        inputs_mask : torch.FloatTensor
            shape : (batch_size, seq_len, d_model)
            padding (0) otherwise (1)

        targets : torch.FloatTensor
            shape : (batch_size, seq_len, d_model)
            masked target features

        targets_mask : torch.FloatTensor
            shape : (batch_size, seq_len, d_model)
            padding (0) otherwise (1)
        """
        encoder_out = self.encoder(x = inputs,
                                    mask = inputs_mask)
        decoder_out = self.decoder(x = targets,
                                    encoder_out = encoder_out,
                                    targets_mask = targets_mask,
                                    inputs_mask = inputs_mask)

        return decoder_out

if __name__ == '__main__':
    """
    Example
    """
    
    batch_size, d_hidden = 32, 512
    src = torch.rand((batch_size, 10, d_hidden)) # k_len = 10
    tgt = torch.rand((batch_size, 20, d_hidden)) # q_len = 20

    transformer = Transformer()
    out = transformer(inputs = src,
                    inputs_mask = None,
                    targets = tgt,
                    targets_mask = None)

    print(out)
    print(out.shape) # (batch_size, q_len, d_hidden)