from torch import nn as nn

from .layer_norm import LayerNorm
from .tool import clones


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm1 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(x)
        x = self.norm2(x)
        x = self.feed_forward(x)
        return x + self.dropout2(x)
