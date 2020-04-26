import copy

from torch import nn

from models.Transform.attention import MultiHeadedAttention
from models.Transform.decoder import Decoder, DecoderLayer
from models.Transform.embeddings import Embeddings
from models.Transform.encoder import Encoder, EncoderLayer
from models.Transform.feedforward import PositionwiseFeedForward
from models.Transform.generator import Generator
from models.Transform.model import EncoderDecoder
from models.Transform.positional_encoding import PositionalEncoding


def make_nlp_model(src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def make_image_model(stack_size=6, input_channel=64, output_channel=64,
                     d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=1024):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len=max_len)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), stack_size),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), stack_size),
        nn.Sequential(nn.Linear(input_channel, d_model), c(position)),
        nn.Sequential(nn.Linear(output_channel, d_model), c(position)),
        nn.Sequential(nn.Linear(d_model, output_channel), nn.Sigmoid()))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
