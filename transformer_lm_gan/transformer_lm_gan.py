from __future__ import annotations

import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import (
    TransformerWrapper,
    Decoder,
    Encoder
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class Discriminator(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dim_head,
        heads,
        max_seq_len,
        depth
    ):
        super().__init__()

        self.discriminator = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Encoder(
                dim = dim,
                attn_dim_head = dim_head,
                heads = heads
            )
        )

    def forward(self, x):

        x = self.discriminator(x)

        return x

class LanguageModelGenerator(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dim_head,
        heads,
        max_seq_len,
        depth
    ):
        super().__init__()

        self.transformer = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                attn_dim_head = dim_head,
                heads = heads
            )
        )

    def forward(self, x):

        logits = self.transformer(x)

        return logits


class GAN(Module):
    def __init__(
        self,
        generator: LanguageModelGenerator,
        discriminator: Discriminator
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        return x
