from __future__ import annotations

import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import (
    TransformerWrapper,
    Decoder,
    Encoder
)

from vector_quantize_pytorch.vector_quantize_pytorch import (
    rotate_to
)

# einstein notation related

from einx import get_at
from einops import rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def gradient_penalty(
    inputs,
    output,
    weight = 10,
    center = 0.
):
    device = inputs.device

    gradients = torch_grad(
        outputs = output,
        inputs = inputs,
        grad_outputs = torch.ones_like(output, device = device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - center) ** 2).mean()

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

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.transformer = Encoder(
            dim = dim,
            attn_dim_head = dim_head,
            heads = heads
        )

        self.to_real_fake_pred = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
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
            tie_embedding = True,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                attn_dim_head = dim_head,
                heads = heads
            )
        )

    def forward(
        self,
        x,
        return_loss = False,
        return_intermediates = False,
        cache = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        logits, intermediates = self.transformer(x, cache = cache, return_intermediates = True)

        if not return_loss:
            if not return_intermediates:
                return logits

            return logits, intermediates

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = -1
        )

        return loss

class GAN(Module):
    def __init__(
        self,
        generator: LanguageModelGenerator,
        discriminator: Discriminator
    ):
        super().__init__()
        self.generator = generator

        # weight tie the token embeddings

        discriminator.token_emb = generator.token_emb
        self.discriminator = discriminator

    def forward(self, x):

        return x
