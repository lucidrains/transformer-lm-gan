from __future__ import annotations

import torch
from torch import nn
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
from einops.layers.torch import Rearrange, Reduce

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

# hinge gan losses

def generator_hinge_loss(fake):
    return fake.mean()

def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# classes

class Discriminator(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dim_head,
        heads,
        max_seq_len,
        depth,
        gp_weight = 10.
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.transformer = Encoder(
            dim = dim,
            attn_dim_head = dim_head,
            heads = heads,
            depth = depth,
            use_rmsnorm = True,
            rotary_pos_emb = True
        )

        self.to_real_fake_pred = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

        # loss related

        self.gp_weight = gp_weight

    def forward(
        self,
        x,
        return_gradient_penalty = False
    ):

        if x.dtype in (torch.int, torch.long):
            tokens = self.token_emb(x)
        else:
            tokens = x

        embed = self.transformer(tokens)

        real_fake_logit = self.to_real_fake_pred(embed)

        if not return_gradient_penalty:
            return real_fake_logit

        # compute the zero-mean gradient penalty for both reals and fakes
        # from recent Cornell / Brown paper claiming this fixes GAN stability. we will see..

        assert self.training

        zero_mean_gp = gradient_penalty(tokens, real_fake_logits, gp_weight = self.gp_weight)

        return real_fake_logit, zero_mean_gp

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
                heads = heads,
                use_rmsnorm = True,
                rotary_pos_emb = True
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
