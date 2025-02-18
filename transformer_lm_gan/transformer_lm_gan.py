from __future__ import annotations
import math

import torch
from torch import nn, cat, stack, tensor, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torch.nn.functional as F

from x_transformers import (
    TransformerWrapper,
    Decoder,
    Encoder
)

from vector_quantize_pytorch.vector_quantize_pytorch import (
    rotate_to
)

from adam_atan2_pytorch import AdoptAtan2

# einstein notation related

from einx import get_at
from einops import einsum, rearrange, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

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

    gradients = rearrange(gradients, 'b ... -> b (...)')
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

        zero_mean_gp = gradient_penalty(tokens, real_fake_logit, weight = self.gp_weight)

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
            return_only_embed = True,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                attn_dim_head = dim_head,
                heads = heads,
                use_rmsnorm = True,
                rotary_pos_emb = True,
            )
        )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
        cache_kv = True,
        return_with_prompt = True,
        eps = 1e-10
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        cache = None

        gumbel_noises = []

        for _ in range(sample_num_times):
            logits, next_cache = self.forward(out, return_intermediates = True, cache = cache)
            logits = logits[:, -1]

            if cache_kv:
                cache = next_cache

            logits = top_k(logits, thres = filter_thres)

            logits = logits / max(temperature, eps)

            noise = gumbel_noise(logits)

            gumbel_noises.append(noise)

            logits = logits + noise

            sample = logits.argmax(dim = -1, keepdim = True)

            out = torch.cat((out, sample), dim = -1)

        if not return_with_prompt:
            out = out[..., prompt_seq_len:]

        return out, (filter_thres, temperature, stack(gumbel_noises))

    def forward(
        self,
        x,
        return_ar_loss = False,
        return_intermediates = False,
        cache = None,
        return_only_embed = False,
        rotate_embed_to_next_for_discr = False
    ):
        token_embed = self.transformer.token_emb

        if return_ar_loss or rotate_embed_to_next_for_discr:
            x, labels = x[:, :-1], x[:, 1:]

        embed, intermediates = self.transformer(
            x,
            cache = cache,
            return_intermediates = True,
        )

        if rotate_embed_to_next_for_discr:
            label_embed = token_embed(labels)
            return rotate_to(embed, label_embed) # same rotation trick Fifty et al applied for VQ in lieu of straight through

        if return_only_embed:
            return embed

        logits = einsum(embed, token_embed.emb.weight, 'b n d, l d -> b n l')

        if not return_ar_loss:
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
        generator: LanguageModelGenerator | dict,
        discriminator: Discriminator | dict,
        learning_rate = 2e-4,
        optimizer_klass = AdoptAtan2,
        optimizer_kwargs: dict = dict()
    ):
        super().__init__()

        if isinstance(generator, dict):
            generator = LanguageModelGenerator(**generator)

        if isinstance(discriminator, dict):
            discriminator = Discriminator(**discriminator)

        self.generator = generator

        # weight tie the token embeddings

        self.token_emb = generator.transformer.token_emb

        discriminator.token_emb = self.token_emb
        self.discriminator = discriminator

        # optimizers

        self.generator_optim = optimizer_klass(self.generator.parameters(), lr = learning_rate, **optimizer_kwargs)

        self.discriminator_optim = optimizer_klass(self.discriminator.parameters(), lr = learning_rate, **optimizer_kwargs)

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

    def generate_forward(
        self,
        seq,
        generate_kwargs: dict = dict()
    ):
        seq_len = seq.shape[-1]

        prompt = seq[:, :(seq_len // 2)]

        generated, sampling_hparams = self.generator.generate(prompt, seq_len, **generate_kwargs)

        discr_input = self.generator(generated, rotate_embed_to_next_for_discr = True)

        logits = self.discriminator(discr_input)

        loss = generator_hinge_loss(logits)
        return loss

    def discriminate_forward(
        self,
        seq,
        generate_kwargs: dict = dict(
            temperature = 1.
        ),
        apply_grad_penalty = True,
        return_loss_breakdown = False
    ):
        seq_len = seq.shape[-1]

        real = seq

        prompt = seq[:, :(seq_len // 2)]
        fake, sampling_hparams = self.generator.generate(prompt, seq_len, **generate_kwargs)

        real_embed = self.token_emb(real[:, :-1])
        fake_embed = self.generator(fake, rotate_embed_to_next_for_discr = True)

        discr_input, packed_shape = pack((real_embed, fake_embed), '* n d')

        if apply_grad_penalty:
            real_fake_logits, zero_mean_gp_loss = self.discriminator(discr_input, return_gradient_penalty = True)
        else:
            real_fake_logits = self.discriminator(discr_input)
            zero_mean_gp_loss = self.zero

        real_logits, fake_logits = unpack(real_fake_logits, packed_shape, '*')

        discr_loss = discriminator_hinge_loss(real_logits, fake_logits)

        total_loss =  discr_loss + zero_mean_gp_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (discr_loss, zero_mean_gp_loss)

    def forward(self, seq):
        # plain autoregressive loss for generator

        return self.generator(seq, return_ar_loss = True)
