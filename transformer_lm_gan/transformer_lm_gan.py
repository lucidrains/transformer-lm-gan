import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from x_transformers import (
    TransformerWrapper,
    Decoder
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class LanguageModel(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dim_head,
        heads,
        max_seq_len
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

