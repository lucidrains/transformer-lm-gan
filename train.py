import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset

from transformer_lm_gan.transformer_lm_gan import (
    LanguageModelGenerator,
    Discriminator,
    GAN
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
GRAD_ACCUM_EVERY = 1
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 32
GENERATE_EVERY = 500
GENERATE_LENGTH = 128
SEQ_LEN = 128

APPLY_GRAD_PENALTY_EVERY = 4 # only do zero mean gp every number of steps, for efficiency, proven out by Karras et al since Stylegan2

# training

ADVERSARIAL = True
AUTOREGRESSIVE = True

AUTOREGRESSIVE_EVERY = 1
ADVERSARIAL_EVERY = 1
ADVERSARIAL_AFTER = 500
STRATEGY = 'gumbel_one_hot'

# STRATEGY = 'gumbel_one_hot'

assert ADVERSARIAL or AUTOREGRESSIVE

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# the language model generator

gan = GAN(
    generator = dict(
        num_tokens = 256,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        max_seq_len = SEQ_LEN
    ),
    discriminator = dict(
        num_tokens = 256,
        dim = 512,
        depth = 4,
        dim_head = 64,
        heads = 8,
        max_seq_len = SEQ_LEN
    ),
    strategy = STRATEGY
).cuda()

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

last_gp_loss = tensor(0.)

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    gan.train()

    if (
        ADVERSARIAL and
        divisible_by(i, ADVERSARIAL_EVERY) and
        i > ADVERSARIAL_AFTER
    ):
        apply_grad_penalty = divisible_by(i // ADVERSARIAL_EVERY, APPLY_GRAD_PENALTY_EVERY)

        gan.discriminator_optim.zero_grad()

        for _ in range(GRAD_ACCUM_EVERY):
            data = next(train_loader)

            loss, (hinge_loss, gp_loss) = gan.discriminate_forward(
                data,
                return_loss_breakdown = True,
                apply_grad_penalty = apply_grad_penalty
            )

            (loss / GRAD_ACCUM_EVERY).backward()

            if apply_grad_penalty:
                last_gp_loss = gp_loss

        gan.discriminator_optim.step()

        gan.generator_optim.zero_grad()

        for _ in range(GRAD_ACCUM_EVERY):
            data = next(train_loader)

            gen_loss = gan.generate_forward(data)

            (gen_loss / GRAD_ACCUM_EVERY).backward()

        gan.generator_optim.step()

        print(f"discr: {hinge_loss.item():.4f} | gen: {gen_loss.item():.4f} | gp: {last_gp_loss.item():.4f}")

    if AUTOREGRESSIVE and divisible_by(i, AUTOREGRESSIVE_EVERY):
        gan.ar_generator_optim.zero_grad()

        for _ in range(GRAD_ACCUM_EVERY):
            data = next(train_loader)

            loss = gan.generator(data, return_ar_loss = True)

            (loss / GRAD_ACCUM_EVERY).backward()

        print(f"ar train loss: {loss.item():.4f}")

        torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), 0.5)

        gan.ar_generator_optim.step()

    if divisible_by(i, VALIDATE_EVERY):
        gan.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = gan.generator(valid_data, return_ar_loss = True)
            print(f"ar valid loss: {loss.item():.4f}")

    if divisible_by(i, GENERATE_EVERY):
        gan.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)

        print("\n", "*" * 100, "\n")
        print(f"\n{prime}\n")

        prompt = inp[None, ...]

        sampled, _ = gan.generator.generate(prompt, GENERATE_LENGTH, progress_bar = True)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\n{base_decode_output}\n")
