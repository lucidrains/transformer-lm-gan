import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformer_lm_gan.transformer_lm_gan import (
    LanguageModelGenerator,
    Discriminator,
    GAN
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

APPLY_GRAD_PENALTY_EVERY = 4 # only do zero mean gp every number of steps, for efficiency, proven out by Karras et al since Stylegan2

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# the language model generator

generator = LanguageModelGenerator(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    max_seq_len = SEQ_LEN
).cuda()

discr = Discriminator(
    num_tokens = 256,
    dim = 512,
    depth = 2,
    dim_head = 64,
    heads = 9,
    max_seq_len = SEQ_LEN
)

gan = GAN(
    generator = generator,
    discriminator = discr
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

optim = Adam(gan.generator.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    gan.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = gan.generator(data, return_ar_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()

    print(f"generator ar train loss: {loss.item():.4f}")

    torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        gan.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = gan.generator(valid_data, return_ar_loss = True)
            print(f"generator ar valid loss: {loss.item():.4f}")

    if i % GENERATE_EVERY == 0:
        gan.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)

        print("\n", "*" * 100, "\n")
        print(f"\n{prime}\n")

        prompt = inp[None, ...]

        sampled = gan.generator.generate(prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\n{base_decode_output}\n")
