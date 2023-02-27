from enum import Enum
from dataset import (
    BrainDataset,
    BigBrainDataset,
    big_brain_collate_fn,
    BinBatchSampler,
    BinSampler,
    UltraDuperBigBrainDataset,
)
from transformer import TransformerModel, generate_square_subsequent_mask
from torch.utils.data import DataLoader
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
the_path = "wikitext-103/wiki.train.tokens"


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model(vocab_size) -> torch.nn.Module:
    model = TransformerModel(vocab_size, 32, 8, 64, 4)
    model = model.to(device)
    return model


def warmup():
    model = get_gpt2_model(1000)
    x = torch.randint(high=100, size=(32, 640), dtype=torch.int32)
    attn = generate_square_subsequent_mask(32)
    attn = attn.to(device)
    x = x.to(device)
    outputs = model(x, attn)


ds_1 = BrainDataset(the_path)
ds_2 = BigBrainDataset(the_path)
ds_3 = UltraDuperBigBrainDataset(the_path, n_bins=10)


def run_epoch(data_mode: DataMode, batch_size) -> None:
    if data_mode == DataMode.BRAIN:
        ds = ds_1
        train_loader = DataLoader(ds, batch_size, shuffle=True, num_workers=6)
    elif data_mode == DataMode.BIG_BRAIN:
        ds = ds_2
        train_loader = DataLoader(ds, batch_size, shuffle=True, num_workers=6, collate_fn=big_brain_collate_fn)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        ds = ds_3
        ss = BinBatchSampler(
            BinSampler(ds, replacement=False, num_samples=(len(ds) // batch_size) + ds.n_bins * 10),
            batch_size=batch_size,
            drop_last=False,
        )
        train_loader = DataLoader(ds, collate_fn=big_brain_collate_fn, batch_sampler=ss, num_workers=6)
    else:
        raise ValueError("Not implemented")

    vocab_size = len(ds.ix2word)
    model = get_gpt2_model(vocab_size)
    for i, x in enumerate(train_loader):
        mask_size = x.shape[0]
        # print(i, x.shape)
        attn = generate_square_subsequent_mask(mask_size)
        attn = attn.to(device)
        x = x.to(device)
        outputs = model(x, attn)
        # print(outputs)
    torch.cuda.synchronize()


import timeit

number = 1
rep = 3
for mode in (DataMode.BRAIN, DataMode.BIG_BRAIN, DataMode.ULTRA_DUPER_BIG_BRAIN):
    for batch_size in [16, 32]:
        warmup()
        t1 = timeit.repeat(lambda: run_epoch(mode, batch_size), number=number, repeat=rep)
        print(mode, batch_size, np.mean(t1), t1)
# run_epoch(DataMode.BRAIN, 32)
