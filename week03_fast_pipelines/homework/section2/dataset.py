# %%
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import RandomSampler, DataLoader, BatchSampler, Sampler
from typing import Sized, Iterator, Union, Iterable, List
import torch.nn.functional as F
import torchtext
from collections import Counter
import pandas as pd
import numpy as np


MAX_LENGTH = 640


class BaseBrainDataset(Dataset):
    UNK = "unk"
    EOS = "eos"

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        lines = self.load_file(self.data_path)
        tokenized_lines = self.tokenize(lines)
        self.ix2word, self.word2ix = self.create_indexes(tokenized_lines)
        self.tokenized_indexes = self.tokenize_to_indexes(tokenized_lines)

    def __len__(self):
        return len(self.tokenized_indexes)

    def tokenize_to_indexes(self, tokenized_lines):
        tokenized_indexes = [[self.word2ix.get(x, self.word2ix[self.UNK]) for x in row] for row in tokenized_lines]
        return tokenized_indexes

    def create_indexes(self, tokenized_lines):
        counter = Counter()
        for x in tokenized_lines:
            counter.update(x)
        ix2word = {}

        top_10k = counter.most_common(1000)
        for i, x in enumerate(top_10k):
            ix2word[i] = x[0]

        ix2word[i + 1] = self.UNK
        ix2word[i + 2] = self.EOS
        word2ix = {v: k for k, v in ix2word.items()}
        return ix2word, word2ix

    def tokenize(self, lines):
        tokenized_lines = [self.tokenizer(x) for x in lines]
        return tokenized_lines

    def load_file(self, data_path):
        with open(data_path, "r+") as f:
            lines = [x.strip().replace("\n", "") for x in f.readlines()]
            lines = [x for x in lines if x]
            lines = [x for x in lines if x[0] != "="]
        return lines


class BrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        indexes = self.tokenized_indexes[idx][: self.max_length]
        indexes = torch.as_tensor(indexes, dtype=torch.int32)
        pad_size = self.max_length - len(indexes)
        indexes = F.pad(indexes, (0, pad_size), "constant", self.word2ix[self.EOS])
        return indexes


class BigBrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)

    def __getitem__(self, idx: int):
        indexes = self.tokenized_indexes[idx][: self.max_length]
        indexes = torch.as_tensor(indexes, dtype=torch.int32)
        return indexes, self.word2ix[self.EOS]


def big_brain_collate_fn(tuples: list[torch.Tensor]) -> torch.Tensor:
    ls = [x[0] for x in tuples]
    eos_ix = [x[1] for x in tuples][0]
    max_len = max([x.shape[0] for x in ls])
    indexes_ls = []
    for x in ls:
        pad_size = max_len - x.shape[0]
        indexes = F.pad(x, (0, pad_size), "constant", eos_ix)
        indexes_ls.append(indexes)

    result = torch.vstack(indexes_ls)
    return result


class UltraDuperBigBrainDataset(BaseBrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        super().__init__(data_path)
        self.n_bins = n_bins
        self.load_bins()

    def load_bins(self):
        self.bins = pd.qcut([len(x) for x in self.tokenized_indexes], self.n_bins, labels=False)
        self.bins_ixes = {}
        self.num_samples = {}
        for i in range(self.n_bins):
            bin_ixes = [j for j, x in enumerate(self.bins) if x == i]
            self.bins_ixes[i] = bin_ixes
            self.num_samples[i] = len(bin_ixes)

    def __getitem__(self, idx: int):
        indexes = self.tokenized_indexes[idx][: self.max_length]
        indexes = torch.as_tensor(indexes, dtype=torch.int32)
        return indexes, self.word2ix[self.EOS]


class BinSampler(RandomSampler):
    def __init__(
        self, data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None, generator=None
    ) -> None:
        super().__init__(data_source, replacement, num_samples, generator)
        self.init_generators()

    def init_generators(self):
        self.generators = {}
        self.ns = {}
        for i in range(self.data_source.n_bins):
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.generators[i] = generator
            self.ns[i] = 0

    def __iter__(self) -> Iterator[int]:
        bin_numbers = np.random.randint(low=0, high=self.data_source.n_bins, size=self._num_samples)
        yield from bin_numbers.tolist()
        # generator = self.generators[bin_number]
        # ixes = self.data_source.bins_ixes[bin_number]
        # num_samples = self.data_source.num_samples[bin_number]
        # n = self.data_source.num_samples[bin_number]
        # for _ in range(self.num_samples // n):
        #     yield from torch.randperm(n, generator=generator).tolist()
        # yield from torch.randperm(n, generator=generator).tolist()[: self.num_samples % n]


class BinBatchSampler(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.current_ns = {i: 0 for i in range(self.sampler.data_source.n_bins)}

    def __iter__(self) -> Iterator[List[int]]:
        for bin_number in self.sampler:
            # if self.current_ns[bin_number] < self.sampler.data_source.num_samples[bin_number]:
            the_batch = self.sampler.data_source.bins_ixes[bin_number][
                self.current_ns[bin_number] : self.current_ns[bin_number] + self.batch_size
            ]
            self.current_ns[bin_number] += len(the_batch)
            # if len(the_batch) == self.batch_size:
            if not the_batch:
                continue
            yield the_batch
        # batch = [0] * self.batch_size
        # idx_in_batch = 0
        # for idx in self.sampler:
        #     batch[idx_in_batch] = idx
        #     idx_in_batch += 1
        #     if idx_in_batch == self.batch_size:
        #         yield batch
        #         idx_in_batch = 0
        #         batch = [0] * self.batch_size
        # if idx_in_batch > 0:
        #     yield batch[:idx_in_batch]


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    pass


if __name__ == "__main__":
    the_path = "wikitext-103/wiki.test.tokens"

    ds_1 = BrainDataset(the_path)
    ds_2 = BigBrainDataset(the_path)
    ds_3 = UltraDuperBigBrainDataset(the_path, n_bins=10)

    # print(ds.bins)
    # print(len([x for x in ds.bins if x == 2]))
    # print(len([x for x in ds.bins if x == 1]))
    # print(len([x for x in ds.bins if x == 0]))

    # %%
    batch_size = 16
    lls_1 = []
    loader_1 = DataLoader(ds_1, batch_size=16)
    for x in loader_1:
        lls_1.append(x.shape[1])
    # %%
    lls_2 = []
    loader_2 = DataLoader(ds_2, batch_size=16, collate_fn=big_brain_collate_fn)
    for x in loader_2:
        lls_2.append(x.shape[1])

    # %%
    lls_3 = []
    ss = BinBatchSampler(
        BinSampler(ds_3, replacement=False, num_samples=(len(ds_3) // batch_size) + ds_3.n_bins * 10),
        batch_size=batch_size,
        drop_last=False,
    )
    loader = DataLoader(ds_3, collate_fn=big_brain_collate_fn, batch_sampler=ss)
    for x in loader:
        lls_3.append(x.shape[1])

    # %%

    for x in lls_3:
        print(x)
    print(len(lls_1), len(lls_2), len(lls_3))

    # %%
    import matplotlib.pyplot as plt

    plt.hist(lls_1, bins=20)

    # %%
    import matplotlib.pyplot as plt

    plt.hist(lls_2, bins=20)

    # %%
    import matplotlib.pyplot as plt

    plt.hist(lls_3, bins=20)

    # %%
