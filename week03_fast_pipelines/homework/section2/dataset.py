from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchtext
from collections import Counter


MAX_LENGTH = 640


class BrainDataset(Dataset):
    UNK = "unk"
    EOS = "eos"

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.max_length = max_length

        lines = self.load_file(self.data_path)
        tokenized_lines = self.tokenize(lines)
        self.ix2word, self.word2ix = self.create_indexes(tokenized_lines)
        self.tokenized_indexes = self.tokenize_to_indexes(tokenized_lines)

    def __getitem__(self, idx: int):
        indexes = self.tokenized_indexes[idx][:self.max_length]
        indexes = torch.as_tensor(indexes)
        pad_size = self.max_length - len(indexes)
        indexes = F.pad(indexes, (0,pad_size), 'constant', self.word2ix[self.EOS])
        return indexes

    def tokenize_to_indexes(self, tokenized_lines):
        tokenized_indexes = [[self.word2ix.get(x, self.word2ix[self.UNK]) for x in row] for row in tokenized_lines]
        return tokenized_indexes

    def create_indexes(self, tokenized_lines):
        counter = Counter()
        for x in tokenized_lines:
            counter.update(x)
        ix2word = {}

        top_10k = counter.most_common(10000)
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


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


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


ds = BrainDataset("wikitext-103/wiki.test.tokens")

# print(ds[10])
