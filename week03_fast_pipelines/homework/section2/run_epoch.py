from enum import Enum
from dataset import BrainDataset
import torch


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model() -> torch.nn.Module:
    pass


def run_epoch(data_mode: DataMode) -> None:
    df = None
    if data_mode == DataMode.BRAIN:
        ds = BrainDataset("wikitext-103/wiki.test.tokens")
    else:
        raise ValueError("Not implemented")
        
run_epoch(DataMode.BRAIN)
