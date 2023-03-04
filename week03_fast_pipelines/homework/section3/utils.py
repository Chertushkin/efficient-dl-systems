import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 595959) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    return device


class Settings:
    batch_size: int = 128
    epochs: int = 2
    lr: float = 3e-5
    gamma: float = 0.7
    seed: int = 42
    device: str = get_device()


class CatsAndDogs:
    directory = "/home/misha/Coursera/efficient-dl-systems-2022/week03_fast_pipelines/homework/section3/data"
    train_dir = "/home/misha/Coursera/efficient-dl-systems-2022/week03_fast_pipelines/homework/section3/data/train"
    test_dir = "/home/misha/Coursera/efficient-dl-systems-2022/week03_fast_pipelines/homework/section3/data/test"
    regexp = "*.jpg"
