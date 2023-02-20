import torch
from torch import nn
from tqdm.auto import tqdm
import typing as tp
from timeit import default_timer
import numpy as np

from unet import Unet

from dataset import get_train_data


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.L1Loss,
    optimizer: tp.Type[torch.optim.Optimizer],
    device: torch.device,
) -> None:
    model.train()
    # print(model)
    # factor = 1
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            # loss *= factor

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # loss.backward()


            # ls = []
            # for p in model.parameters():
            #     ls.append(torch.norm(p.grad, 2).item())
            # grad_mean = torch.as_tensor(np.mean(ls))
            # if grad_mean < 1:
            #     factor = 1. / grad_mean
            # else:
            #     factor = 1

            # loss = loss * factor
            # loss.backward()
            # for p in model.parameters():
            #     p.grad = p.grad / factor

            # optimizer.step()

            optimizer.zero_grad()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train():
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 3
    start = default_timer()
    for _ in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device)
    end = default_timer()

    print(f"Took {end - start} seconds")


if __name__ == "__main__":
    train()
