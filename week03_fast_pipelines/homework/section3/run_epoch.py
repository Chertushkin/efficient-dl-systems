import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from timeit import default_timer

import dataset
from utils import Settings
from vit_old import ViT as SubOptimalViT
from torch.profiler import profile, record_function, ProfilerActivity

EPOCHS = 5


def get_vit_model() -> torch.nn.Module:
    model = SubOptimalViT(
        dim=128,
        mlp_dim=128,
        depth=12,
        heads=8,
        image_size=224,
        patch_size=32,
        num_classes=2,
        channels=3,
    ).to(Settings.device)
    return model


def get_train_loader() -> torch.utils.data.DataLoader:
    train_list = dataset.extract_dataset_globs(half=False)
    print(f"Train Data: {len(train_list)}")
    train_transforms = dataset.get_train_transforms()
    train_data = dataset.CatsDogsDataset(train_list, transform=train_transforms)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=Settings.batch_size,
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader


def run_epoch(model, train_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_training"):
    for i, (data, label) in tqdm(enumerate(train_loader), desc=f"[Train]"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        # print(acc)
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        # if i>=2:
        #     break

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    return epoch_loss, epoch_accuracy


def run_epoch_tb(model, train_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/vit_old_final"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i, (data, label) in tqdm(enumerate(train_loader), desc=f"[Train]"):
            if i >= (1 + 1 + 3) * 2:
                break

            data = data.to(Settings.device)
            label = label.to(Settings.device)
            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            # print(acc)
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

    return epoch_loss, epoch_accuracy


def main():
    model = get_vit_model()
    train_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)
    start = default_timer()
    # for epoch in range(EPOCHS):
    epoch_loss, epoch_accuracy = run_epoch_tb(model, train_loader, criterion, optimizer)
    end = default_timer()
    print(f"Took {end - start} seconds...")
    print(f"{epoch_loss = }, {epoch_accuracy = }")


if __name__ == "__main__":
    main()
