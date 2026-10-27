import os
import sys

import pybigtools
import pandas as pd
import torch

import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.data import Dataset, DataLoader


from model_infrastruture.data_prep import (
    ALL_TRACKS,
    BIN_SIZE,
    DOWNSTREAM_WINDOW,
    N_BINS,
    UPSTREAM_WINDOW,
    make_loader,
)
from model_infrastruture.run_log import RunLogger, describe_run

from models.model import ConvResNet


def train(
    model: nn.Module,
    loss_function,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs=5,
    logger: RunLogger | None = None,
    scheduler=None,  # stepped once per epoch
):

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs).squeeze(-1)  # (B, 1) -> (B,) to match labels
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        msg = f"epoch {epoch + 1}: lr {optimizer.param_groups[0]['lr']:.2e} | train loss {train_loss:.3f}"
        val_loss = val_rho = None
        if val_loader:
            val_loss, val_rho = evaluate(model, loss_function, val_loader)
            msg += f" | val loss {val_loss:.3f} | val spearman {val_rho:.3f}"
        print(msg)
        if logger:
            logger.log_epoch(epoch + 1, train_loss, val_loss, val_rho)
        if scheduler:
            scheduler.step()
    print("Finished Training")


def save_checkpoint(model: nn.Module, path: str, model_config: dict, tracks: list[str],
                    normalize: bool = False):
    """Save weights plus everything predict.py needs to rebuild the model and its inputs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": type(model).__name__,  # key into models.model.MODELS
            "model_config": model_config,  # kwargs for that class
            "tracks": list(tracks),  # channel order the model was trained on
            "normalize": normalize,  # whether inputs were z-scored per cell line and track
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


@torch.no_grad()
def evaluate(model: nn.Module, loss_function, dataloader: DataLoader):
    """Mean loss and Spearman correlation over the whole loader, usually the val set"""
    model.eval()
    preds, labels = [], []
    for inputs, y in dataloader:
        preds.append(model(inputs).squeeze(-1))
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    rho = (
        pd.Series(preds.numpy()).rank().corr(pd.Series(labels.numpy()).rank())
    )  # spearman = pearson on ranks
    return loss_function(preds, labels).item(), rho


CHECKPOINT_PATH = os.path.join("checkpoints", "model.pt")

if __name__ == '__main__':

    NORMALIZE = True  # z-score each track within its cell line
    train_loader = make_loader(split='train', normalize=NORMALIZE)
    val_loader = make_loader(split='val', shuffle=False, normalize=NORMALIZE)

    model_config = dict(in_channels=len(ALL_TRACKS), n_blocks=3, dropout=0.5,
                        pooling="flatten", n_bins=N_BINS,
                        strides=(1, 2, 2))  # downsample 4x instead of 8x; dropout now only in the head
    model = ConvResNet(**model_config)
    n_epochs = 20
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)  # weight_decay was 1e-2
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)  # lr 1e-3 -> 0 over the run
    loss_function = nn.HuberLoss(delta=1.0)

    # optional: python -m model_infrastruture.train "what I changed and why"
    note = " ".join(sys.argv[1:])
    config = describe_run(
        model, model_config, optimizer, loss_function, train_loader, n_epochs,
        data=dict(tracks=ALL_TRACKS, upstream=UPSTREAM_WINDOW,
                  downstream=DOWNSTREAM_WINDOW, bin_size=BIN_SIZE, normalize=NORMALIZE),
        scheduler=f"{type(scheduler).__name__}(T_max={n_epochs})",
        # anything else you change (seed=...) goes here so it gets logged
    )
    with RunLogger(config, note=note) as run_log:
        train(model=model,
              optimizer=optimizer,
              loss_function=loss_function,
              train_loader=train_loader,
              val_loader=val_loader,
              n_epochs=n_epochs,
              logger=run_log,
              scheduler=scheduler,
              )
        save_checkpoint(model, CHECKPOINT_PATH, model_config, ALL_TRACKS, NORMALIZE)
