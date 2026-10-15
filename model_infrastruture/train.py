import pybigtools
import pandas as pd
import torch
import typer

import torch.nn as nn

from torch.optim import SGD


from torch.utils.data import Dataset, DataLoader


from model_infrastruture.data_prep import WindowSettings, make_loader

from models.model import ConvMaxPoolFeedForward

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train(
    model: nn.Module,
    loss_function,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs=50,
):
    model.to(DEVICE)  # in place, so an optimizer built on model.parameters() still works
    print(f"training on {DEVICE}")
    history = []  # (epoch, val_loss, val_spearman) per epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(inputs).squeeze(-1)  # (B, 1) -> (B,) to match labels
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        msg = f"epoch {epoch + 1}: train loss {running_loss / len(train_loader):.3f}"
        if val_loader:
            val_loss, val_rho = evaluate(model, loss_function, val_loader)
            msg += f" | val loss {val_loss:.3f} | val spearman {val_rho:.3f}"
            history.append((epoch + 1, val_loss, val_rho))
        print(msg)
    print("Finished Training")
    return history


@torch.no_grad()
def evaluate(model: nn.Module, loss_function, dataloader: DataLoader):
    """Mean loss and Spearman correlation over the whole loader, usually the val set"""
    model.eval()
    preds, labels = [], []
    for inputs, y in dataloader:
        preds.append(model(inputs.to(DEVICE)).squeeze(-1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    rho = (
        pd.Series(preds.numpy()).rank().corr(pd.Series(labels.numpy()).rank())
    )  # spearman = pearson on ranks
    return loss_function(preds, labels).item(), rho


def main(
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    upstream: int = WindowSettings.upstream,
    downstream: int = WindowSettings.downstream,
    bin_size: int = WindowSettings.bin_size,
):
    """Train ConvMaxPoolFeedForward on X1+X2 train, validate on X1+X2 val."""
    window = WindowSettings(upstream, downstream, bin_size)
    print(window)
    train_loader = make_loader(split='train', window=window)
    val_loader = make_loader(split='val', window=window, shuffle=False)

    model = ConvMaxPoolFeedForward(in_channels=7)
    train(model=model,
          optimizer=SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
          loss_function=nn.MSELoss(),
          train_loader=train_loader,
          val_loader=val_loader,
          n_epochs=epochs,
          )


if __name__ == '__main__':
    typer.run(main)