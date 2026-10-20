import pybigtools
import pandas as pd
import torch

import torch.nn as nn

from torch.optim import SGD, AdamW


from torch.utils.data import Dataset, DataLoader


from model_infrastruture.data_prep import make_loader

from models.model import ConvMaxPoolFeedForward, ConvAttPoolModel, TwoScaleModel


def train(
    model: nn.Module,
    loss_function,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs=15,
):

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for *inputs, labels in train_loader:  # inputs: [fine] or [fine, coarse]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            outputs = model(*inputs).squeeze(-1)  # (B, 1) -> (B,) to match labels
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        msg = f"epoch {epoch + 1}: train loss {running_loss / len(train_loader):.3f}"
        if val_loader:
            val_loss, val_rho = evaluate(model, loss_function, val_loader)
            msg += f" | val loss {val_loss:.3f} | val spearman {val_rho:.3f}"
        print(msg)
    print("Finished Training")


@torch.no_grad()
def evaluate(model: nn.Module, loss_function, dataloader: DataLoader):
    """Mean loss and Spearman correlation over the whole loader, usually the val set"""
    model.eval()
    preds, labels = [], []
    for *inputs, y in dataloader:
        preds.append(model(*inputs).squeeze(-1))
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    rho = (
        pd.Series(preds.numpy()).rank().corr(pd.Series(labels.numpy()).rank())
    )  # spearman = pearson on ranks
    return loss_function(preds, labels).item(), rho


if __name__ == '__main__':

    # two-scale input: fine window (promoter) + coarse window (distal), see data_prep.py
    train_loader = make_loader(split='train', coarse=True)
    val_loader = make_loader(split='val', shuffle=False, coarse=True)

    model = TwoScaleModel(in_channels=7)  # dropout 0.2 in conv blocks, 0.3 in the MLP
    train(model=model,
          optimizer=AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2),
          loss_function=nn.MSELoss(),
          train_loader=train_loader,
          val_loader=val_loader,
          )

    # single-scale alternatives (use make_loader without coarse=True):
    # model = ConvAttPoolModel(in_channels=7)  /  ConvMaxPoolFeedForward(in_channels=7)
    # optimizer=SGD(model.parameters(), lr=1e-2, momentum=0.9)
