import pybigtools
import pandas as pd
import torch

import torch.nn as nn

from torch.optim import SGD


from torch.utils.data import Dataset, DataLoader


from model_infrastruture.data_prep import make_loader

from models.model import ConvMaxPoolFeedForward


def train(
    model: nn.Module,
    loss_function,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs=5,
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
    for inputs, y in dataloader:
        preds.append(model(inputs).squeeze(-1))
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    rho = (
        pd.Series(preds.numpy()).rank().corr(pd.Series(labels.numpy()).rank())
    )  # spearman = pearson on ranks
    return loss_function(preds, labels).item(), rho


if __name__ == '__main__':

    train_loader = make_loader(split='train')
    val_loader = make_loader(split='val', shuffle=False)

    model = ConvMaxPoolFeedForward(in_channels=7)
    train(model=model,
          optimizer=SGD(model.parameters(), lr=1e-2, momentum=0.9),
          loss_function=nn.MSELoss(),
          train_loader=train_loader,
          val_loader=val_loader,
          )