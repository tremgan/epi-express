import pybigtools
import pandas as pd
import torch

import torch.nn as nn

from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.data import Dataset, DataLoader


from model_infrastruture.data_prep import make_loader, N_BINS, N_CHANNELS, ALL_TRACKS

from models.model import make_model

MODEL_NAME = "resmlp"  # "resmlp" = deep ResNet + flatten + MLP, "resatt" = deep ResNet + attention pooling, "att" = plain conv + attention pooling, "maxpool" = conv + max-pool + MLP

# Training-time augmentation (never used for val/test). Set a strength to 0 to switch that part off.
AUGMENT = True
AUG_GAIN = 0.1       # per-gene, per-track scale jitter: x * (1 + N(0, AUG_GAIN))
AUG_OFFSET = 0.1     # per-gene, per-track shift jitter (inputs are z-scored, so ~0.1 sd)
AUG_P_DROP = 0.1     # probability of zeroing a whole track ("average signal") for a gene
AUG_MAX_SHIFT = 2    # whole window moved by up to +-this many bins (one shift per batch)


def shift_bins(x, k):
    """Move the window k bins along the genome (k > 0: content moves right); gaps become 0."""
    if k == 0:
        return x
    out = torch.zeros_like(x)
    if k > 0:
        out[..., k:] = x[..., :-k]
    else:
        out[..., :k] = x[..., -k:]
    return out


def augment(x, n_tracks=len(ALL_TRACKS)):
    """x: (B, channels, bins). Only the first n_tracks channels (chromatin tracks) get the
    gain/offset/dropout; the sequence channels (GC, CpG) are the same in every cell line,
    so they only take part in the shift, which keeps them aligned with the tracks."""
    B = x.size(0)
    t = x[:, :n_tracks]
    if AUG_GAIN > 0:
        t = t * (1 + AUG_GAIN * torch.randn(B, n_tracks, 1))
    if AUG_OFFSET > 0:
        t = t + AUG_OFFSET * torch.randn(B, n_tracks, 1)
    if AUG_P_DROP > 0:
        t = t * (torch.rand(B, n_tracks, 1) >= AUG_P_DROP)
    x = torch.cat([t, x[:, n_tracks:]], dim=1)
    if AUG_MAX_SHIFT > 0:
        x = shift_bins(x, int(torch.randint(-AUG_MAX_SHIFT, AUG_MAX_SHIFT + 1, (1,))))
    return x


def train(
    model: nn.Module,
    loss_function,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    n_epochs=20,
    lr_decay=True,
    augment_fn=None,
):
    # cosine decay: lr goes from its initial value to ~0 over n_epochs, stepped once per epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs) if lr_decay else None

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            if augment_fn is not None:  # training batches only; evaluate() never augments
                inputs = augment_fn(inputs)

            # forward + backward + optimize

            outputs = model(inputs).squeeze(-1)  # (B, 1) -> (B,) to match labels
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        msg = f"epoch {epoch + 1}: train loss {running_loss / len(train_loader):.3f}"
        msg += f" | lr {optimizer.param_groups[0]['lr']:.2e}"
        if scheduler:
            scheduler.step()
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
    val_loader = make_loader(split='val', shuffle=False, stats=train_loader.stats)

    model = make_model(MODEL_NAME, N_BINS, N_CHANNELS)
    if MODEL_NAME in ("att", "resatt", "resmlp"):  # these train much better with Adam-style optimizers
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    else:
        optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    train(model=model,
          optimizer=optimizer,
          loss_function=nn.L1Loss(),  # MAE: less sensitive to outlier genes than MSE
          train_loader=train_loader,
          val_loader=val_loader,
          augment_fn=augment if AUGMENT else None,
          )
    torch.save(model.state_dict(), f"model_{MODEL_NAME}.pt") 
    print(f"Saved weights to model_{MODEL_NAME}.pt")