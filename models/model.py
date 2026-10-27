from typing import Any

import torch
import torch.nn as nn



class ConvTower(nn.Module):
    def __init__(self, in_channels=7, kernel_size=5, n_layers=3, stride=3, filter_factor=3) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(n_layers):
            out_channels = channels * filter_factor
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )
            channels = out_channels
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvAttPoolModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self):
        pass


class ConvMaxPoolFeedForward(nn.Module):
    """1d conv until and maxpool along length dimension to get a fixed length vector as input to a borring mlp"""
    def __init__(self, in_channels=7, hidden=64, **tower_kwargs: Any) -> None:
        super().__init__()
        self.tower = ConvTower(in_channels, **tower_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.tower.out_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                      # x: (B, tracks, bins)
        x = self.tower(x)                      # (B, C, L')
        x = x.amax(dim=-1)                     # max over length -> (B, C)
        return self.mlp(x)                     # (B, 1)


class ResidualBlock(nn.Module):
    """Basic block: conv-BN-GELU-conv-BN + skip, then GELU. No dropout in the convolutions.
    A 1x1 conv on the skip path handles channel changes / downsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class ConvResNet(nn.Module):
    """1d ResNet: stem conv -> n_blocks residual blocks (channels double every block, the length
    shrinks by strides[i] in block i) -> MLP head with BatchNorm and dropout.
    Dropout is only used in the head, not in the convolutions.

    strides:  one stride per block, default 2 for every block. (1, 2, 2) downsamples 4x in total.
    pooling="global":  avg+max pooling over length. Position-invariant: the head cannot tell
                       where along the window a signal sits.
    pooling="flatten": flatten the final (channels x length) map, so the head sees the position
                       of features relative to the TSS. Needs n_bins (the input length).
    hidden_dims: widths of the head's hidden layers, each Linear-BatchNorm-GELU-Dropout."""
    def __init__(self, in_channels=7, base_channels=32, n_blocks=3, kernel_size=5,
                 dropout=0.1, hidden_dims=(512, 128), pooling="global", n_bins=96,
                 strides=None) -> None:
        super().__init__()
        assert pooling in ("global", "flatten"), pooling
        strides = tuple(strides) if strides is not None else (2,) * n_blocks
        assert len(strides) == n_blocks, "need one stride per block"
        self.pooling = pooling
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        blocks, channels, length = [], base_channels, n_bins
        for stride in strides:
            blocks.append(ResidualBlock(channels, channels * 2, kernel_size, stride=stride))
            channels *= 2
            length = (length + 2 * (kernel_size // 2) - kernel_size) // stride + 1
        self.blocks = nn.Sequential(*blocks)

        width = channels * length if pooling == "flatten" else 2 * channels
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):                      # x: (B, tracks, bins)
        x = self.blocks(self.stem(x))          # (B, C, L')
        if self.pooling == "flatten":
            x = x.flatten(1)                   # (B, C * L')
        else:
            x = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)  # (B, 2C)
        return self.head(x)                    # (B, 1)


class LargeMLP(nn.Module):
    """Flatten the whole (tracks x bins) window and run it through a wide, deep MLP."""
    def __init__(self, in_channels=7, n_bins=96, hidden_dims=(1024, 1024, 512, 256), dropout=0.2) -> None:
        super().__init__()
        layers, width = [nn.Flatten()], in_channels * n_bins
        for h in hidden_dims:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                      # x: (B, tracks, bins)
        return self.net(x)                     # (B, 1)


# name -> class, so checkpoints can be rebuilt by predict.py
MODELS = {cls.__name__: cls for cls in (ConvMaxPoolFeedForward, ConvResNet, LargeMLP)}
