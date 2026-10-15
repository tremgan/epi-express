from typing import Any

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
