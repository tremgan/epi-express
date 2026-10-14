from typing import Any

import torch.nn as nn



class ConvTower(nn.Module):
    def __init__(self, in_channels, start_n_filters=10, kernel_size=5, n_layers=3, stride=3, filter_factor=3) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(n_layers):
            out_channel = start_n_filters * (filter_factor ** i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channel, out_channel, kernel_size=kernel_size, stride=stride, padding=2),
                    nn.BatchNorm1d(out_channel),
                    nn.GELU(),
                )
            )
            channel = out_channel
        self.out_channels = channel

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvAttPoolModel(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self):
        pass
