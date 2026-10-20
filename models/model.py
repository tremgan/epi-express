from typing import Any

import torch
import torch.nn as nn



class ConvTower(nn.Module):
    """Stack of Conv1d -> BatchNorm -> GELU blocks.

    Default (strides/dilations/channels all None) is the original tower: n_layers blocks with the same
    stride and widths multiplied by filter_factor each layer (7 -> 21 -> 63 -> 189, length 96 -> 32 -> 11 -> 4).

    For deeper towers that keep more positions, pass per-layer lists (they set the number of layers):
      strides   e.g. (1, 2, 1, 2, 1, 2)  -> length 96 -> 96 -> 48 -> 48 -> 24 -> 24 -> 12
      dilations e.g. (1, 2, 4, 8)        -> same length, but the receptive field grows exponentially
      channels  e.g. (32, 64, 64, 128)   -> explicit output width of each layer
    dropout > 0 adds channel-wise Dropout1d after every block.
    Padding is dilation * (kernel_size - 1) // 2, so stride 1 keeps the length (odd kernel sizes).
    """
    def __init__(self, in_channels=7, kernel_size=5, n_layers=3, stride=3, filter_factor=3,
                 strides=None, dilations=None, channels=None, dropout=0.0) -> None:
        super().__init__()

        given = [len(v) for v in (strides, dilations, channels) if v is not None]
        if given:
            n_layers = given[0]
            if any(g != n_layers for g in given):
                raise ValueError("strides, dilations and channels must have the same length")
        strides = list(strides) if strides is not None else [stride] * n_layers
        dilations = list(dilations) if dilations is not None else [1] * n_layers
        if channels is None:
            channels = [in_channels * filter_factor ** (i + 1) for i in range(n_layers)]
        channels = list(channels)

        self.layers = nn.ModuleList()
        prev = in_channels
        for out_channels, s, d in zip(channels, strides, dilations):
            block = [
                nn.Conv1d(prev, out_channels, kernel_size=kernel_size, stride=s,
                          padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ]
            if dropout > 0:
                block.append(nn.Dropout1d(dropout))
            self.layers.append(nn.Sequential(*block))
            prev = out_channels
        self.out_channels = prev

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionPool(nn.Module):
    """Learned weighted average over the length dimension: (B, C, L) -> (B, C).

    A small scorer gives every position a score from its C features, softmax turns the
    scores into weights that sum to 1 over the positions, and the output is the weighted
    sum of the position vectors. Unlike a max, it can blend information from several positions.
    """
    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                                   # x: (B, C, L)
        x = x.transpose(1, 2)                               # (B, L, C)
        attn = self.score(x).squeeze(-1).softmax(dim=-1)    # (B, L), sums to 1 over L
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)        # (B, C)
        return pooled, attn


class ConvAttPoolModel(nn.Module):
    """1d conv tower -> attention pooling along length -> mlp. Same interface as ConvMaxPoolFeedForward."""
    def __init__(self, in_channels=7, hidden=64, attn_hidden=32, **tower_kwargs: Any) -> None:
        super().__init__()
        self.tower = ConvTower(in_channels, **tower_kwargs)
        self.pool = AttentionPool(self.tower.out_channels, attn_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(self.tower.out_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, return_attn=False):               # x: (B, tracks, bins)
        x = self.tower(x)                                  # (B, C, L')
        x, attn = self.pool(x)                             # (B, C), (B, L')
        out = self.mlp(x)                                  # (B, 1)
        return (out, attn) if return_attn else out


class TwoScaleModel(nn.Module):
    """Fine (promoter) and coarse (distal) windows, each: ConvTower -> AttentionPool.
    The two pooled vectors are concatenated and fed to a dropout-regularised MLP.

    Use with make_loader(..., coarse=True): forward(fine, coarse) with shapes (B, T, 96) and (B, T, 160).
    Small by default: channels (32, 64, 64), strides (1, 2, 2) -> 96 -> 96 -> 48 -> 24 and 160 -> 160 -> 80 -> 40 positions.
    """
    def __init__(self, in_channels=7, hidden=64, attn_hidden=32, dropout=0.2, mlp_dropout=0.3,
                 **tower_kwargs: Any) -> None:
        super().__init__()
        cfg = dict(channels=(32, 64, 64), strides=(1, 2, 2), dropout=dropout)
        cfg.update(tower_kwargs)
        self.fine_tower = ConvTower(in_channels, **cfg)
        self.coarse_tower = ConvTower(in_channels, **cfg)
        self.fine_pool = AttentionPool(self.fine_tower.out_channels, attn_hidden)
        self.coarse_pool = AttentionPool(self.coarse_tower.out_channels, attn_hidden)
        width = self.fine_tower.out_channels + self.coarse_tower.out_channels
        self.mlp = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, fine, coarse, return_attn=False):
        f, f_attn = self.fine_pool(self.fine_tower(fine))          # (B, C), (B, L_fine)
        c, c_attn = self.coarse_pool(self.coarse_tower(coarse))    # (B, C), (B, L_coarse)
        out = self.mlp(torch.cat([f, c], dim=1))                   # (B, 1)
        return (out, f_attn, c_attn) if return_attn else out


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
