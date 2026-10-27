from typing import Any

import torch
import torch.nn as nn



class ConvTower(nn.Module):
    def __init__(self, in_channels=7, kernel_size=5, n_layers=3, stride=3, filter_factor=3) -> None:
        super().__init__()

        self.kernel_size, self.stride, self.padding, self.n_layers = kernel_size, stride, 2, n_layers
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(n_layers):
            out_channels = channels * filter_factor
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                )
            )
            channels = out_channels
        self.out_channels = channels

    def out_length(self, n_bins: int) -> int:
        """Number of positions left after the tower for an input with n_bins bins."""
        for _ in range(self.n_layers):
            n_bins = (n_bins + 2 * self.padding - self.kernel_size) // self.stride + 1
        return n_bins

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Two convs with BatchNorm plus a skip connection: out = GELU(shortcut(x) + body(x)).
    With stride 2 the length halves; the skip then uses a 1x1 conv so the shapes still match."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch)
            )
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.shortcut(x) + self.body(x))


class ResTower(nn.Module):
    """Deep residual conv tower: stem conv (k=7) -> stages of ResBlocks. The first stage keeps the
    length, every later stage halves it. Defaults: 4 stages x 2 blocks = 16 convs + stem, and
    200 bins -> 200 -> 100 -> 50 -> 25 positions, 128 channels at the end."""

    def __init__(self, in_channels=7, widths=(32, 64, 128, 128), blocks_per_stage=2, kernel_size=5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.GELU(),
        )
        blocks, ch = [], widths[0]
        for i, w in enumerate(widths):
            for b in range(blocks_per_stage):
                stride = 2 if (i > 0 and b == 0) else 1
                blocks.append(ResBlock(ch, w, stride, kernel_size))
                ch = w
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = ch
        self.n_halvings = len(widths) - 1

    def out_length(self, n_bins: int) -> int:
        for _ in range(self.n_halvings):
            n_bins = (n_bins - 1) // 2 + 1
        return n_bins

    def forward(self, x):
        return self.blocks(self.stem(x))


class AttentionPool(nn.Module):
    """Pools (B, L, D) -> (B, D) with one learned query that attends over the L positions."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.weights = None  # (B, L) attention over positions from the last forward, for inspection

    def forward(self, x):
        x = self.norm(x)
        q = self.query.expand(x.size(0), -1, -1)
        out, w = self.attn(q, x, x, need_weights=True)  # w: (B, 1, L), averaged over heads
        self.weights = w.squeeze(1).detach()
        return out.squeeze(1)


class ConvAttPoolModel(nn.Module):
    """Conv tower -> 1x1 projection -> + learned position embedding -> (optional transformer
    layers) -> attention pooling -> linear. n_bins must match the input length (data_prep.N_BINS)."""

    def __init__(
        self,
        in_channels=7,
        n_bins=96,
        d_model=64,
        n_heads=4,
        n_enc_layers=0,
        dropout=0.1,
        tower: nn.Module | None = None,
        **tower_kwargs: Any,
    ) -> None:
        super().__init__()
        # any module with .out_channels and .out_length(n_bins); default is the plain conv tower
        self.tower = tower if tower is not None else ConvTower(in_channels, **tower_kwargs)
        self.n_tokens = self.tower.out_length(n_bins)
        self.proj = nn.Conv1d(self.tower.out_channels, d_model, kernel_size=1)
        # one learned vector per position; positions run from upstream (left) to downstream (right)
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        self.encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=n_enc_layers,
            )
            if n_enc_layers > 0
            else nn.Identity()
        )
        self.pool = AttentionPool(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                      # x: (B, tracks, bins)
        x = self.tower(x)                      # (B, C, L')
        x = self.proj(x).transpose(1, 2)       # (B, L', d_model): one token per position
        x = self.encoder(x + self.pos)         # position-aware tokens
        x = self.pool(x)                       # (B, d_model)
        return self.head(self.drop(x))         # (B, 1)


class ResMLPModel(nn.Module):
    """Deep residual conv tower -> flatten -> MLP -> 1. The flatten layer gives every position its
    own weights, so the head knows where along the window (upstream/downstream) a signal sits.
    n_bins must match the input length (data_prep.N_BINS)."""

    def __init__(self, in_channels=7, n_bins=200, hidden=(256, 128), dropout=0.2, **tower_kwargs: Any) -> None:
        super().__init__()
        self.tower = ResTower(in_channels, **tower_kwargs)
        self.n_tokens = self.tower.out_length(n_bins)
        layers, prev = [nn.Flatten()], self.tower.out_channels * self.n_tokens
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):                      # x: (B, channels, bins)
        return self.head(self.tower(x))        # (B, 1)


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


def make_model(name: str, n_bins: int, in_channels: int = 7) -> nn.Module:
    """Single place that builds the model, so train.py and predict.py always agree."""
    if name == "att":
        return ConvAttPoolModel(in_channels=in_channels, n_bins=n_bins)
    if name == "maxpool":
        return ConvMaxPoolFeedForward(in_channels=in_channels)
    if name == "resatt":  # deep residual tower + the same position embedding and attention pooling
        return ConvAttPoolModel(in_channels=in_channels, n_bins=n_bins, tower=ResTower(in_channels))
    if name == "resmlp":  # deep residual tower + flatten + MLP head
        return ResMLPModel(in_channels=in_channels, n_bins=n_bins)
    raise ValueError(f"unknown model {name!r}")
