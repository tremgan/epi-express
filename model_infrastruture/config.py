from dataclasses import dataclass

import torch

# ---- paths (relative to the project root) ----
DATA_DIR = "data"
WORK_DIR = "."  # submission.csv is written here
CKPT_DIR = "checkpoints"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---- data ----
ALL_TRACKS = [
    "DNase",
    "H3K4me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K27me3",
    "H3K9me3",
]


@dataclass(frozen=True)
class WindowSettings:
    """bp around the TSS (strand-aware) and bin size for the track windows"""

    # i would say upstream is more informative in regrards to regulatory aspect, other methods shift their windows to include more upstream as well
    upstream: int = 1024
    downstream: int = 512
    bin_size: int = 16

    def __post_init__(self):
        if self.upstream % self.bin_size or self.downstream % self.bin_size:
            raise ValueError(f"upstream/downstream must be multiples of bin_size, got {self}")

    @property
    def n_bins(self):
        return (self.upstream + self.downstream) // self.bin_size


# ---- training defaults (CLI flags in train.py override these) ----
BATCH_SIZE = 64
LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 50
