from functools import lru_cache
import pandas as pd
import glob
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

DATA_DIR = "data"  # relative to the project root, where this notebook lives
WORK_DIR = "."  # submission.csv is written here

ALL_FILES = sorted(
    p
    for p in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
    if os.path.isfile(p)
)


def find_file(name):
    # Path of the unique file whose name is `name` (e.g. the CAGE-train *_info.tsv / *_y.tsv files).
    hits = [p for p in ALL_FILES if os.path.basename(p) == name]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 file named {name}, found {len(hits)}: {hits}"
        )
    return hits[0]


def find_track(mark, cell_line, kind):
    # Path of a chromatin track: data/<mark>-<kind>/<cell_line>.<ext>
    # kind: 'bed' (narrowPeak peaks) or 'bigwig' (continuous signal; extension is .bw or .bigwig).
    exts = (
        (".bed", ".bed.gz", ".narrowpeak", ".narrowpeak.gz")
        if kind == "bed"
        else (".bw", ".bigwig")
    )
    folder = os.path.join(DATA_DIR, f"{mark}-{kind}")
    hits = [
        p
        for p in glob.glob(os.path.join(folder, f"{cell_line}.*"))
        if p.lower().endswith(exts)
    ]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 {kind} file for {mark}/{cell_line}, found {len(hits)}: {hits}"
        )
    return hits[0]


def load_split(cell_line, split):
    info = pd.read_csv(find_file(f"{cell_line}_{split}_info.tsv"), sep="\t")
    if split == "test":
        return info
    y = pd.read_csv(find_file(f"{cell_line}_{split}_y.tsv"), sep="\t")
    return info.merge(y, on="gene_name", how="inner", validate="one_to_one")


def get_gene_info():
    """Gene tables keyed by (cell_line, split); train/val include `gex`, X3 test does not."""
    return {
        ("X1", "train"): load_split("X1", "train"),
        ("X1", "val"): load_split("X1", "val"),
        ("X2", "train"): load_split("X2", "train"),
        ("X2", "val"): load_split("X2", "val"),
        ("X3", "test"): load_split("X3", "test"),
    }


# all gene ids
# every gene row from every split, with `cell` and `split` columns (gex is NaN for X3 test)
ALL_GENES = (
    pd.concat(get_gene_info(), names=["cell", "split"])
    .reset_index(["cell", "split"])
    .reset_index(drop=True)
)

print(all_genes := ALL_GENES.head(3))  # show a few rows


# =========== Track signals stuff ==============

import pybigtools
import numpy as np


ALL_TRACKS = [
    "DNase",
    "H3K4me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K27me3",
    "H3K9me3",
]

# i would say upstream is more informative in regrards to regulatory aspect, other methods shift their windows to include more upstream as well
UPSTREAM_WINDOW = 1024  # fine scale: promoter resolution (multiples of BIN_SIZE so bins are exact)
DOWNSTREAM_WINDOW = 512
BIN_SIZE = 16
N_BINS = (UPSTREAM_WINDOW + DOWNSTREAM_WINDOW) // BIN_SIZE  # 96

COARSE_UPSTREAM_WINDOW = 10240  # coarse scale: distal context, used when make_loader(coarse=True)
COARSE_DOWNSTREAM_WINDOW = 10240
COARSE_BIN_SIZE = 128
COARSE_N_BINS = (COARSE_UPSTREAM_WINDOW + COARSE_DOWNSTREAM_WINDOW) // COARSE_BIN_SIZE  # 160

assert (UPSTREAM_WINDOW + DOWNSTREAM_WINDOW) % BIN_SIZE == 0
assert (COARSE_UPSTREAM_WINDOW + COARSE_DOWNSTREAM_WINDOW) % COARSE_BIN_SIZE == 0


def tss_features(genes, cell_line, tracks=ALL_TRACKS):
    # Mean bigWig signal in [TSS - WINDOW, TSS + WINDOW] for each track. One row per gene.
    tss = ((genes["TSS_start"] + genes["TSS_end"]) // 2).to_numpy()
    feats = {}
    for track in tracks:
        bw = pybigtools.open(find_track(track, cell_line, "bigwig"))
        sizes = bw.chroms()
        feats[track] = np.array(
            [
                bw.values(
                    chrom,
                    max(int(pos) - UPSTREAM_WINDOW, 0),
                    min(int(pos) + DOWNSTREAM_WINDOW, sizes[chrom]),
                    bins=1,
                    summary="mean",
                    fillna=0,
                )[0]
                for chrom, pos in zip(genes["chr"], tss)
            ]
        )
    return pd.DataFrame(feats, index=genes.index)


def tss_window_tracks(
    gene_info, cell_line, tracks=ALL_TRACKS, upstream=None, downstream=None, n_bins=None
):
    """get actual sequence resolved track signal over multiple tracks (as opposed to mean summary).

    Window sizes default to the fine scale (UPSTREAM_WINDOW / DOWNSTREAM_WINDOW / N_BINS).
    """
    upstream = UPSTREAM_WINDOW if upstream is None else upstream
    downstream = DOWNSTREAM_WINDOW if downstream is None else downstream
    n_bins = N_BINS if n_bins is None else n_bins

    tss = ((gene_info["TSS_start"] + gene_info["TSS_end"]) // 2).to_numpy()
    minus = (gene_info["strand"] == "-").to_numpy()
    windows = {}

    for track in tracks:
        bw = pybigtools.open(find_track(track, cell_line, "bigwig"))
        rows = []
        for chrom, pos, neg in zip(gene_info["chr"], tss, minus):
            # upstream is to the right on the minus strand
            up, down = (downstream, upstream) if neg else (upstream, downstream)
            v = bw.values(
                chrom,
                start=max(int(pos) - up, 0),
                end=int(pos) + down,  # past chromosome end -> oob=0
                bins=n_bins,
                summary="mean",
                fillna=0,
                oob=0,
            )
            rows.append(v[::-1] if neg else v)  # flip so upstream is always on the left
        windows[track] = np.stack(rows)

    return np.stack([windows[t] for t in tracks], axis=1)  # (n_genes, n_tracks, n_bins)


@lru_cache(maxsize=None)
def cell_windows(cell, tracks, upstream, downstream, n_bins):
    """log1p track windows for ALL genes of one cell line, in ALL_GENES order. Computed once per
    (cell, tracks, window) and cached, so train / val / stats share a single pass over the bigWigs."""
    genes = ALL_GENES[ALL_GENES["cell"] == cell]
    X = tss_window_tracks(genes, cell, list(tracks), upstream, downstream, n_bins)
    return np.log1p(X).astype(np.float32)  # distribution is heavilly skewed, hence logarithmized


@lru_cache(maxsize=None)
def cell_track_stats(cell, tracks, upstream, downstream, n_bins):
    """Mean and std of every track in one cell line (log1p scale), over all its genes and bins.

    Uses the inputs of ALL splits of that cell (no labels, so no leakage), which keeps
    train and val on the same scale. Returns two (1, n_tracks, 1) float32 arrays.
    """
    X = cell_windows(cell, tracks, upstream, downstream, n_bins)
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, np.maximum(std, 1e-6)  # guard against constant tracks


def _scale_tensor(genes, split, cells, tracks, upstream, downstream, n_bins, normalize):
    """(len(genes), n_tracks, n_bins) tensor for one window scale, rows aligned with `genes`."""
    tracks = tuple(tracks)
    X = np.empty(shape=(len(genes), len(tracks), n_bins), dtype=np.float32)
    for cell in cells:
        cell_genes = ALL_GENES[ALL_GENES["cell"] == cell]
        full = cell_windows(cell, tracks, upstream, downstream, n_bins)
        if normalize:  # z-score each track within each cell line (cells differ in depth / scale)
            mean, std = cell_track_stats(cell, tracks, upstream, downstream, n_bins)
            full = (full - mean) / std
        X[(genes["cell"] == cell).to_numpy()] = full[(cell_genes["split"] == split).to_numpy()]
    return torch.from_numpy(X)


def make_loader(
    split,
    tracks=ALL_TRACKS,
    cells=("X1", "X2"),
    batch_size=64,
    shuffle=True,
    normalize=True,
    coarse=False,
):
    """Batches are (fine, y), or (fine, coarse, y) with coarse=True. The test split has no y."""

    genes = ALL_GENES[(ALL_GENES["cell"].isin(cells)) & (ALL_GENES["split"] == split)]

    inputs = [
        _scale_tensor(genes, split, cells, tracks, UPSTREAM_WINDOW, DOWNSTREAM_WINDOW, N_BINS, normalize)
    ]
    if coarse:
        inputs.append(
            _scale_tensor(
                genes, split, cells, tracks,
                COARSE_UPSTREAM_WINDOW, COARSE_DOWNSTREAM_WINDOW, COARSE_N_BINS, normalize,
            )
        )

    if split == "test":
        return DataLoader(TensorDataset(*inputs), batch_size=batch_size, shuffle=False)
    y = torch.from_numpy(np.log1p(genes["gex"].to_numpy(np.float32)))
    return DataLoader(TensorDataset(*inputs, y), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")
