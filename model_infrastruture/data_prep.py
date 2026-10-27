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


# =========== Track signals stuff ==============

import pybigtools
import numpy as np

from model_infrastruture.genome import cpg_island_track


ALL_TRACKS = [
    "DNase",
    "H3K4me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "CpG",  # CpG islands, computed from the genome sequence (see genome.py)
    
    # long-range context of each mark: ~10 kb around the TSS in 104 bp bins
    "DNase@10kb",
    "H3K4me3@10kb",
    "H3K4me1@10kb",
    "H3K27ac@10kb",
    "H3K36me3@10kb",
#    "H3K27me3",
#    "H3K9me3",
]

# tracks computed from the genome sequence instead of read from a bigWig. They are the same in
# every cell line and already lie in 0..1, so they skip the log1p and the per-cell z-score.
GENOME_TRACKS = {"CpG"}

# i would say upstream is more informative in regrards to regulatory aspect, other methods shift their windows to include more upstream as well
UPSTREAM_WINDOW = 2048  # multiples of BIN_SIZE so bins are exact
DOWNSTREAM_WINDOW = 1024

BIN_SIZE = 16
N_BINS = (UPSTREAM_WINDOW + DOWNSTREAM_WINDOW) // BIN_SIZE

# long-range tracks ("<mark>@10kb"): ~10 kb centred on the TSS, read directly in N_BINS bins of
# LONG_BIN_SIZE bp so they stack with the other tracks (96 * 104 = 9984 bp, i.e. +-4992 bp)
LONG_SUFFIX = "@10kb"
LONG_BIN_SIZE = 104
LONG_WINDOW = N_BINS * LONG_BIN_SIZE


def tss_features(genes, cell_line, tracks=ALL_TRACKS):
    # Mean bigWig signal in [TSS - WINDOW, TSS + WINDOW] for each track. One row per gene.
    # plain bigWig tracks only
    tracks = [t for t in tracks if t not in GENOME_TRACKS and not t.endswith(LONG_SUFFIX)]
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


def _genome_track(track, chroms, tss, minus):
    """Windows of a genome-derived track, laid out exactly like the bigWig windows."""
    assert track == "CpG", f"unknown genome track {track}"
    rows = []
    for chrom, pos, neg in zip(chroms, tss, minus):
        up, down = (
            (DOWNSTREAM_WINDOW, UPSTREAM_WINDOW)
            if neg
            else (UPSTREAM_WINDOW, DOWNSTREAM_WINDOW)
        )
        v = cpg_island_track(chrom, int(pos) - up, int(pos) + down, BIN_SIZE)
        rows.append(v[::-1] if neg else v)  # flip so upstream is always on the left
    return np.stack(rows)


def tss_window_tracks(gene_info, cell_line, tracks=ALL_TRACKS):
    """get actual sequence resolved track signal over multiple tracks (as opposed to mean summary)"""

    tss = ((gene_info["TSS_start"] + gene_info["TSS_end"]) // 2).to_numpy()
    minus = (gene_info["strand"] == "-").to_numpy()
    windows = {}

    for track in tracks:
        if track in GENOME_TRACKS:
            windows[track] = _genome_track(track, gene_info["chr"], tss, minus)
            continue
        long_range = track.endswith(LONG_SUFFIX)
        bw = pybigtools.open(find_track(track.removesuffix(LONG_SUFFIX), cell_line, "bigwig"))
        rows = []
        for chrom, pos, neg in zip(gene_info["chr"], tss, minus):
            if long_range:
                v = bw.values(
                    chrom,
                    start=max(int(pos) - LONG_WINDOW // 2, 0),
                    end=int(pos) + LONG_WINDOW // 2,  # past chromosome end -> oob=0
                    bins=N_BINS,
                    summary="mean",
                    fillna=0,
                    oob=0,
                )
                rows.append(v[::-1] if neg else v)
                continue
            # upstream is to the right on the minus strand
            up, down = (
                (DOWNSTREAM_WINDOW, UPSTREAM_WINDOW)
                if neg
                else (UPSTREAM_WINDOW, DOWNSTREAM_WINDOW)
            )
            v = bw.values(
                chrom,
                start=max(int(pos) - up, 0),
                end=int(pos) + down,  # past chromosome end -> oob=0
                bins=N_BINS,
                summary="mean",
                fillna=0,
                oob=0,
            )
            rows.append(v[::-1] if neg else v)  # flip so upstream is always on the left
        windows[track] = np.stack(rows)

    return np.stack([windows[t] for t in tracks], axis=1)  # (n_genes, n_tracks, N_BINS)


_LOG_WINDOW_CACHE = {}  # (cell, split, tracks) -> log1p windows, shared by the train/val loaders


def _log_windows(cell, split, tracks):
    """log1p track windows (n_genes, n_tracks, N_BINS) of one cell line and split,
    in ALL_GENES order. log1p because the signal distribution is heavily skewed."""
    key = (cell, split, tuple(tracks))
    if key not in _LOG_WINDOW_CACHE:
        genes = ALL_GENES[(ALL_GENES["cell"] == cell) & (ALL_GENES["split"] == split)]
        w = tss_window_tracks(genes, cell, tracks).astype(np.float32)
        signal = [i for i, t in enumerate(tracks) if t not in GENOME_TRACKS]
        w[:, signal] = np.log1p(w[:, signal])  # genome tracks are already 0..1
        _LOG_WINDOW_CACHE[key] = w
    return _LOG_WINDOW_CACHE[key]


def _track_stats(cell, tracks):
    """Mean and std per track of one cell line, shape (1, n_tracks, 1), over all its genes and
    bins. Taken from the train genes (X3 has no train split, so from its test genes).
    No labels are involved."""
    has_train = ((ALL_GENES["cell"] == cell) & (ALL_GENES["split"] == "train")).any()
    w = _log_windows(cell, "train" if has_train else "test", tracks)
    mean, std = w.mean(axis=(0, 2), keepdims=True), np.maximum(w.std(axis=(0, 2), keepdims=True), 1e-6)
    for i, t in enumerate(tracks):
        if t in GENOME_TRACKS:  # identical in every cell line: leave it as it is
            mean[:, i], std[:, i] = 0.0, 1.0
    return mean, std


def make_loader(
    split, tracks=ALL_TRACKS, cells=("X1", "X2"), batch_size=128, shuffle=True, normalize=True
):
    """normalize: z-score every track within its cell line (after log1p). Whatever was used
    for training has to be used for prediction too."""

    genes = ALL_GENES[(ALL_GENES["cell"].isin(cells)) & (ALL_GENES["split"] == split)]

    X = np.empty(shape=(len(genes), len(tracks), N_BINS), dtype=np.float32)

    for cell in cells:
        mask = (genes["cell"] == cell).to_numpy()
        W = _log_windows(cell, split, tracks)
        if normalize:
            mean, std = _track_stats(cell, tracks)
            W = (W - mean) / std
        X[mask] = W

    X = torch.from_numpy(X)

    if split == "test":
        return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    y = torch.from_numpy(np.log1p(genes["gex"].to_numpy(np.float32)))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")
