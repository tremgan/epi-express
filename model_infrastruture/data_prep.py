import pandas as pd
import glob
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model_infrastruture.config import DATA_DIR, ALL_TRACKS, BATCH_SIZE, WindowSettings

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


def tss_features(genes, cell_line, window=WindowSettings(), tracks=ALL_TRACKS):
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
                    max(int(pos) - window.upstream, 0),
                    min(int(pos) + window.downstream, sizes[chrom]),
                    bins=1,
                    summary="mean",
                    fillna=0,
                )[0]
                for chrom, pos in zip(genes["chr"], tss)
            ]
        )
    return pd.DataFrame(feats, index=genes.index)


def tss_window_tracks(gene_info, cell_line, window=WindowSettings(), tracks=ALL_TRACKS):
    """get actual sequence resolved track signal over multiple tracks (as opposed to mean summary)"""

    tss = ((gene_info["TSS_start"] + gene_info["TSS_end"]) // 2).to_numpy()
    minus = (gene_info["strand"] == "-").to_numpy()
    windows = {}

    for track in tracks:
        bw = pybigtools.open(find_track(track, cell_line, "bigwig"))
        rows = []
        for chrom, pos, neg in zip(gene_info["chr"], tss, minus):
            # upstream is to the right on the minus strand
            up, down = (
                (window.downstream, window.upstream)
                if neg
                else (window.upstream, window.downstream)
            )
            v = bw.values(
                chrom,
                start=max(int(pos) - up, 0),
                end=int(pos) + down,  # past chromosome end -> oob=0
                bins=window.n_bins,
                summary="mean",
                fillna=0,
                oob=0,
            )
            rows.append(v[::-1] if neg else v)  # flip so upstream is always on the left
        windows[track] = np.stack(rows)

    return np.stack([windows[t] for t in tracks], axis=1)  # (n_genes, n_tracks, n_bins)


def make_loader(
    split,
    window=WindowSettings(),
    tracks=ALL_TRACKS,
    cells=("X1", "X2"),
    batch_size=BATCH_SIZE,
    shuffle=True,
):

    genes = ALL_GENES[(ALL_GENES["cell"].isin(cells)) & (ALL_GENES["split"] == split)]

    X = np.empty(shape=(len(genes), len(tracks), window.n_bins), dtype=np.float32)

    for cell in cells:
        mask = (genes["cell"] == cell).to_numpy()
        X[mask] = tss_window_tracks(genes[mask], cell, window, tracks)

    X = torch.from_numpy(
        np.log1p(X)
    )  # distribution is heavilly skewed, hence logarithmized

    if split == "test":
        return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    y = torch.from_numpy(np.log1p(genes["gex"].to_numpy(np.float32)))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")
