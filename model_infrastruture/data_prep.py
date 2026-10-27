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


ALL_TRACKS = [
    "DNase",
    "H3K4me3",
    "H3K27ac",
    "H3K36me3",
]

# i would say upstream is more informative in regrards to regulatory aspect, other methods shift their windows to include more upstream as well
UPSTREAM_WINDOW = 5000 # multiples of BIN_SIZE so bins are exact
DOWNSTREAM_WINDOW = 5000

BIN_SIZE = 50
N_BINS = (UPSTREAM_WINDOW + DOWNSTREAM_WINDOW) // BIN_SIZE

# Sequence-derived channels from the reference genome (hg38), binned like the tracks.
# Set USE_SEQ_FEATURES = False to go back to the 7 track channels only.
USE_SEQ_FEATURES = True
SEQ_FEATURES = ["GC", "CpG"]  # GC fraction, CpG dinucleotides per bp
N_CHANNELS = len(ALL_TRACKS) + (len(SEQ_FEATURES) if USE_SEQ_FEATURES else 0)
GENOME_FASTA = os.path.join("genome", "hg38.fa")  # uncompressed; pyfaidx writes hg38.fa.fai next to it


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


def tss_window_tracks(gene_info, cell_line, tracks=ALL_TRACKS):
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


# =========== Reference genome (sequence) features ==============

_GENOME = None


def _genome():
    """Open the hg38 FASTA once and check it matches the bigWig coordinate system."""
    global _GENOME
    if _GENOME is None:
        if not os.path.isfile(GENOME_FASTA):
            raise FileNotFoundError(
                f"{GENOME_FASTA} not found. Download "
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz, "
                "unzip it to genome/hg38.fa, or set USE_SEQ_FEATURES = False."
            )
        from pyfaidx import Fasta

        genome = Fasta(GENOME_FASTA)
        sizes = pybigtools.open(find_track("DNase", "X1", "bigwig")).chroms()
        for chrom in ("chr1", "chr14", "chr19"):
            if chrom in sizes and len(genome[chrom]) != sizes[chrom]:
                raise ValueError(
                    f"{chrom}: FASTA has {len(genome[chrom])} bp but the bigWigs have "
                    f"{sizes[chrom]} bp. Wrong genome build? Expected hg38."
                )
        _GENOME = genome
    return _GENOME


def window_seq(genome, chrom, start, end):
    """Upper-case bases of [start, end) as a uint8 array of length end - start.
    Positions outside the chromosome are 'N'."""
    chrom_len = len(genome[chrom])
    s, e = max(start, 0), min(end, chrom_len)
    if e <= s:
        return np.full(end - start, ord("N"), dtype=np.uint8)
    seq = "N" * (s - start) + genome[chrom][s:e].seq.upper() + "N" * (end - e)
    return np.frombuffer(seq.encode(), dtype=np.uint8)


def tss_sequence_features(gene_info):
    """GC fraction and CpG density per bin, laid out exactly like tss_window_tracks
    (same window, upstream always on the left). Shape (n_genes, 2, N_BINS).
    Both features are the same on both strands (CG is its own reverse complement)."""
    genome = _genome()
    tss = ((gene_info["TSS_start"] + gene_info["TSS_end"]) // 2).to_numpy()
    minus = (gene_info["strand"] == "-").to_numpy()
    A, C, G, T = (ord(b) for b in "ACGT")
    out = np.zeros((len(gene_info), 2, N_BINS), dtype=np.float32)

    def per_bin(a):
        return a.reshape(N_BINS, BIN_SIZE).sum(axis=1)

    for i, (chrom, pos, neg) in enumerate(zip(gene_info["chr"], tss, minus)):
        up, down = (
            (DOWNSTREAM_WINDOW, UPSTREAM_WINDOW)
            if neg
            else (UPSTREAM_WINDOW, DOWNSTREAM_WINDOW)
        )
        s = window_seq(genome, chrom, int(pos) - up, int(pos) + down)
        is_c, is_g = s == C, s == G
        valid = is_c | is_g | (s == A) | (s == T)
        cg = np.zeros_like(is_c)
        cg[:-1] = is_c[:-1] & is_g[1:]  # a C directly followed by a G, counted in the bin of the C
        gc = per_bin(is_c | is_g) / np.maximum(per_bin(valid), 1)  # N-only bins -> 0
        cpg = per_bin(cg) / BIN_SIZE
        f = np.stack([gc, cpg]).astype(np.float32)
        out[i] = f[:, ::-1] if neg else f
    return out


def make_loader(
    split,
    tracks=ALL_TRACKS,
    cells=("X1", "X2"),
    batch_size=64,
    shuffle=True,
    stats=None,
    use_seq_features=USE_SEQ_FEATURES,
):
    """stats: per-cell (mean, std) per track. Pass `train_loader.stats` when building the
    val loader so it is scaled like train. Cells missing from `stats` (e.g. X3) get
    their own stats from their own genes (no labels involved)."""

    genes = ALL_GENES[(ALL_GENES["cell"].isin(cells)) & (ALL_GENES["split"] == split)]

    X = np.empty(shape=(len(genes), len(tracks), N_BINS), dtype=np.float32)

    for cell in cells:
        mask = (genes["cell"] == cell).to_numpy()
        X[mask] = tss_window_tracks(genes[mask], cell, tracks)

    X = np.log1p(X)  # distribution is heavilly skewed, hence logarithmized

    if use_seq_features:  # appended after the log1p: these are fractions, not skewed signal
        X = np.concatenate([X, tss_sequence_features(genes)], axis=1)

    # z-score each track within its own cell line (one mean/std per track, over genes and bins)
    stats = {} if stats is None else dict(stats)
    for cell in cells:
        mask = (genes["cell"] == cell).to_numpy()
        if cell not in stats:
            mu = X[mask].mean(axis=(0, 2), keepdims=True)  # (1, n_tracks, 1)
            sd = X[mask].std(axis=(0, 2), keepdims=True) + 1e-6
            stats[cell] = (mu, sd)
        mu, sd = stats[cell]
        X[mask] = (X[mask] - mu) / sd

    X = torch.from_numpy(X)

    if split == "test":
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    else:
        y = torch.from_numpy(np.log1p(genes["gex"].to_numpy(np.float32)))
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    loader.stats = stats
    return loader


if __name__ == "__main__":
    print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")
