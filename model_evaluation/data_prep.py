import pandas as pd
import glob
import os

DATA_DIR = "data"   # relative to the project root, where this notebook lives
WORK_DIR = "."      # submission.csv is written here

ALL_FILES = sorted(
    p for p in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
    if os.path.isfile(p)
)

def find_file(name):
    # Path of the unique file whose name is `name` (e.g. the CAGE-train *_info.tsv / *_y.tsv files).
    hits = [p for p in ALL_FILES if os.path.basename(p) == name]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly 1 file named {name}, found {len(hits)}: {hits}")
    return hits[0]


def find_track(mark, cell_line, kind):
    # Path of a chromatin track: data/<mark>-<kind>/<cell_line>.<ext>
    # kind: 'bed' (narrowPeak peaks) or 'bigwig' (continuous signal; extension is .bw or .bigwig).
    exts = (".bed", ".bed.gz", ".narrowpeak", ".narrowpeak.gz") if kind == "bed" else (".bw", ".bigwig")
    folder = os.path.join(DATA_DIR, f"{mark}-{kind}")
    hits = [p for p in glob.glob(os.path.join(folder, f"{cell_line}.*")) if p.lower().endswith(exts)]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly 1 {kind} file for {mark}/{cell_line}, found {len(hits)}: {hits}")
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
    ("X1", "val"):   load_split("X1", "val"),
    ("X2", "train"): load_split("X2", "train"),
    ("X2", "val"):   load_split("X2", "val"),
    ("X3", "test"):  load_split("X3", "test"),
    }

# all gene ids
# every gene row from every split, with `cell` and `split` columns (gex is NaN for X3 test)
ALL_GENES = pd.concat(get_gene_info(), names=["cell", "split"]).reset_index(["cell", "split"]).reset_index(drop=True)


# =========== Track signals stuff ==============

import pybigtools
import numpy as np




ALL_TRACKS = ["DNase", "H3K4me3", "H3K4me1", "H3K27ac", "H3K36me3", "H3K27me3", "H3K9me3"]

# i would say upstream is more informative in regrards to regulatory aspect, other methods shift their windows to include more upstream as well
UPSTREAM_WINDOW = 1000            
DOWNSTREAM_WINDOW = 500 

BIN_SIZE = 10
N_BINS = (UPSTREAM_WINDOW + DOWNSTREAM_WINDOW) // BIN_SIZE

def tss_features(genes, cell_line, tracks=ALL_TRACKS):
    # Mean bigWig signal in [TSS - WINDOW, TSS + WINDOW] for each track. One row per gene.
    tss = ((genes["TSS_start"] + genes["TSS_end"]) // 2).to_numpy()
    feats = {}
    for track in tracks:
        bw = pybigtools.open(find_track(track, cell_line, "bigwig"))
        sizes = bw.chroms()
        feats[track] = np.array([
            bw.values(chrom, max(int(pos) - UPSTREAM_WINDOW, 0), min(int(pos) + DOWNSTREAM_WINDOW, sizes[chrom]),
                      bins=1, summary="mean", fillna=0)[0]
            for chrom, pos in zip(genes["chr"], tss)
        ])
    return pd.DataFrame(feats, index=genes.index)

def tss_window_tracks(genes, cell_line, tracks=ALL_TRACKS):
    """get actual sequence resolved track signal over multiple tracks (as opposed to mean summary)"""

    tss = ((genes["TSS_start"] + genes["TSS_end"]) // 2).to_numpy()
    minus = (genes["strand"] == "-").to_numpy()
    windows = {}

    for track in tracks:
        bw = pybigtools.open(find_track(track, cell_line, "bigwig"))
        rows = []
        for chrom, pos, neg in zip(genes["chr"], tss, minus):
            # upstream is to the right on the minus strand
            up, down = (DOWNSTREAM_WINDOW, UPSTREAM_WINDOW) if neg else (UPSTREAM_WINDOW, DOWNSTREAM_WINDOW)
            v = bw.values(chrom,
                          start=max(int(pos) - up, 0),
                          end=int(pos) + down,  # past chromosome end -> oob=0
                          bins=N_BINS,
                          summary="mean",
                          fillna=0,
                          oob=0)
            rows.append(v[::-1] if neg else v)  # flip so upstream is always on the left
        windows[track] = np.stack(rows)

    return np.stack([windows[t] for t in tracks], axis=1)  # (n_genes, n_tracks, N_BINS)



if __name__ == '__main__':
    print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")

    
    print(tss_window_tracks(genes=ALL_GENES, cell_line='X1', tracks=ALL_TRACKS).shape)