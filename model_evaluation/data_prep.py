import pandas as pd
import glob
import os

DATA_DIR = "data"   # relative to the project root, where this notebook lives
WORK_DIR = "."      # submission.csv is written here

ALL_FILES = sorted(
    p for p in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
    if os.path.isfile(p)
)
print(f"{len(ALL_FILES)} files found under {DATA_DIR}/")

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

