"""How different do the chromatin tracks look in X1, X2 and X3 at the SAME loci?

    python -m model_infrastruture.track_stats [--n-genes 2000]

Takes the X3 test genes (chr1) and reads their TSS windows from the bigWigs of all three cell
lines. Same genes everywhere, so any difference comes from the cell line itself (assay strength,
noise, coverage), not from which genes are in the set. No expression labels are needed.

What to look for:
  * one track whose mean / zero_frac / p99 in X3 is far from both X1 and X2 (weak or noisy assay)
  * tss_enrich (signal at the TSS relative to the whole window) that differs between cell lines
  * per-gene similarity (Spearman) that is clearly lower for X3 than between X1 and X2
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

CELLS = ("X1", "X2", "X3")
METRICS = {
    "mean": "mean of log1p(signal)",
    "std": "std of log1p(signal)",
    "zero_frac": "fraction of bins with no signal",
    "p99": "99th percentile of raw signal",
    "tss_enrich": "mean log1p signal in +-half bins around the TSS / mean over the window",
}


def summarize(arrays, tracks, tss_bin, half=4):
    """arrays: {cell: raw signal (n_genes, n_tracks, n_bins)} for the same genes in every cell.
    Returns (per-cell/track statistics, per-gene cross-cell Spearman per track)."""
    rows = []
    for cell, a in arrays.items():
        for i, track in enumerate(tracks):
            raw = a[:, i]
            x = np.log1p(raw)
            rows.append(
                dict(
                    cell=cell,
                    track=track,
                    mean=x.mean(),
                    std=x.std(),
                    zero_frac=(raw <= 0).mean(),
                    p99=np.percentile(raw, 99),
                    tss_enrich=x[:, tss_bin - half : tss_bin + half].mean() / max(x.mean(), 1e-9),
                )
            )
    stats = pd.DataFrame(rows)

    sim = []
    for i, track in enumerate(tracks):
        per_gene = {c: pd.Series(np.log1p(arrays[c][:, i]).mean(axis=1)) for c in arrays}
        row = {"track": track}
        for a, b in combinations(arrays, 2):
            row[f"{a}-{b}"] = per_gene[a].rank().corr(per_gene[b].rank())
        sim.append(row)
    return stats, pd.DataFrame(sim).set_index("track")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=2000)
    args = ap.parse_args()

    from model_infrastruture.data_prep import (
        ALL_GENES,
        ALL_TRACKS,
        BIN_SIZE,
        UPSTREAM_WINDOW,
        tss_window_tracks,
    )

    genes = ALL_GENES[(ALL_GENES["cell"] == "X3") & (ALL_GENES["split"] == "test")]
    genes = genes.sample(min(args.n_genes, len(genes)), random_state=0)
    print(f"{len(genes)} X3 test genes (chr1), windows read from X1, X2 and X3 bigWigs\n")

    arrays = {c: tss_window_tracks(genes, c) for c in CELLS}
    stats, sim = summarize(arrays, ALL_TRACKS, UPSTREAM_WINDOW // BIN_SIZE)

    pd.set_option("display.width", 120)
    for metric, desc in METRICS.items():
        table = stats.pivot(index="track", columns="cell", values=metric).reindex(ALL_TRACKS)
        print(f"--- {metric}: {desc}")
        print(table[list(CELLS)].round(3).to_string(), "\n")

    print("--- per-gene Spearman of mean log1p signal between cell lines")
    print(sim.reindex(ALL_TRACKS).round(3).to_string())
