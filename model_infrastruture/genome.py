"""Genome sequence access (indexed FASTA) and a CpG-island track derived from it."""
import os

import numpy as np

GENOME_FASTA = os.path.join("genome", "hg38.fa")  # needs hg38.fa.fai next to it

# CpG island criteria of Gardiner-Garden & Frommer (1987), evaluated in a sliding window:
# the window centred on a base is an island if GC > 50% and observed/expected CpG > 0.6
CPG_WINDOW = 200
CPG_MIN_GC = 0.5
CPG_MIN_OBS_EXP = 0.6

_N = ord("N")


class IndexedFasta:
    """Random access to a FASTA file through its .fai index (no extra dependency)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.index = {}
        with open(path + ".fai", encoding="utf-8") as fai:
            for line in fai:
                name, length, offset, bases, width = line.rstrip("\n").split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(bases), int(width))

    def fetch(self, chrom, start, end):
        """Upper-case bases [start, end) as uint8; positions outside the chromosome are 'N'."""
        length, offset, bases, width = self.index[chrom]
        out = np.full(max(end - start, 0), _N, dtype=np.uint8)
        s, e = max(start, 0), min(end, length)
        if e <= s:
            return out
        first = offset + (s // bases) * width + s % bases
        last = offset + ((e - 1) // bases) * width + (e - 1) % bases + 1
        self.f.seek(first)
        raw = np.frombuffer(self.f.read(last - first), dtype=np.uint8)
        raw = raw[(raw != 10) & (raw != 13)]  # drop line breaks (new, writable array)
        raw[(raw >= 97) & (raw <= 122)] -= 32  # soft-masked (lower-case) -> upper-case
        assert len(raw) == e - s, f"short read on {chrom}:{s}-{e}"
        out[s - start : e - start] = raw
        return out


_fasta = None


def get_fasta():
    global _fasta
    if _fasta is None:
        _fasta = IndexedFasta(GENOME_FASTA)
    return _fasta


def cpg_island_mask(seq):
    """seq: uint8 bases of a region plus CPG_WINDOW // 2 bases of flank on each side.
    Returns one bool per base of the region: is the CPG_WINDOW window centred on it an island?"""
    L = CPG_WINDOW
    c = (seq == ord("C")).astype(np.int64)
    g = (seq == ord("G")).astype(np.int64)
    cg = np.zeros_like(c)
    cg[:-1] = c[:-1] * g[1:]  # a C directly followed by a G
    cs, gs, cgs = (np.concatenate([[0], np.cumsum(x)]) for x in (c, g, cg))

    p = np.arange(len(seq) - L)  # window start positions
    n_c = cs[p + L] - cs[p]
    n_g = gs[p + L] - gs[p]
    n_cg = cgs[p + L - 1] - cgs[p]  # CpG pairs lying completely inside the window
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_exp = n_cg * L / (n_c * n_g)
    return ((n_c + n_g) / L > CPG_MIN_GC) & (n_c * n_g > 0) & (obs_exp > CPG_MIN_OBS_EXP)


def cpg_island_track(chrom, start, end, bin_size):
    """Fraction of every bin (0..1) that lies in a CpG island, for the region [start, end).
    The island call is symmetric between strands, so minus-strand genes just flip the result."""
    half = CPG_WINDOW // 2
    seq = get_fasta().fetch(chrom, start - half, end + half)
    return cpg_island_mask(seq).reshape(-1, bin_size).mean(axis=1).astype(np.float32)
