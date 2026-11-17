"""Semi-synthetic count matrices with planted biclusters, built from the empirical pig-atlas parameters."""
import numpy as np


def plant_effects(n, m, rng, delta, sizes_I=(30, 80), sizes_J=(5, 10)):
    """Three planted biclusters (shift, shift-scale, anti-correlated) on disjoint gene sets.
    Returns log2 effect matrix and ground-truth (I, J, type) list."""
    E = np.zeros((n, m))
    genes = rng.permutation(n)
    truth, start = [], 0
    for kind in ('shift', 'scale', 'anti'):
        k = int(rng.integers(sizes_I[0], sizes_I[1] + 1)); l = int(rng.integers(sizes_J[0], sizes_J[1] + 1))
        I = np.sort(genes[start:start + k]); start += k
        J = np.sort(rng.choice(m, l, replace=False))
        b = rng.uniform(0.5, 1.0, l)                            # shared condition profile
        if kind == 'shift':
            a = np.ones(k)
        elif kind == 'scale':
            a = rng.uniform(0.5, 1.5, k)
        else:
            a = np.where(rng.random(k) < 0.5, 1.0, -1.0)
        E[np.ix_(I, J)] += delta * np.outer(a, b)
        truth.append((I, J, kind))
    return E, truth


def make_matrix(base, phi, lib, rng, delta, background='nb', Xreal=None):
    """background='nb'  : NB counts with gene means / dispersions / library sizes of the real atlas (no structure)
       background='real': the real atlas counts themselves, with planted effects applied multiplicatively"""
    n, m = base.shape[0], len(lib)
    E, truth = plant_effects(n, m, rng, delta)
    fold = 2.0 ** E
    if background == 'nb':
        s = lib / np.exp(np.mean(np.log(lib)))
        mu = base[:, None] * s[None, :] * fold
        r = (1.0 / phi)[:, None]
        X = rng.negative_binomial(r * np.ones_like(mu), r / (r + mu))
    else:
        X = Xreal.copy().astype(np.int64)
        up = fold > 1; dn = fold < 1
        X[up] = X[up] + rng.poisson(X[up] * (fold[up] - 1.0))
        X[dn] = rng.binomial(X[dn], fold[dn])
    return X, truth


def match_scores(found, truth):
    """Prelic recovery / relevance with cell-level Jaccard."""
    from hive import jaccard_cells
    T = [(t[0], t[1]) for t in truth]
    if len(found) == 0:
        return 0.0, 0.0, np.zeros(len(T))
    S = np.array([[jaccard_cells(f, t) for t in T] for f in found])
    rec_each = S.max(0)
    return float(rec_each.mean()), float(S.max(1).mean()), rec_each
