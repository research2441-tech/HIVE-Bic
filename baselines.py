"""Nine biclustering baselines run on the discovery fold (log-CPM Y1 / row-standardised Z1).
Each returns a list of (I, J) index arrays (at most K biclusters)."""
import numpy as np
import warnings
from hive import standardize, coherence, jaccard_cells, LogitBoost, _stdrows

K = 10


def _dedupe(bics, Z, K=K, jac=0.5):
    bics = [(np.sort(np.asarray(I)), np.sort(np.asarray(J))) for I, J in bics if len(I) >= 3 and len(J) >= 3]
    bics.sort(key=lambda b: -coherence(Z, b[0], b[1]) * len(b[0]) * len(b[1]))
    out = []
    for b in bics:
        if all(jaccard_cells(b, o) < jac for o in out):
            out.append(b)
        if len(out) == K:
            break
    return out


def _biclustlib(algo, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = algo.run(np.asarray(data, float))
    return [(np.asarray(b.rows), np.asarray(b.cols)) for b in r.biclusters]


def cc(Y, Z, rng):
    from biclustlib.algorithms import ChengChurchAlgorithm
    return _dedupe(_biclustlib(ChengChurchAlgorithm(num_biclusters=K), Y), Z)


def las(Y, Z, rng):
    from biclustlib.algorithms import LargeAverageSubmatrices
    # LAS searches for large positive averages; run on Z and on -Z so that down-regulated blocks are reachable
    b = _biclustlib(LargeAverageSubmatrices(num_biclusters=K), Z) + _biclustlib(LargeAverageSubmatrices(num_biclusters=K), -Z)
    return _dedupe(b, Z)


def plaid(Y, Z, rng):
    from biclustlib.algorithms import Plaid
    return _dedupe(_biclustlib(Plaid(num_biclusters=K), Y), Z)


def isa(Y, Z, rng, n_seeds=200, tg=2.0, tc=2.0, iters=30):
    """Iterative signature algorithm (Bergmann et al.): alternate thresholded gene and condition scores."""
    Eg = _stdrows(Z.T).T                      # genes standardised across conditions (rows)
    Ec = _stdrows(Z)                          # conditions standardised across genes (columns)
    Ec = ((Z - Z.mean(0)) / (Z.std(0) + 1e-9))
    n, m = Z.shape; mods = []
    for _ in range(n_seeds):
        g = np.zeros(n); g[rng.choice(n, 20, replace=False)] = 1
        for _ in range(iters):
            sc = Ec.T @ g; sc = (sc - sc.mean()) / (sc.std() + 1e-9)
            c = np.where(np.abs(sc) > tc, sc, 0.0)
            if not c.any():
                break
            sg = Eg @ c; sg = (sg - sg.mean()) / (sg.std() + 1e-9)
            g_new = np.where(np.abs(sg) > tg, sg, 0.0)
            if np.allclose(np.sign(g_new), np.sign(g)):
                g = g_new; break
            g = g_new
            if not g.any():
                break
        I = np.where(g != 0)[0]; J = np.where(c != 0)[0] if g.any() else np.array([], int)
        if len(I) >= 3 and len(J) >= 3:
            mods.append((I, J))
    return _dedupe(mods, Z)


def ssvd(Y, Z, rng, layers=K):
    """Sparse SVD biclustering (Lee et al.): rank-one layers with soft-thresholded singular vectors,
    sparsity chosen by BIC on a grid, then deflation."""
    R = Z.copy(); out = []
    n, m = R.shape
    for _ in range(layers):
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        u, v = U[:, 0] * s[0], Vt[0]
        best = None
        for _ in range(10):
            # update u with soft threshold chosen by BIC
            cand = []
            z = R @ v
            for lam in np.quantile(np.abs(z), [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]):
                uu = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
                rss = np.sum((R - np.outer(uu, v)) ** 2); df = np.count_nonzero(uu)
                cand.append((n * m * np.log(rss / (n * m)) + np.log(n * m) * df, uu))
            u = min(cand, key=lambda t: t[0])[1]
            if not u.any():
                break
            cand = []
            z = R.T @ u
            for lam in np.quantile(np.abs(z), [0.3, 0.5, 0.7, 0.8, 0.9]):
                vv = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
                rss = np.sum((R - np.outer(u, vv / (np.linalg.norm(vv) + 1e-12))) ** 2); df = np.count_nonzero(vv)
                cand.append((n * m * np.log(rss / (n * m)) + np.log(n * m) * df, vv))
            v = min(cand, key=lambda t: t[0])[1]
            nv = np.linalg.norm(v)
            if nv == 0:
                break
            v = v / nv
        I = np.where(u != 0)[0]; J = np.where(v != 0)[0]
        if len(I) < 3 or len(J) < 3:
            break
        out.append((I, J))
        R = R - np.outer(u, v)
    return _dedupe(out, Z)


def spectral_bic(Y, Z, rng):
    from sklearn.cluster import SpectralBiclustering
    P = 2.0 ** Y                              # strictly positive scale required
    mdl = SpectralBiclustering(n_clusters=(6, 5), method='log', random_state=int(rng.integers(1 << 30))).fit(P)
    bics = []
    for r in range(6):
        for c in range(5):
            I = np.where(mdl.row_labels_ == r)[0]; J = np.where(mdl.column_labels_ == c)[0]
            if 3 <= len(I) <= 400 and len(J) >= 3:
                bics.append((I, J))
    return _dedupe(bics, Z)


def spectral_cocl(Y, Z, rng):
    from sklearn.cluster import SpectralCoclustering
    P = 2.0 ** Y
    mdl = SpectralCoclustering(n_clusters=K, random_state=int(rng.integers(1 << 30))).fit(P)
    bics = [(np.where(mdl.row_labels_ == k)[0], np.where(mdl.column_labels_ == k)[0]) for k in range(K)]
    return _dedupe(bics, Z)


def _msr(Yb):
    return float(np.mean((Yb - Yb.mean(1, keepdims=True) - Yb.mean(0, keepdims=True) + Yb.mean()) ** 2))


def bpso(Y, Z, rng, n_part=20, iters=30, w=0.7, c1=1.5, c2=1.5):
    """Binary PSO biclustering with a Cheng-Church mean-squared-residue fitness (Kennedy-Eberhart transfer)."""
    n, m = Y.shape
    delta = 0.5 * _msr(Y)
    def fit(x):
        I = np.where(x[:n] > 0)[0]; J = np.where(x[n:] > 0)[0]
        if len(I) < 3 or len(J) < 3:
            return -1e9
        r = _msr(Y[np.ix_(I, J)])
        size = np.log(len(I) * len(J))
        return size if r <= delta else size * delta / r
    X = np.zeros((n_part, n + m))
    for p in range(n_part):
        X[p, rng.choice(n, 40, replace=False)] = 1; X[p, n + rng.choice(m, 8, replace=False)] = 1
    V = rng.normal(0, 1, X.shape); pb = X.copy(); pf = np.array([fit(x) for x in X]); g = pb[np.argmax(pf)].copy()
    for _ in range(iters):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (g - X)
        V = np.clip(V, -4, 4)
        X = (rng.random(X.shape) < 1 / (1 + np.exp(-V))).astype(float)
        X[:, :n] *= (rng.random((n_part, n)) < 0.05) | (X[:, :n] * (pb[:, :n] + g[None, :n]) > 0)   # sparsity guard
        f = np.array([fit(x) for x in X])
        better = f > pf; pb[better] = X[better]; pf[better] = f[better]
        g = pb[np.argmax(pf)].copy()
    bics = [(np.where(x[:n] > 0)[0], np.where(x[n:] > 0)[0]) for x in pb]
    return _dedupe(bics, Z)


def lbbso(Y, Z, rng, n_bees=20, lb_iter=50, bso_iter=30, alpha=0.7, beta=1.5, gamma=1.5, thr=0.7):
    """Re-implementation of the LB-BSO procedure described in the base manuscript: random binary membership,
    in-sample LogitBoost relabelling of gene membership, PSO-type displacement on membership scores and a
    Pearson-correlation fitness with a 0.7 threshold.  Where the original description is under-specified,
    the most direct reading was implemented (see the paper text)."""
    n, m = Y.shape
    def fit(I, J):
        if len(I) < 3 or len(J) < 3:
            return -1.0
        S = _stdrows(Y[np.ix_(I, J)]); R = np.abs(S @ S.T) / len(J)
        iu = np.triu_indices(len(I), 1)
        return float(np.mean(R[iu] >= thr))
    pos = np.zeros((n_bees, n + m))
    for b in range(n_bees):
        pos[b, rng.choice(n, 40, replace=False)] = 1; pos[b, n + rng.choice(m, 8, replace=False)] = 1
    vel = np.zeros_like(pos)
    for b in range(n_bees):                    # LogitBoost relabelling (labels = current membership)
        I = np.where(pos[b, :n] > 0.5)[0]; J = np.where(pos[b, n:] > 0.5)[0]
        y = np.zeros(n, int); y[I] = 1
        p = LogitBoost(lb_iter, 2, seed=int(rng.integers(1 << 30))).fit(Y[:, J], y).predict_proba(Y[:, J])
        pos[b, :n] = (p >= 0.5).astype(float)
    pbest = pos.copy(); pf = np.array([fit(np.where(x[:n] > .5)[0], np.where(x[n:] > .5)[0]) for x in pos])
    g = pbest[np.argmax(pf)].copy()
    for _ in range(bso_iter):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = alpha * vel + beta * r1 * (pbest - pos) + gamma * r2 * (g - pos)
        pos = np.clip(pos + vel, 0, 1)
        f = np.array([fit(np.where(x[:n] > .5)[0], np.where(x[n:] > .5)[0]) for x in pos])
        better = f > pf; pbest[better] = pos[better]; pf[better] = f[better]
        g = pbest[np.argmax(pf)].copy()
    bics = [(np.where(x[:n] > .5)[0], np.where(x[n:] > .5)[0]) for x in pbest]
    return _dedupe(bics, Z)


METHODS = dict(CC=cc, ISA=isa, Plaid=plaid, LAS=las, SSVD=ssvd, SpecBic=spectral_bic, SpecCo=spectral_cocl, BPSO=bpso, LBBSO=lbbso)
