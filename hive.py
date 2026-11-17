"""
HIVE-Bic: Honest Inference Via thinned Evidence for Biclustering.

Pipeline
  1. dispersion-conditional negative-binomial data thinning  -> discovery fold X1, confirmation fold X2
  2. diversity-constrained binary bee swarm on the discovery fold (null-corrected coherence objective)
  3. cross-fitted LogitBoost membership refinement (genes and conditions)
  4. certification on the confirmation fold (row/column permutation nulls, Benjamini-Hochberg)
All randomness is driven by numpy Generators so that every run is reproducible from a seed.
"""
import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

# ----------------------------------------------------------------------------------------------
# 1. Dispersion-conditional NB thinning
# ----------------------------------------------------------------------------------------------
def size_factors(lib):
    lib = np.asarray(lib, float)
    return lib / np.exp(np.mean(np.log(lib)))


def estimate_dispersion(X, lib, groups, floor=1e-3, cap=5.0, shrink=0.5):
    """Method-of-moments NB dispersion pooled over replicate groups (conditions with >=2 samples).
    Var(y_gj) = mu_g/s_j + phi_g mu_g^2 for size-factor normalised y = x/s.  Estimates are shrunk
    towards the median on the log scale to stabilise genes with few replicate groups."""
    s = size_factors(lib)
    Y = X / s[None, :]
    num = np.zeros(X.shape[0]); den = np.zeros(X.shape[0])
    for gname in np.unique(groups):
        idx = np.where(groups == gname)[0]
        if len(idx) < 2:
            continue
        m = Y[:, idx].mean(1); v = Y[:, idx].var(1, ddof=1)
        inv_s = np.mean(1.0 / s[idx])
        ok = m > 0
        est = np.zeros_like(m); est[ok] = (v[ok] - m[ok] * inv_s) / m[ok] ** 2
        num += (len(idx) - 1) * est; den += (len(idx) - 1)
    phi = np.clip(num / np.maximum(den, 1), floor, cap)
    lphi = np.log(phi); med = np.median(lphi)
    return np.exp(shrink * lphi + (1 - shrink) * med)


def nb_thin(X, phi, eps, rng):
    """Negative-binomial data thinning: X1 | X ~ BetaBinomial(X, eps*b, (1-eps)*b), b = 1/phi.
    Under X ~ NB(mu, b) this gives independent X1 ~ NB(eps*mu, eps*b) and X2 = X - X1."""
    b = (1.0 / phi)[:, None] * np.ones((1, X.shape[1]))
    p = rng.beta(eps * b, (1 - eps) * b)
    X1 = rng.binomial(X.astype(np.int64), p)
    return X1, X - X1


def log_cpm(X, lib):
    return np.log2(X / np.asarray(lib, float)[None, :] * 1e6 + 1.0)


# ----------------------------------------------------------------------------------------------
# 2. Null-corrected coherence objective
# ----------------------------------------------------------------------------------------------
def _stdrows(Z):
    Z = Z - Z.mean(1, keepdims=True)
    sd = Z.std(1, keepdims=True); sd[sd < 1e-9] = 1.0
    return Z / sd


def standardize(Y):
    """Robust row standardisation (median / MAD): a condition set J that covers a minority of the
    conditions does not pull the centre of its member genes, so cells outside J stay near zero."""
    Y = np.asarray(Y, float)
    med = np.median(Y, 1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(Y - med), 1, keepdims=True)
    sd = Y.std(1, keepdims=True)
    scale = np.maximum(mad, 0.25 * sd); scale[scale < 1e-9] = 1.0
    return np.clip((Y - med) / scale, -8, 8)


def coherence(Z, I, J):
    """Null-corrected rank-one energy of the submatrix Z[I, J] of the row-standardised matrix Z.
    lambda_1 / (|I||J|) is the share of squared signal carried by one gene loading x condition profile;
    (|I|^-1/2 + |J|^-1/2)^2 is its value for a pure-noise block (random-matrix edge), so the difference
    is positive only for structure beyond chance.  Shifting, scaling and sign-flipped (anti-correlated)
    members all load on the same singular pair, so one statistic covers the three pattern families."""
    I = np.asarray(I); J = np.asarray(J)
    if len(I) < 3 or len(J) < 3:
        return -1.0
    S = Z[np.ix_(I, J)]
    G = S.T @ S if len(J) <= len(I) else S @ S.T
    lam = np.linalg.eigvalsh(G)[-1]
    return float(lam / (len(I) * len(J)) - (1.0 / np.sqrt(len(I)) + 1.0 / np.sqrt(len(J))) ** 2)


def selection_null(n, m, ks, ls, rng, restarts=24, iters=8):
    """Search-adjusted null level of the rank-one energy share.  For every block size (k, l) a pure-noise
    n x m matrix is searched by alternating top-k / top-l selection (the same move family the swarm uses),
    and the pct-th percentile of the attained share lambda_1/(k l) over restarts is stored.  Subtracting this
    level removes the winner's-curse inflation that any optimiser creates on noise."""
    mu = np.zeros((len(ks), len(ls))); sd = np.zeros((len(ks), len(ls)))
    for a, k in enumerate(ks):
        for b, l in enumerate(ls):
            vals = []
            for r in range(restarts):
                Zn = standardize(rng.standard_normal((n, m)))
                J = rng.choice(m, l, replace=False)
                v = np.ones(l) / np.sqrt(l)
                for _ in range(iters):
                    I = np.argsort(-np.abs(Zn[:, J] @ v))[:k]
                    U, sv, Vt = np.linalg.svd(Zn[np.ix_(I, J)], full_matrices=False)
                    u = U[:, 0]
                    J = np.argsort(-np.abs(u @ Zn[I]))[:l]
                    v = np.linalg.svd(Zn[np.ix_(I, J)], full_matrices=False)[2][0]
                lam = np.linalg.svd(Zn[np.ix_(I, J)], compute_uv=False)[0] ** 2
                vals.append(lam / (k * l))
            mu[a, b] = np.mean(vals); sd[a, b] = np.std(vals, ddof=1)
    return mu, sd


class NullTable:
    def __init__(self, n, m, rng, ks=(10, 15, 25, 40, 60, 90, 130, 180, 250), ls=None):
        ls = ls or tuple(sorted(set([4, 5, 6, 8, 10, 13, 17, 22, max(23, m // 2)])))
        self.ks, self.ls = np.array(ks, float), np.array([x for x in ls if x <= m], float)
        self.mu, self.sd = selection_null(n, m, [int(k) for k in self.ks], [int(l) for l in self.ls], rng)
        from scipy.interpolate import RegularGridInterpolator
        g = (np.log(self.ks), np.log(self.ls))
        self.fm = RegularGridInterpolator(g, self.mu, bounds_error=False, fill_value=None)
        self.fs = RegularGridInterpolator(g, np.log(self.sd), bounds_error=False, fill_value=None)

    def __call__(self, k, l):
        k = np.clip(k, self.ks[0], self.ks[-1]); l = np.clip(l, self.ls[0], self.ls[-1])
        x = [[np.log(k), np.log(l)]]
        return float(self.fm(x)[0]), float(np.exp(self.fs(x)[0]))


def adjusted_score(Z, I, J, null):
    """Size-calibrated, search-adjusted coherence z-score used as the swarm objective:
    (lambda_1/(|I||J|) - mu0(|I|,|J|)) / sd0(|I|,|J|), with mu0 and sd0 the mean and standard deviation of
    the energy share that the same alternating search attains on pure noise of the same dimensions."""
    if len(I) < 3 or len(J) < 3:
        return -10.0
    S = Z[np.ix_(I, J)]
    G = S.T @ S if len(J) <= len(I) else S @ S.T
    lam = np.linalg.eigvalsh(G)[-1]
    mu, sd = null(len(I), len(J))
    return float((lam / (len(I) * len(J)) - mu) / sd)



def overlap_cells(a, b):
    """Overlap coefficient of the cell sets: shared cells divided by the smaller bicluster area."""
    I1, J1 = a; I2, J2 = b
    inter = len(np.intersect1d(I1, I2)) * len(np.intersect1d(J1, J2))
    return inter / min(len(I1) * len(J1), len(I2) * len(J2))


def jaccard_cells(a, b):
    I1, J1 = a; I2, J2 = b
    ni = len(np.intersect1d(I1, I2)); nj = len(np.intersect1d(J1, J2))
    inter = ni * nj
    union = len(I1) * len(J1) + len(I2) * len(J2) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------------------------
# 3. LogitBoost (Friedman, Hastie, Tibshirani 2000) with depth-limited regression trees
# ----------------------------------------------------------------------------------------------
def _best_split(Fs, O, w, wz, node):
    """Best weighted least-squares split of the samples in `node` (boolean mask).
    Fs: features sorted per column, O: argsort per column.  Returns (gain, feature, threshold)."""
    ww = np.where(node, w, 0.0)[O]; zz = np.where(node, wz, 0.0)[O]
    WL = np.cumsum(ww, 0); SL = np.cumsum(zz, 0)
    W, S = WL[-1], SL[-1]
    WR = W[None, :] - WL; SR = S[None, :] - SL
    valid = (WL > 1e-12) & (WR > 1e-12)
    valid[:-1] &= Fs[1:] > Fs[:-1]
    valid[-1] = False
    gain = np.where(valid, SL ** 2 / np.maximum(WL, 1e-12) + SR ** 2 / np.maximum(WR, 1e-12), -np.inf)
    idx = np.argmax(gain); i, j = np.unravel_index(idx, gain.shape)
    if not np.isfinite(gain[i, j]):
        return None
    return gain[i, j], j, 0.5 * (Fs[i, j] + Fs[i + 1, j])


class _Tree2:
    """Depth-limited (<=2) weighted regression tree on presorted features (fast, deterministic)."""
    def fit(self, F, Fs, O, z, w, depth):
        n = len(z); wz = w * z
        self.root = self._grow(F, Fs, O, z, w, wz, np.ones(n, bool), depth)
        return self

    def _leaf(self, w, wz, node):
        W = w[node].sum()
        return ('leaf', wz[node].sum() / W if W > 0 else 0.0)

    def _grow(self, F, Fs, O, z, w, wz, node, depth):
        if depth == 0 or node.sum() < 4:
            return self._leaf(w, wz, node)
        sp = _best_split(Fs, O, w, wz, node)
        if sp is None:
            return self._leaf(w, wz, node)
        _, j, thr = sp
        left = node & (F[:, j] <= thr); right = node & ~left
        return ('split', j, thr, self._grow(F, Fs, O, z, w, wz, left, depth - 1), self._grow(F, Fs, O, z, w, wz, right, depth - 1))

    def predict(self, F):
        return self._pred(self.root, F)

    def _pred(self, nd, F):
        if nd[0] == 'leaf':
            return np.full(F.shape[0], nd[1])
        _, j, thr, l, r = nd
        out = np.empty(F.shape[0]); m = F[:, j] <= thr
        out[m] = self._pred(l, F[m]); out[~m] = self._pred(r, F[~m])
        return out


class LogitBoost:
    """Two-class LogitBoost (Friedman, Hastie and Tibshirani, 2000): Newton steps on the binomial
    log-likelihood, each fitting a depth-limited regression tree to the working response
    z = (y - p) / (p (1 - p)) with weights w = p (1 - p); F <- F + f/2 and p = 1 / (1 + exp(-2F))."""
    def __init__(self, n_rounds=50, depth=2, nu=1.0, zmax=4.0, seed=0):
        self.n_rounds, self.depth, self.nu, self.zmax, self.seed = n_rounds, depth, nu, zmax, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        O = np.argsort(X, axis=0, kind='stable'); Fs = np.take_along_axis(X, O, 0)
        F = np.zeros(len(y)); p = np.full(len(y), 0.5); self.trees = []
        for m in range(self.n_rounds):
            w = np.clip(p * (1 - p), 1e-6, None)
            z = np.clip((y - p) / w, -self.zmax, self.zmax)
            t = _Tree2().fit(X, Fs, O, z, w, self.depth)
            F += 0.5 * self.nu * t.predict(X)
            p = expit(2 * F); self.trees.append(t)
        return self

    def decision(self, X):
        X = np.asarray(X, float); F = np.zeros(X.shape[0])
        for t in self.trees:
            F += 0.5 * self.nu * t.predict(X)
        return F

    def predict_proba(self, X):
        return expit(2 * self.decision(X))


# ----------------------------------------------------------------------------------------------
# Membership features (all computed on the row-standardised discovery matrix Z)
# ----------------------------------------------------------------------------------------------
def _gene_features(Z, J, ref_genes):
    """Features of every gene relative to the shared condition profile v estimated from ref_genes over J."""
    S = Z[ref_genes][:, J]
    v = np.linalg.svd(S, full_matrices=False)[2][0]
    if v.sum() < 0:
        v = -v
    proj = Z[:, J] @ v / np.sqrt(len(J))             # signed projection on the profile
    sgn = np.sign(proj); sgn[sgn == 0] = 1
    notJ = np.setdiff1d(np.arange(Z.shape[1]), J)
    contrast = Z[:, J].mean(1) - Z[:, notJ].mean(1)
    zj = Z[:, J]
    cj = zj - zj.mean(1, keepdims=True)
    r = (cj @ (v - v.mean())) / (np.linalg.norm(cj, axis=1) * np.linalg.norm(v - v.mean()) + 1e-9)
    aligned = (zj * sgn[:, None])[:, np.argsort(-v)]
    return np.column_stack([proj, np.abs(proj), contrast * sgn, r * sgn, aligned])


def _cond_features(Z, I, J):
    S = Z[np.ix_(I, J)]
    u = np.linalg.svd(S, full_matrices=False)[0][:, 0]
    score = u @ Z[I] / np.sqrt(len(I))               # projection of every condition on the gene loading
    top = np.argsort(-np.abs(u))[:min(15, len(I))]
    aligned = (Z[I][top] * np.sign(u[top])[:, None]).T
    return np.column_stack([score, np.abs(score), aligned])


def _cap(p, keep_min, keep_max, tau):
    sel = np.where(p >= tau)[0]
    if len(sel) < keep_min:
        sel = np.argsort(-p)[:keep_min]
    if len(sel) > keep_max:
        sel = np.argsort(-p)[:keep_max]
    return np.sort(sel)


def crossfit_membership(Z, I, J, cfg, rng):
    """Cross-fitted LogitBoost update of gene and condition membership.  Labels come from the current
    bicluster; every probability is predicted out-of-fold, and the condition profile used to build a
    held-out gene's features is estimated without that gene, so no unit is scored by a model or a
    profile that already contains its own label."""
    n, m = Z.shape
    K = cfg['cf_folds']
    y = np.zeros(n, int); y[I] = 1
    folds = rng.permutation(n) % K
    p = np.zeros(n)
    for k in range(K):
        tr = folds != k; te = ~tr
        ref = np.where(tr & (y == 1))[0]
        if len(ref) < 3:
            return I, J
        F = _gene_features(Z, J, ref)
        lb = LogitBoost(cfg['lb_rounds'], cfg['lb_depth'], seed=int(rng.integers(1 << 30)))
        p[te] = lb.fit(F[tr], y[tr]).predict_proba(F[te])
    I_new = _cap(p, cfg['gmin'], cfg['gmax'], cfg['tau_g'])
    yc = np.zeros(m, int); yc[J] = 1
    Fc = _cond_features(Z, I_new, J)
    Kc = min(K, m); folds = rng.permutation(m) % Kc
    pc = np.zeros(m)
    for k in range(Kc):
        tr = folds != k; te = ~tr
        if yc[tr].sum() < 2:
            pc[te] = yc[te]; continue
        lb = LogitBoost(cfg['lb_rounds'], cfg['lb_depth'], seed=int(rng.integers(1 << 30)))
        pc[te] = lb.fit(Fc[tr], yc[tr]).predict_proba(Fc[te])
    J_new = _cap(pc, cfg['smin'], cfg['smax'], cfg['tau_c'])
    return I_new, J_new


def insample_membership(Z, I, J, cfg, rng):
    """Ablation: LogitBoost trained and scored on the same labels (the circular update of LB-BSO)."""
    n, m = Z.shape
    y = np.zeros(n, int); y[I] = 1
    F = _gene_features(Z, J, I)
    p = LogitBoost(cfg['lb_rounds'], cfg['lb_depth'], seed=int(rng.integers(1 << 30))).fit(F, y).predict_proba(F)
    I_new = _cap(p, cfg['gmin'], cfg['gmax'], cfg['tau_g'])
    yc = np.zeros(m, int); yc[J] = 1
    Fc = _cond_features(Z, I_new, J)
    pc = LogitBoost(cfg['lb_rounds'], cfg['lb_depth'], seed=int(rng.integers(1 << 30))).fit(Fc, yc).predict_proba(Fc)
    return I_new, _cap(pc, cfg['smin'], cfg['smax'], cfg['tau_c'])


# ----------------------------------------------------------------------------------------------
# Diversity-constrained binary bee swarm
# ----------------------------------------------------------------------------------------------
DEFAULT = dict(n_bees=20, t_max=40, stall=6, hv_tol=1e-3, limit=6, archive=20, lam=1.0, jac_admit=0.5,
               gmin=10, gmax=250, smin=4, smax=None, beta0=2.0, eta=1.5, temp=2.0, n_random=40,
               lb_rounds=50, lb_depth=2, cf_folds=5, tau_g=0.4, tau_c=0.7, refine_rounds=2,
               eps=0.5, n_perm=499, q=0.10, use_thinning=True, use_boost=True, use_crossfit=True,
               use_diversity=True, use_selnull=True, n_refine=8, anchor_temp=1.0, polish=1, adaptive=False, member_cert=False, objective='delta')


def pearson_score(Z, I, J):
    """Base-paper style objective: share of gene pairs with |Pearson r| >= 0.7 over J."""
    S = _stdrows(Z[np.ix_(I, J)]); R = np.abs(S @ S.T) / len(J)
    iu = np.triu_indices(len(I), 1)
    return float(np.mean(R[iu] >= 0.7)) - 0.5


class Swarm:
    def __init__(self, Z, cfg, rng, null=None):
        self.Z, self.cfg, self.rng, self.null = Z, cfg, rng, null
        self.n, self.m = Z.shape
        if cfg['smax'] is None:
            cfg['smax'] = max(cfg['smin'] + 1, self.m // 2)
        self.n_eval = 0
        self.R = None
        self.tabu = None

    def score(self, I, J):
        self.n_eval += 1
        if self.cfg['objective'] == 'pearson':
            return pearson_score(self.Z, I, J)
        if self.null is not None:
            return adjusted_score(self.Z, I, J, self.null)
        return coherence(self.Z, I, J)

    def obj(self, I, J):
        d = self.score(I, J)
        return d if self.null is not None else d * len(I) * len(J)

    def fitness(self, I, J, archive, skip=None):
        d = self.score(I, J)
        f = d if self.null is not None else (d * len(I) * len(J) if d > 0 else d)
        if self.cfg['use_diversity'] and f > 0:
            js = [overlap_cells((I, J), (a['I'], a['J'])) for a in archive if a is not skip]
            if js:
                f *= (1.0 - self.cfg['lam'] * min(1.0, max(js)))
        return f, d

    def seed(self):
        """Seed a food source on the discovery fold.  Two seed types are alternated:
        (a) anchor gene -> its most correlated partners (|r|) -> conditions on which the group deviates;
        (b) anchor gene -> the conditions where it deviates most -> genes co-deviating on them."""
        c = self.cfg
        if self.R is None:
            self.R = (self.Z @ self.Z.T) / self.m
            A = np.abs(self.R); np.fill_diagonal(A, 0)
            kc = np.sort(A, 1)[:, -20:].mean(1)              # local co-expression connectivity
            zc = (kc - kc.mean()) / (kc.std() + 1e-9)
            w = np.exp(c['anchor_temp'] * zc); self.anchor_p = w / w.sum()
        p = self.anchor_p.copy()
        if self.tabu is not None and len(self.tabu):
            p[self.tabu] *= 0.02                        # anchors already covered by the archive are rarely reused
        g0 = int(self.rng.choice(self.n, p=p / p.sum()))
        k = int(self.rng.integers(20, 61))
        l = int(self.rng.integers(c['smin'], min(c['smax'], c['smin'] + 6) + 1))
        if self.rng.random() < 0.5:
            r = self.R[g0]
            I = np.argsort(-np.abs(r))[:k]
            sgn = np.sign(r[I]); sgn[sgn == 0] = 1
            J = np.argsort(-np.abs(sgn @ self.Z[I]))[:l]
        else:
            J = np.argsort(-np.abs(self.Z[g0]))[:l]
            sc = self.Z[:, J] @ np.sign(self.Z[g0, J])
            I = np.argsort(-np.abs(sc))[:k]
        return np.sort(I), np.sort(J)

    def _bits(self, xi, xk, xg, guide):
        c = self.cfg; size = len(xi)
        phi = self.rng.uniform(-1, 1, size); r = self.rng.uniform(0, 1, size)
        v = c['beta0'] * (2 * xi - 1) + phi * (xi - xk) + r * (xg - xi) + c['eta'] * guide
        return self.rng.random(size) < expit(c['temp'] * v)

    def neighbour(self, src, other, leader):
        c = self.cfg; Z = self.Z
        I, J = src['I'], src['J']
        S = Z[np.ix_(I, J)]
        U, sv, Vt = np.linalg.svd(S, full_matrices=False)
        u, v = U[:, 0], Vt[0]
        pool = np.unique(np.concatenate([I, other['I'], leader['I'], self.rng.choice(self.n, c['n_random'], replace=False)]))
        g = np.abs(Z[np.ix_(pool, J)] @ v) / np.sqrt(len(J))
        g = (g - g.mean()) / (g.std() + 1e-9)
        xi = np.isin(pool, I).astype(float); xk = np.isin(pool, other['I']).astype(float); xg = np.isin(pool, leader['I']).astype(float)
        newI = pool[self._bits(xi, xk, xg, g)]
        allj = np.arange(self.m)
        sc = np.abs(u @ Z[I]) / np.sqrt(len(I))
        sc = (sc - sc.mean()) / (sc.std() + 1e-9)
        xi = np.isin(allj, J).astype(float); xk = np.isin(allj, other['J']).astype(float); xg = np.isin(allj, leader['J']).astype(float)
        newJ = allj[self._bits(xi, xk, xg, sc)]
        if len(newI) < c['gmin'] or len(newJ) < c['smin']:
            return None
        if len(newI) > c['gmax']:
            newI = newI[np.argsort(-np.abs(Z[np.ix_(newI, J)] @ v))[:c['gmax']]]
        if len(newJ) > c['smax']:
            newJ = newJ[np.argsort(-sc[newJ])[:c['smax']]]
        if c['polish'] > 0:
            newI, newJ = self.polish(newI, newJ, c['polish'])
        return np.sort(newI), np.sort(newJ)

    def polish(self, I, J, steps):
        """Coherence-guided local polish (memetic step).  Genes and conditions are ranked by their absolute
        projections on the current singular pair, and the block size along each axis is chosen to maximise
        the total search-adjusted excess energy |I||J| x score(I, J)."""
        Z, c = self.Z, self.cfg
        kgrid = np.unique(np.geomspace(c['gmin'], c['gmax'], 14).astype(int))
        for _ in range(steps):
            v = np.linalg.svd(Z[np.ix_(I, J)], full_matrices=False)[2][0]
            og = np.argsort(-np.abs(Z[:, J] @ v))
            ks = kgrid if c['adaptive'] else [len(I)]
            I = og[:max(ks, key=lambda k: self.obj(og[:k], J))]
            u = np.linalg.svd(Z[np.ix_(I, J)], full_matrices=False)[0][:, 0]
            oc = np.argsort(-np.abs(u @ Z[I]))
            ls = range(c['smin'], c['smax'] + 1) if c['adaptive'] else [len(J)]
            J = oc[:max(ls, key=lambda l: self.obj(I, oc[:l]))]
        return np.sort(I), np.sort(J)

    def hypervolume(self, archive):
        pts = sorted([(a['delta'], np.log(len(a['I']) * len(a['J'])) / np.log(self.n * self.m)) for a in archive if a['delta'] > 0], key=lambda t: -t[0])
        hv, best = 0.0, 0.0
        for x, y in pts:                              # 2-D dominated area w.r.t. reference (0, 0)
            if y > best:
                hv += x * (y - best); best = y
        return hv

    def update_archive(self, archive, I, J, d):
        if d <= 0:
            return archive
        cand = dict(I=I, J=J, delta=d, f=d if self.null is not None else d * len(I) * len(J))
        for k, a in enumerate(archive):
            if redundant(cand, a, self.cfg['jac_admit']):
                if cand['f'] > a['f']:
                    archive[k] = cand
                return archive
        archive.append(cand)
        archive.sort(key=lambda a: -a['f'])
        return archive[:self.cfg['archive']]

    def run(self):
        c = self.cfg
        srcs, archive = [], []
        for _ in range(c['n_bees']):
            I, J = self.seed(); f, d = self.fitness(I, J, archive)
            srcs.append(dict(I=I, J=J, f=f, d=d, trial=0))
            archive = self.update_archive(archive, I, J, d)
        hv_hist = [self.hypervolume(archive)]
        stall = 0
        for t in range(c['t_max']):
            fits = np.array([s['f'] for s in srcs])
            prob = np.exp((fits - fits.max()) / (np.std(fits) + 1e-9)); prob /= prob.sum()
            order = list(range(c['n_bees'])) + list(self.rng.choice(c['n_bees'], c['n_bees'], p=prob))  # employed + onlookers
            for i in order:
                k = int(self.rng.integers(c['n_bees'] - 1)); k += k >= i
                leader = archive[int(self.rng.integers(min(5, len(archive))))] if archive else srcs[i]
                nb = self.neighbour(srcs[i], srcs[k], leader)
                if nb is None:
                    srcs[i]['trial'] += 1; continue
                own = [a for a in archive if jaccard_cells((a['I'], a['J']), (srcs[i]['I'], srcs[i]['J'])) >= c['jac_admit']]
                f, d = self.fitness(nb[0], nb[1], [a for a in archive if not any(a is o for o in own)])
                if f > srcs[i]['f']:
                    srcs[i] = dict(I=nb[0], J=nb[1], f=f, d=d, trial=0)
                    archive = self.update_archive(archive, nb[0], nb[1], d)
                else:
                    srcs[i]['trial'] += 1
            for i in range(c['n_bees']):                   # scouts
                if srcs[i]['trial'] > c['limit']:
                    I, J = self.seed(); f, d = self.fitness(I, J, archive)
                    srcs[i] = dict(I=I, J=J, f=f, d=d, trial=0)
                    archive = self.update_archive(archive, I, J, d)
            self.tabu = np.unique(np.concatenate([a['I'] for a in archive])) if archive else None
            hv = self.hypervolume(archive); hv_hist.append(hv)
            rel = (hv - hv_hist[-2]) / (abs(hv_hist[-2]) + 1e-12)
            stall = stall + 1 if rel < c['hv_tol'] else 0
            if stall >= c['stall']:
                break
        return archive, hv_hist


# ----------------------------------------------------------------------------------------------
# Certification on the confirmation fold
# ----------------------------------------------------------------------------------------------
def certify(Z2, cands, cfg, rng, mean_expr):
    n, m = Z2.shape
    edges = np.quantile(mean_expr, np.linspace(0, 1, 11)[1:-1])
    bins = np.digitize(mean_expr, edges)
    out = []
    B = cfg['n_perm']
    for c in cands:
        I, J = np.asarray(c['I']), np.asarray(c['J'])
        T = coherence(Z2, I, J)
        null_col = np.array([coherence(Z2, I, rng.choice(m, len(J), replace=False)) for _ in range(B)])
        bI = bins[I]; ub, cnt = np.unique(bI, return_counts=True)
        pools = {b: np.where(bins == b)[0] for b in ub}
        null_row = np.empty(B)
        for r in range(B):
            Ir = np.concatenate([rng.choice(pools[b], k, replace=False) for b, k in zip(ub, cnt)])
            null_row[r] = coherence(Z2, Ir, J)
        p_col = (1 + np.sum(null_col >= T)) / (B + 1)
        p_row = (1 + np.sum(null_row >= T)) / (B + 1)
        out.append(dict(I=I, J=J, T=T, p_col=p_col, p_row=p_row, p=max(p_col, p_row), delta_disc=c.get('delta', np.nan), I0=c.get('I0', I), J0=c.get('J0', J)))
    return out


def certify_members(Z1, Z2, I, J, q, rng, B=999):
    """Member-level certification inside a certified bicluster, on the confirmation fold.
    Gene g: projection of its confirmation-fold profile over J on the leave-one-out bicluster profile
    (signs fixed by the discovery-fold loadings), tested against permutations of the gene's own row.
    Condition j: sign-aligned gene average at j, tested against the conditions outside J.
    Benjamini-Hochberg is applied within the bicluster for genes and for conditions separately."""
    I = np.asarray(I); J = np.asarray(J); m = Z2.shape[1]
    u1 = np.linalg.svd(Z1[np.ix_(I, J)], full_matrices=False)[0][:, 0]
    sg = np.sign(u1); sg[sg == 0] = 1
    if np.sum(sg * u1) < 0:
        sg = -sg
    A = Z2[I] * sg[:, None]                               # sign-aligned confirmation rows
    tot = A[:, J].sum(0)
    perms = np.argsort(rng.random((B, m)), 1)[:, :len(J)]  # random condition sets for the row null
    pg = np.empty(len(I))
    for a in range(len(I)):
        prof = (tot - A[a, J]) / max(len(I) - 1, 1)
        prof = prof / (np.linalg.norm(prof) + 1e-12)
        t = A[a, J] @ prof
        tn = A[a][perms] @ prof
        pg[a] = (1 + np.sum(tn >= t)) / (B + 1)
    keep_g, _ = bh(pg, q)
    col = A.mean(0)
    notJ = np.setdiff1d(np.arange(m), J)
    pc = np.array([(1 + np.sum(col[notJ] >= col[j])) / (1 + len(notJ)) for j in J])
    keep_c, _ = bh(pc, q)
    return I[keep_g], J[keep_c], pg, pc


def bh(p, q):
    p = np.asarray(p, float); n = len(p)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0)
    o = np.argsort(p); ranked = p[o] * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(n); qvals[o] = np.minimum(qv, 1)
    return qvals <= q, qvals


def redundant(a, b, jac):
    """Two biclusters are redundant when their cells overlap (Jaccard >= jac) or when they share most genes
    on overlapping conditions (gene overlap coefficient >= 0.3 and condition overlap coefficient >= 0.5)."""
    if jaccard_cells((a['I'], a['J']), (b['I'], b['J'])) >= jac:
        return True
    oi = len(np.intersect1d(a['I'], b['I'])) / min(len(a['I']), len(b['I']))
    oj = len(np.intersect1d(a['J'], b['J'])) / min(len(a['J']), len(b['J']))
    return oi >= 0.3 and oj >= 0.5


def consolidate(cands, sw, cfg):
    """Merge complementary pieces of one module and drop redundant ones (discovery fold only).
    A candidate whose conditions largely coincide with a kept bicluster is merged into it when the union of
    their genes raises the total search-adjusted excess energy; otherwise it is dropped if redundant."""
    tot = sw.obj
    kept = []
    for c in sorted(cands, key=lambda c: -sw.obj(c['I'], c['J'])):
        merged = False
        for k, u in enumerate(kept):
            oj = len(np.intersect1d(c['J'], u['J'])) / min(len(c['J']), len(u['J']))
            if oj >= 0.5:
                Iu = np.union1d(u['I'], c['I'])
                if len(Iu) <= cfg['gmax'] and tot(Iu, u['J']) > tot(u['I'], u['J']):
                    kept[k] = dict(u, I=Iu, delta=sw.score(Iu, u['J'])); merged = True; break
            if redundant(c, u, cfg['jac_admit']):
                merged = True; break
        if not merged:
            kept.append(c)
    return kept


def thin_folds(X, lib, groups, eps, rng):
    phi = estimate_dispersion(X, lib, np.asarray(groups))
    X1, X2 = nb_thin(X, phi, eps, rng)
    return log_cpm(X1, lib * eps), log_cpm(X2, lib * (1 - eps)), phi


def run_hive(X, lib, groups, seed=0, folds=None, **over):
    """folds: optional precomputed (Y1, Y2) so that HIVE-Bic and baselines share one thinning draw."""
    cfg = dict(DEFAULT); cfg.update(over)
    rng = np.random.default_rng(seed)
    X = np.asarray(X); lib = np.asarray(lib, float)
    if folds is not None:
        Y1, Y2 = folds
    elif cfg['use_thinning']:
        Y1, Y2, _ = thin_folds(X, lib, groups, cfg['eps'], rng)
    else:                                               # double dipping: same data for search and test
        Y1 = Y2 = log_cpm(X, lib)
    Z1, Z2 = standardize(Y1), standardize(Y2)
    smax = cfg['smax'] if cfg['smax'] is not None else max(cfg['smin'] + 1, Z1.shape[1] // 2)
    null = NullTable(Z1.shape[0], Z1.shape[1], np.random.default_rng(12345)) if cfg['use_selnull'] else None
    sw = Swarm(Z1, cfg, rng, null)
    archive, hv = sw.run()
    cands = []
    for a in archive[:cfg['n_refine']]:
        I, J = a['I'], a['J']; I0, J0 = I, J
        if cfg['use_boost']:
            for _ in range(cfg['refine_rounds']):
                I2, J2 = (crossfit_membership if cfg['use_crossfit'] else insample_membership)(Z1, I, J, cfg, rng)
                same = len(I2) == len(I) and len(J2) == len(J) and np.all(I2 == I) and np.all(J2 == J)
                if sw.score(I2, J2) > 0:
                    I, J = I2, J2
                if same:
                    break
        d = sw.score(I, J)
        if d > 0:
            cands.append(dict(I=I, J=J, delta=d, I0=I0, J0=J0))
    uniq = consolidate(cands, sw, cfg)
    cert = certify(Z2, uniq, cfg, rng, Y2.mean(1))
    rej, qv = bh([c['p'] for c in cert], cfg['q'])
    for c, r, qq in zip(cert, rej, qv):
        c['certified'] = bool(r); c['q'] = float(qq)
        if r and cfg['member_cert']:
            Ic, Jc, pg, pc = certify_members(Z1, Z2, c['I'], c['J'], cfg['q'], rng)
            c['I_cert'], c['J_cert'] = Ic, Jc
            if len(Ic) < 3 or len(Jc) < 2:
                c['certified'] = False
        else:
            c['I_cert'], c['J_cert'] = c['I'], c['J']
    return dict(candidates=cert, hv=hv, Y1=Y1, Y2=Y2, cfg=cfg, n_eval=sw.n_eval, null=null)
