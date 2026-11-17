"""Semi-synthetic benchmark: HIVE-Bic, its ablations and nine baselines on shared thinning folds."""
import numpy as np, pandas as pd, sys, json, time, os
from hive import *
from synth import *
from baselines import METHODS
D = os.environ.get('HIVE_DATA', 'data'); OUT = os.environ.get('HIVE_OUT', 'results')
X = pd.read_csv(f'{D}/pig_counts_2000.tsv', sep='\t', index_col=0); meta = pd.read_csv(f'{D}/pig_samples.tsv', sep='\t', index_col=0)
lib = meta.lib_size.values; groups = meta.tissue.values
phi = estimate_dispersion(X.values, lib, groups); base = (X.values / size_factors(lib)).mean(1)

def false_frac(found, truth):
    if not found: return np.nan
    f = 0
    for I, J in found:
        ov = max(overlap_cells((I, J), (t[0], t[1])) for t in truth) if truth else 0.0
        f += ov < 0.1
    return f / len(found)

def evaluate(name, found, cert_flags, truth, t_used, extra=None):
    rec, rel, each = match_scores(found, truth) if truth else (np.nan, np.nan, [])
    fc = [b for b, c in zip(found, cert_flags) if c]
    recc, relc, eachc = match_scores(fc, truth) if truth else (np.nan, np.nan, [])
    r = dict(method=name, n_out=len(found), n_cert=len(fc), rec=rec, rel=rel, rec_cert=recc, rel_cert=relc,
             false_out=false_frac(found, truth), false_cert=false_frac(fc, truth), time=t_used,
             any_false_cert=int(any(max([overlap_cells(b, (t[0], t[1])) for t in truth] or [0]) < 0.1 for b in fc)))
    if truth:
        for (I, J, kind), e, ec in zip(truth, each if len(each) else [0]*3, eachc if len(eachc) else [0]*3):
            r[f'rec_{kind}'] = float(e); r[f'recc_{kind}'] = float(ec)
    if extra: r.update(extra)
    return r

def run_one(scenario, background, delta, seed, variants, baselines=True):
    rng = np.random.default_rng(seed)
    if delta > 0:
        Xs, truth = make_matrix(base, phi, lib, rng, delta, background, Xreal=X.values)
    else:
        Xs, truth = make_matrix(base, phi, lib, rng, 0.0, background, Xreal=X.values); truth = []
    Y1, Y2, _ = thin_folds(Xs, lib, groups, 0.5, rng)
    Z1, Z2 = standardize(Y1), standardize(Y2)
    rows = []
    for vname, over in variants.items():
        t = time.time()
        if over.get('use_thinning', True) is False or 'eps' in over:
            res = run_hive(Xs, lib, groups, seed=seed, **over)
        else:
            res = run_hive(Xs, lib, groups, seed=seed, folds=(Y1, Y2), **over)
        tt = time.time() - t
        found = [(c['I'], c['J']) for c in res['candidates']]
        flags = [c['certified'] for c in res['candidates']]
        rows.append(evaluate(vname, found, flags, truth, tt, dict(n_iter=len(res['hv']) - 1, n_eval=res['n_eval'], hv=res['hv'])))
    if baselines:
        cfg = dict(DEFAULT); cfg['n_perm'] = 199
        for name, f in METHODS.items():
            t = time.time(); found = f(Y1, Z1, np.random.default_rng(seed)); tt = time.time() - t
            cert = certify(Z2, [dict(I=I, J=J) for I, J in found], cfg, np.random.default_rng(seed + 7), Y2.mean(1))
            rej, _ = bh([c['p'] for c in cert], cfg['q'])
            rows.append(evaluate(name, found, list(rej), truth, tt))
    for r in rows:
        r.update(scenario=scenario, background=background, delta=delta, seed=seed)
    return rows

if __name__ == '__main__':
    exp = sys.argv[1]; part = int(sys.argv[2]); nparts = int(sys.argv[3])
    jobs = []
    NS = int(os.environ.get('HIVE_SEEDS', 12))
    FULL = {'HIVE-Bic': {}}
    ABL = {'HIVE-Bic': {}, 'w/o thinning': dict(use_thinning=False), 'w/o search-adjusted null': dict(use_selnull=False),
           'w/o diversity': dict(use_diversity=False), 'w/o LogitBoost': dict(use_boost=False),
           'in-sample LogitBoost': dict(use_crossfit=False), 'w/o polish': dict(polish=0),
           'Pearson objective': dict(objective='pearson', use_selnull=False)}
    if exp == 'main':
        for d in (1.5, 2.0, 2.5):
            for s in range(NS): jobs.append(("main", "nb", d, 1000 + s, FULL, True))
        for s in range(NS): jobs.append(("main", "real", 2.0, 2000 + s, FULL, True))
    elif exp == 'null':
        for s in range(NS): jobs.append(("null", "nb", 0.0, 3000 + s, {'HIVE-Bic': {}, 'w/o thinning': dict(use_thinning=False)}, True))
    elif exp == 'ablation':
        for s in range(NS): jobs.append(("ablation", "nb", 2.0, 1000 + 30 + s, ABL, False))
    elif exp == 'sens':
        grid = {}
        for v in (10, 20, 40): grid[f'n_bees={v}'] = dict(n_bees=v)
        for v in (10, 25, 50, 100): grid[f'lb_rounds={v}'] = dict(lb_rounds=v)
        for v in (0.3, 0.5, 0.7): grid[f'eps={v}'] = dict(eps=v)
        for v in (0.0, 0.5, 1.0): grid[f'lam={v}'] = dict(lam=v)
        for v in (0.3, 0.4, 0.5): grid[f'tau_g={v}'] = dict(tau_g=v)
        for s in range(max(2, NS // 2)): jobs.append(('sens', 'nb', 2.0, 4000 + s, grid, False))
    jobs = jobs[part::nparts]
    fn = f'{OUT}/{exp}_part{part}.jsonl'
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l); done.add((r['scenario'], r['background'], r['delta'], r['seed']))
    for (sc, bg, d, s, var, bl) in jobs:
        if (sc, bg, d, s) in done: continue
        rows = run_one(sc, bg, d, s, var, bl)
        with open(fn, 'a') as f:
            for r in rows: f.write(json.dumps(r, default=float) + '\n')
        print(sc, bg, d, s, 'done', flush=True)
