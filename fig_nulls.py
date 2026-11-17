"""Fig. 10: permutation nulls of the certification statistic, double-dipped versus held-out, on a null matrix and on the atlas."""
import numpy as np, pandas as pd, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hive import *
from synth import make_matrix
D = os.environ.get('HIVE_DATA', 'data'); R = os.environ.get('HIVE_OUT', 'results'); F = os.environ.get('HIVE_FIGS', 'figs')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix', 'savefig.dpi': 600})


def nulls(Z, I, J, rng, bins, B=499):
    m = Z.shape[1]
    col = np.array([coherence(Z, I, rng.choice(m, len(J), replace=False)) for _ in range(B)])
    ub, cnt = np.unique(bins[I], return_counts=True); pools = {b: np.where(bins == b)[0] for b in ub}
    row = np.array([coherence(Z, np.concatenate([rng.choice(pools[b], k, replace=False) for b, k in zip(ub, cnt)]), J) for _ in range(B)])
    return col, row


def main():
    X = pd.read_csv(f'{D}/pig_counts_2000.tsv', sep='\t', index_col=0); meta = pd.read_csv(f'{D}/pig_samples.tsv', sep='\t', index_col=0)
    lib = meta.lib_size.values; groups = meta.tissue.values
    phi = estimate_dispersion(X.values, lib, groups); base = (X.values / size_factors(lib)).mean(1)
    rng = np.random.default_rng(3000)
    Xs, _ = make_matrix(base, phi, lib, rng, 0.0)
    Y1, Y2, _ = thin_folds(Xs, lib, groups, 0.5, rng)
    res = run_hive(Xs, lib, groups, seed=3000, folds=(Y1, Y2))
    Z1, Z2 = standardize(Y1), standardize(Y2)
    c = max(res['candidates'], key=lambda c: c['delta_disc'])
    I, J = c['I'], c['J']
    r = np.random.default_rng(1)
    b1 = np.digitize(Y1.mean(1), np.quantile(Y1.mean(1), np.linspace(0, 1, 11)[1:-1]))
    b2 = np.digitize(Y2.mean(1), np.quantile(Y2.mean(1), np.linspace(0, 1, 11)[1:-1]))
    col1, row1 = nulls(Z1, I, J, r, b1); col2, row2 = nulls(Z2, I, J, r, b2)
    t1, t2 = coherence(Z1, I, J), coherence(Z2, I, J)
    runs = json.load(open(f'{R}/real_runs.json'))
    Xr = X.values; rr = np.random.default_rng(500)
    Yr1, Yr2, _ = thin_folds(Xr, lib, groups, 0.5, rr)
    Zr2 = standardize(Yr2)
    br = np.digitize(Yr2.mean(1), np.quantile(Yr2.mean(1), np.linspace(0, 1, 11)[1:-1]))
    bm = runs[0]['biclusters'][3]
    Im, Jm = np.array(bm['I']), np.array(bm['J'])
    colm, rowm = nulls(Zr2, Im, Jm, np.random.default_rng(2), br)
    tm = coherence(Zr2, Im, Jm)
    out = dict(null_block=dict(n_genes=len(I), n_cond=len(J), T_disc=t1, T_conf=t2, p_disc=float(max((1 + (col1 >= t1).sum()) / 500, (1 + (row1 >= t1).sum()) / 500)),
                               p_conf=float(max((1 + (col2 >= t2).sum()) / 500, (1 + (row2 >= t2).sum()) / 500)), S=c['delta_disc']),
               muscle=dict(T=tm, max_col=float(colm.max()), max_row=float(rowm.max())))
    json.dump(out, open(f'{R}/fig10_nulls.json', 'w'), indent=1, default=float)
    fig, ax = plt.subplots(3, 1, figsize=(3.45, 4.6), gridspec_kw=dict(hspace=1.0))
    panels = [(ax[0], col1, row1, t1, '(a) Null matrix, block tested on its discovery fold'),
              (ax[1], col2, row2, t2, '(b) Same block tested on the confirmation fold'),
              (ax[2], colm, rowm, tm, '(c) Atlas muscle module (M4), confirmation fold')]
    for a, cn, rn, t, title in panels:
        lo = min(cn.min(), rn.min(), t); hi = max(cn.max(), rn.max(), t)
        bins = np.linspace(lo, hi + 1e-9, 40)
        a.hist(cn, bins=bins, color='0.55', edgecolor='black', lw=0.3, label='condition null')
        a.hist(rn, bins=bins, histtype='step', color='black', lw=0.9, hatch='////', label='gene null')
        a.axvline(t, color='black', lw=1.2, ls='--')
        a.text(t, a.get_ylim()[1] * 1.01, r'$T_{\mathrm{obs}}$', ha='center', va='bottom', fontsize=6.5, clip_on=False)
        a.set_title(title, fontsize=7.5, pad=10); a.set_ylabel('Count'); a.set_xlabel(r'Edge-corrected share $T$')
    ax[0].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.45, 1.0))
    fig.savefig(f'{F}/Fig10_nulls.png', bbox_inches='tight'); plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
