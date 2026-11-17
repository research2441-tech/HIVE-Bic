"""Statistics and black-and-white figures for the HIVE-Bic experiments."""
import numpy as np, pandas as pd, json, glob, os, sys
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.environ.get('HIVE_OUT', 'results'); F = os.environ.get('HIVE_FIGS', 'figs')
os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'mathtext.fontset': 'stix', 'savefig.dpi': 600})
ORDER = ['HIVE-Bic', 'LAS', 'SSVD', 'SpecCo', 'SpecBic', 'ISA', 'Plaid', 'CC', 'BPSO', 'LBBSO']
LABEL = {'LBBSO': 'LB-BSO', 'SpecCo': 'Spec-Co', 'SpecBic': 'Spec-Bic'}
lab = lambda m: LABEL.get(m, m)
NS = 12


def load(exp):
    rows = [json.loads(l) for f in glob.glob(f'{R}/{exp}_part*.jsonl') for l in open(f)]
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return d
    d['scen'] = np.where(d.background == 'real', 'real-2.0', 'nb-' + d.delta.astype(str))
    base = {'main': None, 'null': 3000, 'ablation': 1030, 'sens': 4000}[exp]
    if exp == 'main':
        d = d[((d.background == 'nb') & (d.seed < 1000 + NS)) | ((d.background == 'real') & (d.seed < 2000 + NS))]
    return d


def flat(df):
    df = df.copy(); df.columns = ['_'.join([c for c in col if c]) if isinstance(col, tuple) else col for col in df.columns]; return df


def ci_mean(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (B, len(x))).mean(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def clopper(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main_tables(d):
    out = {}
    d = d.assign(fdp=np.nan_to_num(d.false_cert.values.astype(float)), fdp_raw=np.nan_to_num(d.false_out.values.astype(float)))
    g = d.groupby(['scen', 'method'])
    for metric in ['fdp', 'fdp_raw', 'rec', 'rel', 'rec_cert', 'rel_cert', 'n_out', 'n_cert', 'false_cert', 'time', 'recc_shift', 'recc_scale', 'recc_anti']:
        out[metric] = g[metric].agg(['mean', 'std']).reset_index()
    return out


def friedman(d, metric):
    d = d[d.background == 'nb']
    P = d.pivot_table(index=['scen', 'seed'], columns='method', values=metric)[ORDER].dropna()
    ranks = P.rank(axis=1, ascending=False).mean()
    chi, p = stats.friedmanchisquare(*[P[c].values for c in ORDER])
    k, N = len(ORDER), len(P)
    q05 = {10: 3.164}[k]                     # Studentized range / sqrt(2), alpha = 0.05, k = 10 (Demsar 2006)
    cd = q05 * np.sqrt(k * (k + 1) / (6 * N))
    return ranks.sort_values(), chi, p, cd, N


def wilcoxon_vs(d, metric, ref='HIVE-Bic'):
    d = d[d.background == 'nb']
    P = d.pivot_table(index=['scen', 'seed'], columns='method', values=metric).dropna()
    res = []
    for m in ORDER[1:]:
        x, y = P[ref].values, P[m].values; diff = x - y
        nz = diff[diff != 0]
        if len(nz) == 0:
            res.append((m, 0.0, 1.0, 0.0, 0.0, 0.0)); continue
        w = stats.wilcoxon(x, y, zero_method='wilcox')
        rp = stats.rankdata(np.abs(nz)); rbc = (rp[nz > 0].sum() - rp[nz < 0].sum()) / rp.sum()
        lo, hi = ci_mean(diff)
        res.append((m, diff.mean(), w.pvalue, rbc, lo, hi))
    r = pd.DataFrame(res, columns=['method', 'diff', 'p', 'rbc', 'lo', 'hi'])
    order = np.argsort(r.p.values); adj = np.empty(len(r)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(r) - i) * r.p.values[j])); adj[j] = run
    r['p_holm'] = adj
    return r


# ------------------------------------------------------------------ figures
def fig_null(n):
    g = n.groupby('method')
    meth = ['HIVE-Bic', 'w/o thinning'] + [m for m in ORDER[1:] if m in set(n.method)]
    frac = [g.get_group(m).any_false_cert.mean() for m in meth]
    k = [g.get_group(m).any_false_cert.sum() for m in meth]; N = [len(g.get_group(m)) for m in meth]
    cis = [clopper(a, b) for a, b in zip(k, N)]
    nout = [g.get_group(m).n_out.mean() for m in meth]
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.6), gridspec_kw=dict(hspace=0.95))
    x = np.arange(len(meth))
    ax[0].bar(x, frac, color=['black', '0.45'] + ['white'] * (len(meth) - 2), edgecolor='black', linewidth=0.6, width=0.7)
    ax[0].errorbar(x, frac, yerr=[np.array(frac) - [c[0] for c in cis], [c[1] for c in cis] - np.array(frac)], fmt='none', ecolor='black', capsize=2, lw=0.6)
    ax[0].axhline(0.10, ls='--', color='black', lw=0.7); ax[0].text(len(meth) - 0.6, 0.115, r'$q=0.10$', ha='right', fontsize=7)
    ax[0].set_ylabel('Matrices with a certified block'); ax[0].set_ylim(0, 1.05)
    ax[0].set_xticks(x); ax[0].set_xticklabels([lab(m) for m in meth], rotation=45, ha='right')
    ax[0].set_title('(a) Empirical FDR on structure-free counts', fontsize=8)
    ax[1].bar(x, nout, color='white', edgecolor='black', hatch='////', linewidth=0.6, width=0.7, label='reported')
    ax[1].bar(x, [g.get_group(m).n_cert.mean() for m in meth], color='black', width=0.7, label='certified')
    ax[1].set_ylabel('Blocks per null matrix'); ax[1].set_xticks(x); ax[1].set_xticklabels([lab(m) for m in meth], rotation=45, ha='right')
    ax[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0)); ax[1].set_title('(b) Blocks reported and certified', fontsize=8)
    ax[1].set_ylim(0, max(nout) * 1.25)
    fig.savefig(f'{F}/Fig2_null.png', bbox_inches='tight'); plt.close(fig)


def fig_main(d):
    scen = ['nb-1.5', 'nb-2.0', 'nb-2.5']
    sl = {'nb-1.5': r'NB, $\delta=1.5$', 'nb-2.0': r'NB, $\delta=2.0$', 'nb-2.5': r'NB, $\delta=2.5$', 'real-2.0': r'Atlas, $\delta=2.0$'}
    mk = ['o', 's', '^', 'D']; fc = ['black', 'white', '0.55', 'white']
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 5.6), sharex=False, gridspec_kw=dict(hspace=0.32))
    y = np.arange(len(ORDER))[::-1]
    for a, metric, title in ((ax[0], 'rec_cert', '(a) Recovery of certified blocks'), (ax[1], 'rel_cert', '(b) Relevance of certified blocks')):
        for s, (sc, m_, f_) in enumerate(zip(scen, mk, fc)):
            sub = d[d.scen == sc].groupby('method')[metric]
            mu = np.array([sub.mean().get(m, np.nan) for m in ORDER]); se = np.array([sub.std().get(m, np.nan) / np.sqrt(sub.count().get(m, 1)) for m in ORDER])
            a.errorbar(np.nan_to_num(mu), y + (s - 1) * 0.22, xerr=1.96 * np.nan_to_num(se), fmt=m_, mfc=f_, mec='black', ecolor='black', ms=3.5, lw=0.6, capsize=1.5, label=sl[sc])
        a.set_title(title, fontsize=8); a.set_xlim(-0.02, 1.0); a.grid(axis='x', color='0.85', lw=0.4)
        for yy in y[:-1]: a.axhline(yy - 0.5, color='0.9', lw=0.4)
    for a in ax: a.set_yticks(y); a.set_yticklabels([lab(m) for m in ORDER])
    ax[0].set_xlabel('Recovery'); ax[1].set_xlabel('Relevance')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(f'{F}/Fig3_main.png', bbox_inches='tight'); plt.close(fig)


def fig_pattern(d):
    M = np.array([[d[d.method == m][f'recc_{p}'].mean() for p in ('shift', 'scale', 'anti')] for m in ORDER])
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    im = ax.imshow(M, cmap='Greys', vmin=0, vmax=max(0.8, M.max()), aspect='auto')
    for i in range(M.shape[0]):
        for j in range(3):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.45 else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Shift', 'Shift-scale', 'Anti-correlated'])
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([lab(m) for m in ORDER])
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.set_label('Certified recovery', fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(f'{F}/Fig4_patterns.png', bbox_inches='tight'); plt.close(fig)


def fig_cd(ranks, cd, fname, title):
    k = len(ranks); lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1); ax.axis('off')
    ax.plot([lo, hi], [0.8, 0.8], color='black', lw=0.7)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0.8, 0.83], color='black', lw=0.6); ax.text(t, 0.86, str(t), ha='center', fontsize=7)
    ax.plot([lo, lo + cd], [0.97, 0.97], color='black', lw=1.2); ax.text(lo + cd / 2, 0.99, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=7)
    items = list(ranks.items()); half = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        left = i < half
        yy = 0.68 - 0.105 * (i if left else (k - 1 - i))
        xe = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xe], [0.8, yy, yy], color='black', lw=0.5)
        ax.text(xe + (-0.05 if left else 0.05), yy, f'{lab(m)} ({r:.2f})', ha='right' if left else 'left', va='center', fontsize=6.5)
    # cliques: groups of methods whose rank range is below CD
    vals = ranks.values; cl = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl): cl.append((i, j))
    for c, (i, j) in enumerate(cl):
        yb = 0.75 - 0.035 * c
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yb, yb], color='black', lw=1.6)
    fig.savefig(f'{F}/{fname}', bbox_inches='tight'); plt.close(fig)


def fig_ablation(a):
    var = ['HIVE-Bic', 'w/o thinning', 'w/o search-adjusted null', 'w/o diversity', 'w/o LogitBoost', 'in-sample LogitBoost', 'w/o polish', 'Pearson objective']
    g = a.groupby('method')
    rc = [g.get_group(v).rec_cert.mean() for v in var]; rl = [g.get_group(v).rel_cert.mean() for v in var]
    rce = [1.96 * g.get_group(v).rec_cert.std() / np.sqrt(len(g.get_group(v))) for v in var]
    rle = [1.96 * g.get_group(v).rel_cert.std() / np.sqrt(len(g.get_group(v))) for v in var]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    y = np.arange(len(var))[::-1]; h = 0.36
    ax.barh(y + h / 2, rc, h, xerr=rce, color='black', ecolor='0.4', error_kw=dict(lw=0.6, capsize=1.5), label='Recovery')
    ax.barh(y - h / 2, rl, h, xerr=rle, color='white', edgecolor='black', hatch='////', lw=0.6, ecolor='0.4', error_kw=dict(lw=0.6, capsize=1.5), label='Relevance')
    ax.set_yticks(y); ax.set_yticklabels(var); ax.set_xlabel('Certified score'); ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc='lower right'); ax.grid(axis='x', color='0.88', lw=0.4)
    fig.savefig(f'{F}/Fig6_ablation.png', bbox_inches='tight'); plt.close(fig)


def fig_sens(s):
    groups = [('n_bees', r'Swarm size $N$'), ('lb_rounds', r'Boosting rounds $M$'), ('eps', r'Thinning fraction $\varepsilon$'), ('lam', r'Diversity weight $\lambda$'), ('tau_g', r'Gene threshold $\tau_g$')]
    fig, axs = plt.subplots(3, 2, figsize=(3.45, 4.4), gridspec_kw=dict(wspace=0.12, hspace=0.75))
    axs = axs.ravel(); axs[5].axis('off')
    for ax, (p, t) in zip(axs, groups):
        sub = s[s.method.str.startswith(p + '=')].copy(); sub['v'] = sub.method.str.split('=').str[1].astype(float)
        gg = sub.groupby('v')
        v = np.array(sorted(gg.groups))
        for metric, mk, fc in (('rec_cert', 'o', 'black'), ('rel_cert', 's', 'white')):
            mu = gg[metric].mean().loc[v].values; se = (gg[metric].std() / np.sqrt(gg[metric].count())).loc[v].values
            ax.errorbar(range(len(v)), mu, yerr=1.96 * se, fmt='-' + mk, color='black', mfc=fc, ms=3.5, lw=0.7, capsize=1.5, label='Recovery' if metric == 'rec_cert' else 'Relevance')
        ax.set_xticks(range(len(v))); ax.set_xticklabels([f'{x:g}' for x in v]); ax.set_xlabel(t); ax.set_ylim(0, 1)
        ax.grid(axis='y', color='0.88', lw=0.4)
    for k in (0, 2, 4): axs[k].set_ylabel('Certified score')
    for k in (1, 3): axs[k].set_yticklabels([])
    h, l = axs[0].get_legend_handles_labels(); axs[5].legend(h, l, frameon=False, loc='center')
    fig.savefig(f'{F}/Fig7_sensitivity.png', bbox_inches='tight'); plt.close(fig)


def fig_conv(d):
    h = d[d.method == 'HIVE-Bic']
    T = max(len(x) for x in h.hv)
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.4), gridspec_kw=dict(hspace=0.75))
    for sc, ls in (('nb-1.5', ':'), ('nb-2.0', '--'), ('nb-2.5', '-'), ('real-2.0', '-.')):
        H = np.full((0, T), np.nan)
        for x in h[h.scen == sc].hv:
            x = np.asarray(x, float); x = x / x[-1] if x[-1] > 0 else x
            row = np.full(T, np.nan); row[:len(x)] = x; row[len(x):] = x[-1]; H = np.vstack([H, row])
        ax[0].plot(np.arange(T), np.nanmedian(H, 0), ls=ls, color='black', lw=0.9, label=sc.replace('nb-', r'NB $\delta=$').replace('real-2.0', r'Atlas $\delta=2.0$'))
    ax[0].set_xlabel('Iteration $t$'); ax[0].set_ylabel(r'$\mathrm{HV}_t/\mathrm{HV}_{\mathrm{final}}$ (median)'); ax[0].legend(frameon=False, loc='lower right')
    ax[0].set_title('(a) Archive hypervolume', fontsize=8); ax[0].grid(color='0.9', lw=0.4)
    tm = d.groupby('method').time.median().reindex(ORDER)
    x = np.arange(len(ORDER))
    ax[1].bar(x, tm.values, color=['black'] + ['white'] * (len(ORDER) - 1), edgecolor='black', lw=0.6)
    ax[1].set_yscale('log'); ax[1].set_ylabel('Median run time (s)'); ax[1].set_xticks(x); ax[1].set_xticklabels([lab(m) for m in ORDER], rotation=45, ha='right')
    ax[1].set_title('(b) Run time per matrix', fontsize=8)
    fig.savefig(f'{F}/Fig8_convergence.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    d = load('main'); res = {}
    if len(d):
        fig_main(d); fig_pattern(d[d.background == 'nb'])
        for metric in ('rec_cert', 'rel_cert'):
            ranks, chi, p, cd, N = friedman(d, metric)
            res[f'friedman_{metric}'] = dict(ranks=ranks.to_dict(), chi=chi, p=p, cd=cd, N=N)
            fig_cd(ranks, cd, f'Fig5_cd_{metric}.png', metric)
            res[f'wilcoxon_{metric}'] = wilcoxon_vs(d, metric).to_dict('records')
        for metric in ('recc_shift', 'recc_scale', 'recc_anti'):
            res[f'wilcoxon_{metric}'] = wilcoxon_vs(d, metric).to_dict('records')
        res['main'] = {k: v.to_dict('records') for k, v in main_tables(d).items()}
        fig_conv(d)
        h = d[d.method == 'HIVE-Bic'].assign(fdp=lambda x: np.nan_to_num(x.false_cert.values.astype(float)))
        res['hive_fdp_ci'] = {sc: (float(g.fdp.mean()),) + ci_mean(g.fdp.values) for sc, g in h.groupby('scen')}
        res['fdp_all'] = {m: float(np.nan_to_num(g.false_cert.values.astype(float)).mean()) for m, g in d[d.background == 'nb'].groupby('method')}
    n = load('null')
    if len(n):
        fig_null(n)
        res['null'] = flat(n.groupby('method')[['n_out', 'n_cert', 'any_false_cert']].agg(['mean', 'sum', 'count']).reset_index()).to_dict('records')
    a = load('ablation')
    if len(a):
        fig_ablation(a); a = a.assign(fdp=np.nan_to_num(a.false_cert.values.astype(float))); res['ablation'] = flat(a.groupby('method')[['rec', 'rel', 'rec_cert', 'rel_cert', 'n_out', 'n_cert', 'fdp', 'time', 'n_eval', 'recc_shift', 'recc_scale', 'recc_anti']].agg(['mean', 'std']).reset_index()).to_dict('records')
        res['ablation_w'] = {}
        for metric in ('rec_cert', 'rel_cert'):
            P = a.pivot_table(index='seed', columns='method', values=metric)
            res['ablation_w'][metric] = {m: float(stats.wilcoxon(P['HIVE-Bic'], P[m]).pvalue) if (P['HIVE-Bic'] != P[m]).any() else 1.0 for m in P.columns if m != 'HIVE-Bic'}
    s = load('sens')
    if len(s):
        fig_sens(s); res['sens'] = flat(s.groupby('method')[['rec_cert', 'rel_cert', 'n_cert', 'time']].agg(['mean', 'std']).reset_index()).to_dict('records')
    json.dump(res, open(f'{R}/summary.json', 'w'), default=float, indent=1)
    print('ok')
