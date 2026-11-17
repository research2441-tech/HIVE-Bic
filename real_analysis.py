"""Real-data summaries: recurrence over thinning draws, baseline comparison and Fig. 9."""
import numpy as np, pandas as pd, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hive import log_cpm, standardize
R = os.environ.get('HIVE_OUT', 'results'); F = os.environ.get('HIVE_FIGS', 'figs')
D = os.environ.get('HIVE_DATA', 'data')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix', 'savefig.dpi': 600})
NAMES = ['Neural (M1)', 'Intestinal epithelium (M2)', 'Lymphoid / B cell (M3)', 'Striated muscle (M4)']


def jac(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b)


def main():
    runs = json.load(open(f'{R}/real_runs.json'))
    d = pd.read_json(f'{R}/real_biclusters.json')
    X = pd.read_csv(f'{D}/pig_counts_2000.tsv', sep='\t', index_col=0); meta = pd.read_csv(f'{D}/pig_samples.tsv', sep='\t', index_col=0)
    Z = standardize(log_cpm(X.values, meta.lib_size.values))
    ref = runs[0]['biclusters']
    out = dict(n_cert_runs=[r['n_cert'] for r in runs], n_cand_runs=[r['n_cand'] for r in runs], time=[r['time'] for r in runs],
               iters=[len(r['hv']) - 1 for r in runs])
    rec = []
    for b in ref:
        js = [max(jac(b['I'], c['I']) for c in runs[t]['biclusters']) for t in range(1, len(runs))]
        jc = [max(jac(b['J'], c['J']) for c in runs[t]['biclusters']) for t in range(1, len(runs))]
        rec.append(dict(n_found=int(sum(j >= 0.3 for j in js)) + 1, gene_jac=float(np.mean(js)), cond_jac=float(np.mean(jc))))
    out['recurrence'] = rec
    h = d[(d.method == 'HIVE-Bic') & (d.certified == 1)]
    out['hive_all'] = dict(n=len(h), go_sig=float(h.go_sig.mean()), z_med=float(h.zsummary.median()), z_gt2=float((h.zsummary > 2).mean()),
                           zd_med=float(h.z_density.median()), tc=float(h.tissue_concord.mean()))
    h0 = h[h.run == 0].reset_index(drop=True)
    out['run0'] = h0[['n_genes', 'n_cond', 'tissues', 'animals', 'go_top', 'go_q', 'go_k', 'n_go_sig', 'z_density', 'z_conn', 'zsummary', 'human_top_tissue', 'tissue_concord', 'q', 'T', 'genes']].to_dict('records')
    rows = []
    for m, g in d[d.run == 0].groupby('method'):
        c = g[g.certified == 1]
        rows.append(dict(method=m, n=len(g), cert=int(g.certified.sum()), go_all=float(g.go_sig.mean()),
                         go_cert=float(c.go_sig.mean()) if len(c) else np.nan, z_cert=float(c.zsummary.median()) if len(c) else np.nan,
                         tc_cert=float(c.tissue_concord.mean()) if len(c) else np.nan, z_all=float(g.zsummary.median()),
                         genes=float(g.n_genes.median()), conds=float(g.n_cond.median())))
    out['baselines'] = rows
    json.dump(out, open(f'{R}/real_summary.json', 'w'), indent=1, default=float)

    # ---- Fig. 9: tissue activity of run-0 modules and preservation statistics
    tissues = meta.tissue.values
    order = ['frontal cortex', 'occipital cortex', 'hippocampal formation', 'amygdala', 'dorsal plus ventral thalamus', 'corpus callosum', 'brainstem',
             'pons', 'medulla oblongata', 'cerebellum', 'pituitary gland', 'heart left ventricle', 'skeletal muscle tissue', 'stomach', 'duodenum', 'ileum',
             'caecum', 'colon', 'cortex of kidney', 'alveolus of lung', 'spleen', 'mesenteric lymph node', 'tonsil', 'omentum', 'uterus', 'epididymis', 'penis']
    A = np.zeros((len(ref), len(order))); Jm = np.zeros_like(A)
    for k, b in enumerate(ref):
        I, J = np.array(b['I']), np.array(b['J'])
        u = np.linalg.svd(Z[np.ix_(I, J)], full_matrices=False)[0][:, 0]
        s = np.sign(u); s[s == 0] = 1
        if np.sum(s * np.abs(u)) < 0: s = -s
        prof = (Z[I] * s[:, None]).mean(0)
        for t, name in enumerate(order):
            idx = np.where(tissues == name)[0]
            A[k, t] = prof[idx].mean(); Jm[k, t] = np.mean(np.isin(idx, J))
    fig = plt.figure(figsize=(3.45, 4.3))
    ax = fig.add_axes([0.30, 0.50, 0.66, 0.40])
    im = ax.imshow(A, cmap='Greys', aspect='auto', vmin=min(0, A.min()), vmax=A.max())
    for k in range(A.shape[0]):
        for t in range(A.shape[1]):
            if Jm[k, t] > 0:
                ax.plot(t, k, marker='o', ms=2.2, mfc='white', mec='black', mew=0.4)
    ax.set_yticks(range(len(ref))); ax.set_yticklabels(NAMES[:len(ref)], fontsize=6.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5.5)
    ax.set_title('(a) Module activity across pig tissues', fontsize=7.5)
    cax = fig.add_axes([0.30, 0.955, 0.66, 0.012]); cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cax.xaxis.set_ticks_position('top'); cb.set_label('sign-aligned mean z of module genes', fontsize=5.5, labelpad=-16)
    ax2 = fig.add_axes([0.30, 0.02, 0.66, 0.17])
    zd = [r['z_density'] for r in out['run0']]; zc = [r['z_conn'] for r in out['run0']]; zs = [r['zsummary'] for r in out['run0']]
    y = np.arange(len(zd))[::-1]; hh = 0.26
    ax2.barh(y + hh, zd, hh, color='black', label=r'$Z_{\mathrm{density}}$')
    ax2.barh(y, zc, hh, color='white', edgecolor='black', hatch='////', lw=0.5, label=r'$Z_{\mathrm{connectivity}}$')
    ax2.barh(y - hh, zs, hh, color='0.6', edgecolor='black', lw=0.5, label=r'$Z_{\mathrm{summary}}$')
    for v in (2, 10): ax2.axvline(v, ls='--', color='black', lw=0.5)
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(NAMES[:len(zd)], fontsize=6.5); ax2.set_xlabel('Preservation Z in GTEx v10', fontsize=7)
    ax2.set_title('(b) Pig-to-human preservation', fontsize=7.5)
    ax2.legend(frameon=False, fontsize=5.5, ncol=3, loc='lower center', bbox_to_anchor=(0.35, -0.95))
    fig.savefig(f'{F}/Fig9_real.png', bbox_inches='tight'); plt.close(fig)
    print(json.dumps({k: out[k] for k in ('recurrence', 'hive_all', 'n_cert_runs')}, default=float))


if __name__ == '__main__':
    main()
