"""Fig. 2: search-adjusted null level of the rank-one share versus the random-matrix edge for a fixed block."""
import numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hive import NullTable
F = os.environ.get('HIVE_FIGS', 'figs'); R = os.environ.get('HIVE_OUT', 'results')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'font.size': 7,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix', 'savefig.dpi': 600})
nt = NullTable(2000, 51, np.random.default_rng(12345))
fig, ax = plt.subplots(figsize=(3.45, 2.5))
out = {}
for l, ls, mk in ((5, '-', 'o'), (10, '--', 's'), (22, ':', '^')):
    ks = nt.ks.astype(int)
    mu = np.array([nt(k, l)[0] for k in ks]); sd = np.array([nt(k, l)[1] for k in ks])
    edge = (1 / np.sqrt(ks) + 1 / np.sqrt(l)) ** 2
    ax.errorbar(ks, mu, yerr=sd, fmt=ls + mk, color='black', mfc='white', ms=3, lw=0.8, capsize=1.5, label=rf'searched, $|J|={l}$')
    ax.plot(ks, edge, ls, color='0.6', lw=1.2)
    out[l] = dict(k=ks.tolist(), mu=mu.tolist(), sd=sd.tolist(), edge=edge.tolist())
ax.plot([], [], color='0.6', lw=1.2, label='random-matrix edge, fixed block')
ax.set_xscale('log'); ax.set_xlabel(r'Number of genes $|I|$'); ax.set_ylabel(r'Rank-one share $\rho$ on noise')
ax.legend(frameon=False, fontsize=6, loc='upper right'); ax.grid(color='0.9', lw=0.4)
fig.savefig(f'{F}/Fig_nulltable.png', bbox_inches='tight'); plt.close(fig)
json.dump(out, open(f'{R}/nulltable.json', 'w'), indent=1)
print({l: (round(v['mu'][0], 3), round(v['edge'][0], 3), round(v['mu'][-1], 3), round(v['edge'][-1], 3)) for l, v in out.items()})
