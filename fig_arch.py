"""Fig. 1: HIVE-Bic architecture (grayscale, column width)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('HIVE_FIGS', 'figs'), 'Fig1_architecture.png')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(figsize=(3.5, 5.6), dpi=600)
ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis('off')
FS = 5.4
boxes = {}

def box(key, x, y, w, h, text, fill='white', lw=0.6, ls='-', bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18', fc=fill, ec='black', lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=FS, fontweight='bold' if bold else 'normal', linespacing=1.15)
    boxes[key] = (x, y, w, h)

def pt(key, side, frac=0.5):
    x, y, w, h = boxes[key]
    return {'top': (x + frac * w, y + h), 'bottom': (x + frac * w, y), 'left': (x, y + frac * h), 'right': (x + w, y + frac * h)}[side]

def arrow(p0, p1, ls='-', cs='arc3', lw=0.7):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6, lw=lw, ls=ls, color='black', connectionstyle=cs, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

# shared trunk
box('in', 0.4, 14.55, 9.2, 1.15, 'RNA-seq counts $X\\in\\mathbb{N}^{n\\times m}$ (genes $\\times$ tissue samples),\nlibrary sizes $L_j$, replicate groups', fill='#E6E6E6', bold=False)
box('disp', 0.4, 13.05, 9.2, 0.95, 'Dispersion $\\hat\\phi_g$: pooled within-tissue moments, shrunk on log scale')
box('thin', 0.4, 11.55, 9.2, 0.95, 'Negative-binomial data thinning ($\\varepsilon=0.5$): $X=X^{(1)}+X^{(2)}$', fill='#E6E6E6')
# left: discovery
box('z1', 0.4, 9.85, 4.3, 1.05, 'Discovery fold $X^{(1)}$\nlog-CPM, robust row scaling $\\to Z^{(1)}$')
box('sw', 0.4, 7.75, 4.3, 1.55, 'Binary bee swarm\n(employed / onlooker / scout)\nsigmoid transfer + polish\nobjective $S(I,J)$: search-adjusted $z$')
box('ar', 0.4, 6.15, 4.3, 1.05, 'Elite archive: redundancy filter,\nhypervolume stopping rule')
box('lb', 0.4, 4.45, 4.3, 1.15, 'Cross-fitted LogitBoost\nmembership of genes and\nconditions (out-of-fold $\\hat\\pi$)')
box('co', 0.4, 2.95, 4.3, 0.95, 'Consolidation $\\to$ candidates $(I,J)$')
# right: confirmation
box('z2', 5.3, 9.85, 4.3, 1.05, 'Confirmation fold $X^{(2)}$\nlog-CPM, robust row scaling $\\to Z^{(2)}$')
box('st', 5.3, 2.95, 4.3, 0.95, 'Statistic $T(I,J)$ on $Z^{(2)}$')
box('pn', 5.3, 1.6, 4.3, 0.85, 'Column + matched-gene\npermutation nulls, $p=\\max(p_{\\mathrm{col}},p_{\\mathrm{row}})$')
box('bh', 5.3, 0.25, 4.3, 0.85, 'Benjamini$-$Hochberg ($q=0.10$)\n$\\to$ certified biclusters', fill='#E6E6E6')
box('val', 0.4, 0.25, 4.3, 2.1, 'Biological checks\n(independent of search):\nGO enrichment; pig$\\to$human\npreservation in GTEx v10\n($Z_{\\mathrm{summary}}$), tissue concordance', fill='white', ls='--')

arrow(pt('in', 'bottom'), pt('disp', 'top'))
arrow(pt('disp', 'bottom'), pt('thin', 'top'))
arrow(pt('thin', 'bottom', 0.2337), pt('z1', 'top'))
arrow(pt('thin', 'bottom', 0.7663), pt('z2', 'top'))
arrow(pt('z1', 'bottom'), pt('sw', 'top'))
arrow(pt('sw', 'bottom', 0.35), pt('ar', 'top', 0.35))
# dashed feedback: archive -> swarm (leaders, tabu anchors)
arrow(pt('ar', 'top', 0.72), pt('sw', 'bottom', 0.72), ls='--')
ax.text(pt('ar', 'top', 0.72)[0] + 0.12, (pt('ar', 'top')[1] + pt('sw', 'bottom')[1]) / 2, 'leaders,\ntabu', fontsize=4.6, va='center', ha='left', style='italic')
arrow(pt('ar', 'bottom'), pt('lb', 'top'))
arrow(pt('lb', 'bottom'), pt('co', 'top'))
arrow(pt('co', 'right'), pt('st', 'left'))
arrow(pt('z2', 'bottom'), pt('st', 'top'))
ax.text(pt('z2', 'bottom')[0] + 0.15, 6.4, 'held out from\nsearch and\nrefinement', fontsize=4.8, ha='left', va='center', style='italic')
arrow(pt('st', 'bottom'), pt('pn', 'top'))
arrow(pt('pn', 'bottom'), pt('bh', 'top'))
arrow(pt('bh', 'left'), pt('val', 'right', 0.135))
fig.savefig(OUT, dpi=600, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT)
