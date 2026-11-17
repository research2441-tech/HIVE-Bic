"""Pig tissue atlas (E-MTAB-5895): HIVE-Bic and baselines, GO enrichment and pig-to-human preservation in GTEx v10."""
import numpy as np, pandas as pd, json, os, sys, time
from scipy.stats import hypergeom
from hive import *
from baselines import METHODS
D = os.environ.get('HIVE_DATA', 'data'); OUT = os.environ.get('HIVE_OUT', 'results')

TISSUE_MAP = {'pituitary gland': 'Pituitary', 'cortex of kidney': 'Kidney', 'heart left ventricle': 'Heart', 'omentum': 'Adipose Tissue',
              'spleen': 'Spleen', 'alveolus of lung': 'Lung', 'skeletal muscle tissue': 'Muscle', 'caecum': 'Colon', 'colon': 'Colon',
              'uterus': 'Uterus', 'duodenum': 'Small Intestine', 'ileum': 'Small Intestine', 'stomach': 'Stomach',
              'hippocampal formation': 'Brain', 'amygdala': 'Brain', 'frontal cortex': 'Brain', 'occipital cortex': 'Brain',
              'brainstem': 'Brain', 'medulla oblongata': 'Brain', 'pons': 'Brain', 'cerebellum': 'Brain', 'corpus callosum': 'Brain',
              'dorsal plus ventral thalamus': 'Brain'}


def load():
    X = pd.read_csv(f'{D}/pig_counts_2000.tsv', sep='\t', index_col=0)
    meta = pd.read_csv(f'{D}/pig_samples.tsv', sep='\t', index_col=0)
    genes = pd.read_csv(f'{D}/pig_genes_2000.tsv', sep='\t')
    return X, meta, genes


class GO:
    def __init__(self, genes, min_size=10, max_size=500):
        go = pd.read_parquet(f'{D}/GeneOntologyBiologicalProcess(ToppGene).parquet', columns=['gene_id', 'concept_id', 'concept_name'])
        go['gene_id'] = pd.to_numeric(go.gene_id, errors='coerce')
        uni = set(genes.human_geneid.astype(int))
        go = go[go.gene_id.isin(uni)]
        sizes = go.groupby('concept_id').size()
        keep = sizes[(sizes >= min_size) & (sizes <= max_size)].index
        go = go[go.concept_id.isin(keep)]
        self.names = go.drop_duplicates('concept_id').set_index('concept_id').concept_name
        self.sets = go.groupby('concept_id').gene_id.apply(set)
        self.annotated = set(go.gene_id)
        self.gid = genes.human_geneid.astype(int).values

    def enrich(self, I):
        ids = set(self.gid[I]) & self.annotated
        N = len(self.annotated); n = len(ids)
        if n < 3:
            return None
        res = []
        for t, S in self.sets.items():
            k = len(ids & S)
            if k >= 2:
                res.append((t, k, len(S), hypergeom.sf(k - 1, N, len(S), n)))
        if not res:
            return None
        df = pd.DataFrame(res, columns=['term', 'k', 'size', 'p'])
        m = len(self.sets)                                   # BH over all tested terms (absent terms have p = 1)
        df = df.sort_values('p'); df['q'] = np.minimum(1, df.p * m / (np.arange(len(df)) + 1))
        df['q'] = df.q[::-1].cummin()[::-1]
        df['name'] = df.term.map(self.names)
        return df


class Preservation:
    """Pig-to-human module preservation (density and connectivity Z-scores, Zsummary) in GTEx v10."""
    def __init__(self, genes, Ypig, per_tissue=50, seed=0):
        E = pd.read_parquet(f'{D}/gtex_v10_subset.parquet')
        rng = np.random.default_rng(seed)
        idx = []
        for t, g in E.groupby('tissue_type'):
            idx += list(rng.choice(g.index, min(per_tissue, len(g)), replace=False))
        E = E.loc[idx]
        self.tissue = E.tissue_type.values
        H = E.drop(columns='tissue_type')
        cols = genes.human_ens.values
        self.has = np.array([c in H.columns for c in cols])
        Hm = np.full((len(cols), len(H)), np.nan)
        Hm[self.has] = H[cols[self.has]].values.T
        self.H = Hm
        self.Ch = np.corrcoef(np.nan_to_num(Hm))                      # human gene-gene correlation (all sampled tissues)
        self.Cp = np.corrcoef(Ypig)                                   # pig gene-gene correlation (all 51 samples)
        self.rng = rng
        self.tissues = np.array(sorted(set(self.tissue)))
        self.tprof = np.vstack([np.nanmean(Hm[:, self.tissue == t], 1) for t in self.tissues]).T

    def _stats(self, I, sign):
        Ch = self.Ch[np.ix_(I, I)] * np.outer(sign, sign); Cp = self.Cp[np.ix_(I, I)]
        iu = np.triu_indices(len(I), 1)
        dens = np.mean(Ch[iu])
        conn = np.corrcoef(Cp[iu], self.Ch[np.ix_(I, I)][iu])[0, 1]
        return dens, conn

    def zsummary(self, I, sign, B=200):
        I = np.asarray(I); ok = self.has[I]; I = I[ok]; sign = np.asarray(sign)[ok]
        if len(I) < 5:
            return dict(n_mapped=len(I), z_density=np.nan, z_conn=np.nan, zsummary=np.nan)
        d0, c0 = self._stats(I, sign)
        pool = np.where(self.has)[0]; dn, cn = [], []
        for _ in range(B):
            R = self.rng.choice(pool, len(I), replace=False)
            # null modules keep the first principal sign pattern of their own pig correlation matrix
            s = np.sign(np.linalg.eigh(self.Cp[np.ix_(R, R)])[1][:, -1]); s[s == 0] = 1
            d, c = self._stats(R, s); dn.append(d); cn.append(c)
        zd = (d0 - np.mean(dn)) / np.std(dn); zc = (c0 - np.mean(cn)) / np.std(cn)
        return dict(n_mapped=len(I), density=d0, conn=c0, z_density=zd, z_conn=zc, zsummary=(zd + zc) / 2)

    def tissue_concordance(self, I, sign, pig_tissues):
        I = np.asarray(I); ok = self.has[I]
        if ok.sum() < 3:
            return np.nan, None
        prof = self.tprof[I[ok]]
        prof = (prof - prof.mean(1, keepdims=True)) / (prof.std(1, keepdims=True) + 1e-9)
        score = (prof * np.asarray(sign)[ok][:, None]).mean(0)
        top = self.tissues[np.argmax(score)]
        mapped = {TISSUE_MAP.get(t) for t in pig_tissues} - {None}
        return float(top in mapped) if mapped else np.nan, top


def describe(b, Z, meta, genes, go, pres):
    I, J = np.asarray(b[0]), np.asarray(b[1])
    u = np.linalg.svd(Z[np.ix_(I, J)], full_matrices=False)[0][:, 0]
    sign = np.sign(u); sign[sign == 0] = 1
    if np.sum(sign * np.abs(u)) < 0: sign = -sign
    tissues = list(meta.tissue.values[J])
    e = go.enrich(I)
    top = e.iloc[0] if e is not None else None
    zs = pres.zsummary(I, sign)
    tc, htop = pres.tissue_concordance(I, sign, tissues)
    return dict(n_genes=len(I), n_cond=len(J), tissues=sorted(set(tissues)), animals=len(set(meta.individual.values[J])),
                frac_up=float(np.mean(sign > 0)), go_sig=int(e is not None and (e.q < 0.05).any()),
                go_top=None if top is None else top['name'], go_q=None if top is None else float(top['q']), go_k=None if top is None else int(top['k']),
                n_go_sig=0 if e is None else int((e.q < 0.05).sum()), human_top_tissue=htop, tissue_concord=tc,
                genes=list(genes.pig_sym.values[I][:40]), **zs)


def main(n_runs=10):
    X, meta, genes = load()
    lib = meta.lib_size.values; groups = meta.tissue.values
    go = GO(genes)
    Yfull = log_cpm(X.values, lib)
    pres = Preservation(genes, Yfull)
    rows, runs = [], []
    for r in range(n_runs):
        rng = np.random.default_rng(500 + r)
        Y1, Y2, phi = thin_folds(X.values, lib, groups, 0.5, rng)
        Z1, Z2 = standardize(Y1), standardize(Y2)
        t = time.time(); res = run_hive(X.values, lib, groups, seed=500 + r, folds=(Y1, Y2)); tt = time.time() - t
        cert = [c for c in res['candidates'] if c['certified']]
        runs.append(dict(run=r, n_cand=len(res['candidates']), n_cert=len(cert), time=tt, hv=res['hv'],
                         biclusters=[dict(I=c['I'].tolist(), J=c['J'].tolist(), T=c['T'], p=c['p'], q=c['q'], delta=c['delta_disc']) for c in cert],
                         uncert=[dict(I=c['I'].tolist(), J=c['J'].tolist(), T=c['T'], p=c['p'], q=c['q'], delta=c['delta_disc']) for c in res['candidates'] if not c['certified']]))
        for c in cert:
            d = describe((c['I'], c['J']), Z1, meta, genes, go, pres); d.update(method='HIVE-Bic', run=r, certified=1, T=c['T'], q=c['q']); rows.append(d)
        if r == 0:                                       # baselines on the same folds as run 0
            cfg = dict(DEFAULT); cfg['n_perm'] = 499
            for name, f in METHODS.items():
                t = time.time(); found = f(Y1, Z1, np.random.default_rng(500)); tt2 = time.time() - t
                cr = certify(Z2, [dict(I=I, J=J) for I, J in found], cfg, np.random.default_rng(507), Y2.mean(1))
                rej, qv = bh([c['p'] for c in cr], cfg['q'])
                for (I, J), c, k, qq in zip(found, cr, rej, qv):
                    d = describe((I, J), Z1, meta, genes, go, pres); d.update(method=name, run=0, certified=int(k), T=c['T'], q=float(qq), time=tt2); rows.append(d)
            # HIVE uncertified candidates, described for contrast
            for c in res['candidates']:
                if not c['certified']:
                    d = describe((c['I'], c['J']), Z1, meta, genes, go, pres); d.update(method='HIVE-Bic', run=0, certified=0, T=c['T'], q=c['q']); rows.append(d)
        print('run', r, 'cert', len(cert), 'time', round(tt, 1), flush=True)
        json.dump(runs, open(f'{OUT}/real_runs.json', 'w'), default=float)
        pd.DataFrame(rows).to_json(f'{OUT}/real_biclusters.json', orient='records', default_handler=str)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
