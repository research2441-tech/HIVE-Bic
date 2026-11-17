"""Data preparation for HIVE-Bic: pig tissue atlas E-MTAB-5895 + NCBI one-to-one orthologs."""
import numpy as np, pandas as pd, os, json
D = os.environ.get('HIVE_DATA', 'data')

def load_pig(n_genes=2000, min_cpm=1.0, min_samples=3):
    raw = pd.read_csv(f'{D}/E-MTAB-5895-raw-counts.tsv.undecorated', sep='\t', index_col=0)
    ann = pd.read_csv(f'{D}/E-MTAB-5895_sample_annotation.tsv', sep='\t').set_index('run')
    ann = ann.loc[raw.columns]
    key = ann.tissue + '|' + ann.individual
    # sum sequencing runs of the same biological sample (tissue x animal)
    X = raw.T.groupby(key.values).sum().T
    meta = ann.assign(key=key.values).drop_duplicates('key').set_index('key').loc[X.columns, ['tissue','individual','sex','age']]
    if not os.path.exists(f'{D}/ncbi_pig_human_one2one.tsv'):
        from orthologs import build_one2one
        build_one2one(pd.read_parquet(f'{D}/gene_orthologs.parquet'), pd.read_parquet(f'{D}/gene_info_human_pig.parquet')).to_csv(f'{D}/ncbi_pig_human_one2one.tsv', sep='\t', index=False)
    orth = pd.read_csv(f'{D}/ncbi_pig_human_one2one.tsv', sep='\t')
    X = X.loc[X.index.intersection(orth.p_ens)]
    lib = raw.T.groupby(key.values).sum().T.sum(axis=0).loc[X.columns]  # full library size (all genes)
    cpm = X / lib * 1e6
    keep = (cpm >= min_cpm).sum(axis=1) >= min_samples
    X = X[keep]
    lc = np.log2(X / lib * 1e6 + 1)
    # rank genes by variance of log-CPM; keep top n_genes
    v = lc.var(axis=1).sort_values(ascending=False)
    X = X.loc[v.index[:n_genes]]
    o = orth.set_index('p_ens').loc[X.index]
    genes = pd.DataFrame({'pig_ens': X.index, 'pig_sym': o.p_sym.values, 'human_ens': o.h_ens.values,
                          'human_sym': o.h_sym.values, 'human_geneid': o.h.values})
    meta['lib_size'] = lib.loc[X.columns].values
    return X.astype(np.int64), meta, genes, int(keep.sum())

if __name__ == '__main__':
    X, meta, genes, n_expr = load_pig()
    print(X.shape, n_expr); print(meta.tissue.value_counts().head()); print(meta.lib_size.describe())
    X.to_csv(f'{D}/pig_counts_2000.tsv', sep='\t'); meta.to_csv(f'{D}/pig_samples.tsv', sep='\t'); genes.to_csv(f'{D}/pig_genes_2000.tsv', sep='\t', index=False)
    json.dump({'n_expressed_orthologs': n_expr}, open(f'{D}/prep_info.json','w'))
