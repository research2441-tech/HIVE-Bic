"""One-to-one pig-human orthologs from NCBI gene_orthologs + gene_info (Ensembl IDs from dbXrefs)."""
import pandas as pd

def ens_of(dbx):
    for t in str(dbx).split('|'):
        if t.startswith('Ensembl:'):
            return t.split(':', 1)[1]
    return None

def build_one2one(orth, info):
    o = orth.copy()
    for c in ['#tax_id', 'GeneID', 'Other_tax_id', 'Other_GeneID']: o[c] = pd.to_numeric(o[c], errors='coerce')
    o = o[(o['#tax_id'] == 9606) & (o.Other_tax_id == 9823) & (o.relationship == 'Ortholog')]
    o = o.rename(columns={'GeneID': 'h', 'Other_GeneID': 'p'})[['h', 'p']].dropna().astype(int)
    o = o[~o.h.duplicated(keep=False) & ~o.p.duplicated(keep=False)]
    info = info.copy(); info['GeneID'] = pd.to_numeric(info.GeneID); info['tax_id'] = pd.to_numeric(info.tax_id)
    info['ens'] = info.dbXrefs.map(ens_of)
    hs = info[info.tax_id == 9606].set_index('GeneID'); ps = info[info.tax_id == 9823].set_index('GeneID')
    o = o[o.h.isin(hs.index) & o.p.isin(ps.index)]
    o = o.assign(h_ens=hs.ens.reindex(o.h).values, h_sym=hs.Symbol.reindex(o.h).values,
                 p_ens=ps.ens.reindex(o.p).values, p_sym=ps.Symbol.reindex(o.p).values)
    o = o.dropna(subset=['h_ens', 'p_ens'])
    o = o[~o.h_ens.duplicated(keep=False) & ~o.p_ens.duplicated(keep=False)]
    return o.sort_values('h').reset_index(drop=True)
