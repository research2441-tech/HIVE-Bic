"""Fetch GTEx v10 log2(TPM+1) columns for the human orthologs of the pig gene universe (Hugging Face mirror)."""
import pandas as pd, pyarrow.parquet as pq, os, sys
from huggingface_hub import HfFileSystem
D = os.environ.get('HIVE_DATA', 'data')
REPO = 'datasets/silicobio/hoike_normal_expression_GTEx_Analysis_v10_log2tpmplus1/'
F = 'GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_tpm_non_lcm_log2tpmplus1.parquet'
TISSUES = ['Brain','Heart','Muscle','Lung','Spleen','Kidney','Pituitary','Stomach','Colon','Small Intestine','Adipose Tissue','Uterus']
def main():
    genes = pd.read_csv(f'{D}/pig_genes_2000.tsv', sep='\t')
    fs = HfFileSystem(); pf = pq.ParquetFile(fs.open(REPO + F))
    cols = [c for c in genes.human_ens if c in set(pf.schema_arrow.names)]
    meta = pf.read(columns=['sample_id','tissue_type']).to_pandas()
    keep = meta.tissue_type.isin(TISSUES).values
    out = []
    for i in range(0, len(cols), 200):
        t = pf.read(columns=cols[i:i+200]).to_pandas()
        out.append(t[keep].astype('float32')); print(i, flush=True)
    E = pd.concat(out, axis=1); E.index = meta.sample_id[keep].values
    E.insert(0, 'tissue_type', meta.tissue_type[keep].values)
    E.to_parquet(f'{D}/gtex_v10_subset.parquet'); print(E.shape)
if __name__ == '__main__': main()
