"""Download every input of HIVE-Bic into $HIVE_DATA (default ./data).
Sources: Expression Atlas FTP (E-MTAB-5895 raw counts), Hugging Face mirrors of NCBI gene_orthologs / gene_info,
ToppGene GO biological process sets and GTEx v10 log2(TPM+1)."""
import os, sys, subprocess, urllib.request
import pandas as pd
from huggingface_hub import hf_hub_download
D = os.environ.get('HIVE_DATA', 'data'); os.makedirs(D, exist_ok=True); os.environ['HIVE_DATA'] = D
COUNTS = 'E-MTAB-5895-raw-counts.tsv.undecorated'
URLS = ['https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/E-MTAB-5895/' + COUNTS,
        'http://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/E-MTAB-5895/' + COUNTS]

def get_counts():
    dst = f'{D}/{COUNTS}'
    if os.path.exists(dst): return
    for u in URLS:
        try:
            urllib.request.urlretrieve(u, dst); print('counts from', u); return
        except Exception as e:
            print('failed', u, e)
    sys.exit('Download the raw counts of E-MTAB-5895 from https://www.ebi.ac.uk/gxa/experiments/E-MTAB-5895/Downloads '
             f'and place {COUNTS} in {D}/, then re-run.')

def hf(repo, fn, out=None):
    p = hf_hub_download(repo, fn, repo_type='dataset')
    if out: os.replace(p, f'{D}/{out}') if not os.path.exists(f'{D}/{out}') else None
    return p

if __name__ == '__main__':
    get_counts()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_annotation.py')], check=True)
    if not os.path.exists(f'{D}/gene_orthologs.parquet'):
        pd.read_parquet(hf('pankajrajdeo/gene_orthologs', 'gene_orthologs.parquet')).to_parquet(f'{D}/gene_orthologs.parquet')
    if not os.path.exists(f'{D}/gene_info_human_pig.parquet'):
        parts = []
        for k in (27, 30):
            g = pd.read_parquet(hf('huggingworld/ncbi-all-data-gene-info', f'data/all-data-gene-info_part_{k:05d}.parquet'),
                                columns=['tax_id', 'GeneID', 'Symbol', 'dbXrefs', 'type_of_gene'])
            g['tax_id'] = pd.to_numeric(g.tax_id, errors='coerce'); parts.append(g[g.tax_id.isin([9606, 9823])])
        pd.concat(parts).to_parquet(f'{D}/gene_info_human_pig.parquet')
    if not os.path.exists(f'{D}/GeneOntologyBiologicalProcess(ToppGene).parquet'):
        pd.read_parquet(hf('pankajrajdeo/GeneOntologyBiologicalProcess_ToppGene', 'GeneOntologyBiologicalProcess(ToppGene).parquet')).to_parquet(f'{D}/GeneOntologyBiologicalProcess(ToppGene).parquet')
    print('downloads complete')
