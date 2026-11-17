# HIVE-Bic

**Honest Inference Via thinned Evidence**: selection-honest biclustering of multi-tissue RNA-seq counts.

HIVE-Bic splits one count matrix into two independent folds by negative binomial data thinning. A diversity-constrained binary bee swarm with a search-adjusted coherence objective searches the discovery fold, cross-fitted LogitBoost refines gene and condition membership, and every consolidated bicluster is tested on the confirmation fold with permutation p-values and Benjamini-Hochberg control. Certified pig biclusters are then tested for preservation in human GTEx v10 tissues through one-to-one orthologs.

## Files

| File | Purpose |
|---|---|
| `hive.py` | Thinning, robust scaling, search-adjusted null table, bee swarm, cross-fitted LogitBoost, consolidation, certification (`run_hive`) |
| `synth.py` | Semi-synthetic counts with planted shift, shift-scale and anti-correlated biclusters; recovery and relevance |
| `baselines.py` | CC, ISA, plaid, LAS, SSVD, spectral biclustering and co-clustering, BPSO, LB-BSO reimplementation |
| `download_data.py` | Downloads E-MTAB-5895 counts, NCBI orthologs and gene info, GO sets |
| `make_annotation.py` | Writes the sample annotation of E-MTAB-5895 (tissue, animal, sex, age) |
| `orthologs.py`, `data_prep.py` | One-to-one pig-human orthologs; the 2,000-gene count matrix |
| `gtex_fetch.py` | GTEx v10 log2(TPM+1) for the human orthologs (Hugging Face mirror) |
| `run_synth.py` | Benchmark: `main`, `null`, `ablation`, `sens` |
| `real_data.py`, `real_analysis.py` | Pig atlas study, GO enrichment, GTEx preservation |
| `analysis.py`, `fig_arch.py`, `fig_nulls.py`, `fig_nulltable.py` | Statistics and figures |
| `HIVE_Bic_Colab.ipynb` | Self-contained notebook for the free Colab tier |

## Quick start

```bash
pip install -r requirements.txt
pip install --no-deps biclustlib
export HIVE_DATA=data HIVE_OUT=results HIVE_FIGS=figs
mkdir -p data results figs
python download_data.py && python data_prep.py && python gtex_fetch.py
HIVE_SEEDS=12 python run_synth.py main 0 1      # also: null, ablation, sens
python real_data.py 10 && python real_analysis.py
python analysis.py && python fig_arch.py && python fig_nulls.py && python fig_nulltable.py
```

```python
from hive import run_hive
res = run_hive(counts, library_sizes, tissue_labels, seed=0)   # counts: genes x samples integer array
certified = [c for c in res['candidates'] if c['certified']]
```

Default settings: thinning fraction 0.5, 20 bees, at most 40 iterations, LogitBoost with 50 depth-2 rounds and 5 cross-fitting folds, 499 permutations per null, q = 0.10. One run on the 2,000 x 51 atlas takes about 25 s on two CPU cores.

## Data sources

* E-MTAB-5895 Duroc pig tissue atlas (Expression Atlas, EMBL-EBI)
* NCBI `gene_orthologs` and `gene_info` (Hugging Face mirrors `pankajrajdeo/gene_orthologs`, `huggingworld/ncbi-all-data-gene-info`)
* GO biological process sets (ToppGene; `pankajrajdeo/GeneOntologyBiologicalProcess_ToppGene`)
* GTEx v10 (`silicobio/hoike_normal_expression_GTEx_Analysis_v10_log2tpmplus1`)
