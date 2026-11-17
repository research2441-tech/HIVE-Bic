"""Build HIVE_Bic_Colab.ipynb: every module is written by %%writefile so the notebook runs on its own in Colab."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
MODS = ['hive.py', 'synth.py', 'baselines.py', 'orthologs.py', 'make_annotation.py', 'download_data.py', 'data_prep.py',
        'gtex_fetch.py', 'run_synth.py', 'real_data.py', 'real_analysis.py', 'analysis.py', 'fig_arch.py', 'fig_nulls.py', 'fig_nulltable.py']


def md(s): return dict(cell_type='markdown', metadata={}, source=s)
def code(s): return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)


cells = [md('# HIVE-Bic: selection-honest biclustering of multi-tissue transcriptomes\n\n'
            'Honest Inference Via thinned Evidence. This notebook reproduces the full study on the free Colab tier (2 CPUs, no GPU):\n\n'
            '1. download the pig tissue atlas E-MTAB-5895, NCBI orthologs, GO sets and GTEx v10;\n'
            '2. prepare the 2,000-gene ortholog matrix;\n'
            '3. run HIVE-Bic on one semi-synthetic matrix (about 1 minute);\n'
            '4. run the benchmark (main, null, ablation, sensitivity), the real-data study and the statistics;\n'
            '5. draw the figures.\n\n'
            'Set `HIVE_SEEDS` below to trade run time for precision: 12 seeds reproduce the paper (several hours on 2 CPUs); 3 seeds give a quick check.'),
         code('!pip -q install --no-deps biclustlib\n!pip -q install bottleneck huggingface_hub pyarrow scikit-learn scipy pandas matplotlib\n'
              'import os\nos.makedirs("hive", exist_ok=True); os.chdir("hive")\n'
              'os.environ["HIVE_DATA"] = os.path.abspath("data"); os.environ["HIVE_OUT"] = os.path.abspath("results"); os.environ["HIVE_FIGS"] = os.path.abspath("figs")\n'
              'os.environ["HIVE_SEEDS"] = "3"   # 12 for the paper\n'
              'for d in ("data", "results", "figs"): os.makedirs(d, exist_ok=True)'),
         md('## 1. Source code\nEach cell writes one module of the package.')]
for m in MODS:
    src = open(os.path.join(H, m)).read()
    cells.append(code(f'%%writefile {m}\n' + src))
cells += [
    md('## 2. Data\nThe raw counts come from the Expression Atlas FTP site. If the download is blocked, fetch the raw counts of E-MTAB-5895 by hand from the Expression Atlas Downloads tab and upload the file into `hive/data/`.'),
    code('!python download_data.py\n!python data_prep.py\n!python gtex_fetch.py'),
    md('## 3. A single semi-synthetic run'),
    code('import numpy as np, pandas as pd\nfrom hive import *\nfrom synth import *\n'
         'D = os.environ["HIVE_DATA"]\nX = pd.read_csv(f"{D}/pig_counts_2000.tsv", sep="\\t", index_col=0); meta = pd.read_csv(f"{D}/pig_samples.tsv", sep="\\t", index_col=0)\n'
         'lib = meta.lib_size.values; groups = meta.tissue.values\nphi = estimate_dispersion(X.values, lib, groups); base = (X.values / size_factors(lib)).mean(1)\n'
         'rng = np.random.default_rng(7)\nXs, truth = make_matrix(base, phi, lib, rng, 2.0)\n'
         'res = run_hive(Xs, lib, groups, seed=7)\n'
         'for c in res["candidates"]:\n    print(len(c["I"]), "genes x", len(c["J"]), "samples | T =", round(c["T"], 3), "| p =", round(c["p"], 4), "| certified:", c["certified"])\n'
         'found = [(c["I"], c["J"]) for c in res["candidates"] if c["certified"]]\nprint("recovery / relevance:", match_scores(found, truth)[:2])'),
    md('## 4. Experiments\nResumable: re-running a cell skips finished jobs.'),
    code('for exp in ("main", "null", "ablation", "sens"):\n    !python run_synth.py {exp} 0 1'),
    code('!python real_data.py 10\n!python real_analysis.py'),
    md('## 5. Statistics and figures'),
    code('!python analysis.py\n!python fig_arch.py\n!python fig_nulls.py\n!python fig_nulltable.py\nimport glob\nfrom IPython.display import Image, display\nfor f in sorted(glob.glob("figs/*.png")):\n    print(f); display(Image(f, width=700))'),
]
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                                     language_info=dict(name='python'), colab=dict(provenance=[])), nbformat=4, nbformat_minor=5)
json.dump(nb, open(os.path.join(H, 'HIVE_Bic_Colab.ipynb'), 'w'), indent=1)
print('notebook written with', len(cells), 'cells')
