# PPB_benchmark_GNN_chemBERTa

Reproducible code, figures, leaderboards, and supplementary files for:

**Plasma Protein Binding Prediction under Scaffold Split: Tree Models, ChemBERTa, Graph Networks, and Late Fusion**

Pradyumna Kumar Pradhan

Repository: https://github.com/kcncell/PPB_benchmark_GNN_chemBERTa

ChemRxiv preprint and this repository will be made public together.

This folder is a frozen copy of the files that go to GitHub. Edit the working Project_1 tree, not this snapshot.

## Paper scope

Plasma protein binding (PPB) from DeepChem MoleculeNet (`load_ppb`, n = 1614). One fixed Bemis–Murcko scaffold split (1291 / 161 / 162) and training seeds 42, 43, and 44.

Models: random forest + ECFP, XGBoost + ECFP, ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`), GIN, GCN, GAT, ChemBERTa+GIN/GCN/GAT late fusion, and a GCN+GIN+GAT concat control (no ChemBERTa). Random-split control is RF-only.

## Layout

```
scripts/                 # train, plot, download, compile
shared/                  # loaders, metrics, seeds, device, results I/O
results/                 # leaderboard.csv and per-seed metrics.json
results/ppb/             # PPB runs used in the paper
PPB_pivot/               # PPB-only tables and copies of metrics
figures/                 # main figures and supplementary schematics
supplementary/           # map of Supplementary Figures S1–S4
docs/manuscript/         # LaTeX/Word source
environment.yml
requirements_paper.txt
requirements_dc_freeze.txt
data/raw/PPB.csv         # optional CSV copy; prefer DeepChem download
```

## Environment

Python 3.10, conda env `dc`. Versions in the manuscript:

- DeepChem 2.8.0
- PyTorch 2.10.0 (MPS on Apple M1; CUDA numbers may differ slightly)
- PyTorch Geometric 2.8.0
- Hugging Face Transformers 5.13.0
- RDKit 2025.9.5
- scikit-learn 1.7.2
- XGBoost 3.2.0

```bash
conda activate dc
```

`requirements_dc_freeze.txt` is a full dump from the machine that produced the paper numbers. It includes conda-local URLs and may not pip-install cleanly elsewhere. Use `requirements_paper.txt` plus conda-forge RDKit for a portable stack.

## Data

```bash
conda activate dc
python scripts/download_datasets.py --datasets ppb
```

Z-score normalization is fit on the training fold only (PPB label std ≈ 16.61 percent bound).

## Reproduce the PPB leaderboard

From the repository root, with env `dc` active:

```bash
python scripts/download_datasets.py --datasets ppb

python scripts/train_rf_baseline.py --dataset ppb --seeds 42,43,44
python scripts/train_rf_baseline.py --dataset ppb --seeds 42,43,44 --splitter random
python scripts/train_xgb_baseline.py --dataset ppb --seeds 42,43,44
python scripts/train_chemberta.py --dataset ppb --seeds 42,43,44
python scripts/train_pyg_gnn.py --dataset ppb --seeds 42,43,44 --arch gin
python scripts/train_pyg_gnn.py --dataset ppb --seeds 42,43,44 --arch gcn
python scripts/train_pyg_gnn.py --dataset ppb --seeds 42,43,44 --arch gat

python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --graph-encoder gin
python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --graph-encoder gcn
python scripts/train_multimodal_fusion.py --dataset ppb --seeds 42,43,44 --graph-encoder gat
python scripts/train_gnn_triple_fusion.py --dataset ppb --seeds 42,43,44

python scripts/compile_ppb_pivot.py
python scripts/plot_publication_figures.py
python scripts/plot_architecture_schematics.py
```

Reported PPB test R² (mean ± sample std over seeds 42–44) is in `PPB_pivot/tables/paper_table_ppb.csv` and `results/leaderboard.csv`.

## Figures

| Manuscript | File |
|---|---|
| Figure 1 | `figures/fig1_test_r2_bars.pdf` |
| Figure 2 | `figures/fig3_seed_r2.pdf` |
| Figure 3 | `figures/fig4_generalization_gap.pdf` |
| Figure 4 | `figures/fig5_loss_curves_all.pdf` |
| Figure 5 | `figures/fig7_delta_vs_rf.pdf` |

Supplementary schematics: see `supplementary/README.md`.

## License

MIT. PPB labels come from MoleculeNet / DeepChem; follow those source terms when redistributing data.
