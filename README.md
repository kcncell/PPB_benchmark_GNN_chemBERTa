# PPB_benchmark_GNN_chemBERTa

Code, figures, and leaderboards for:

**Plasma Protein Binding Prediction under Scaffold Split: Tree Models, ChemBERTa, Graph Networks, and Late Fusion**

Pradyumna Kumar Pradhan

https://github.com/kcncell/PPB_benchmark_GNN_chemBERTa

## What is in this repository

```
scripts/     train, plot, and download
shared/      data loaders, metrics, seeds
results/     paper tables and per-seed metrics.json
figures/     main-text Figures 1–5 and Supplementary Figures 1–4
environment.yml
requirements_paper.txt
requirements_dc_freeze.txt
```

## Environment

Python 3.10, conda env `dc`. Versions used in the paper:

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

`requirements_paper.txt` lists those packages. `requirements_dc_freeze.txt` is a full dump from the machine that produced the numbers; it may not pip-install cleanly elsewhere.

## Data

The PPB set is loaded with DeepChem `load_ppb` (n = 1614). One Bemis–Murcko scaffold split (1291 / 161 / 162). Seeds 42, 43, and 44 change training, not the split. Z-score normalization is fit on the training fold only.

```bash
python scripts/download_datasets.py --datasets ppb
```

## Reproduce

```bash
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
python scripts/plot_publication_figures.py
python scripts/plot_architecture_schematics.py
```

Reported means are in `results/paper_table.csv`. Per-seed rows are in `results/leaderboard.csv`. The RF random-split control is in `results/random_vs_scaffold_rf.csv`.

## Figures

| Manuscript | File |
|---|---|
| Figure 1 | `figures/fig1_test_r2_bars.pdf` |
| Figure 2 | `figures/fig3_seed_r2.pdf` |
| Figure 3 | `figures/fig4_generalization_gap.pdf` |
| Figure 4 | `figures/fig5_loss_curves_all.pdf` |
| Figure 5 | `figures/fig7_delta_vs_rf.pdf` |
| Supplementary Figure 1 | `figures/figS1_arch_standalone_gnn.pdf` |
| Supplementary Figure 2 | `figures/figS2_arch_chemberta_gnn.pdf` |
| Supplementary Figure 3 | `figures/figS3_arch_triple_gnn.pdf` |
| Supplementary Figure 4 | `figures/figS4_arch_fusion_schedule.pdf` |

## License

MIT. PPB labels come from MoleculeNet / DeepChem; follow those source terms when redistributing data.
