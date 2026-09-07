# PPB_pivot

Working collection of PPB-only numbers and figures for the manuscript.

Recompile after new PPB runs:

```bash
conda activate dc
cd Project_1
python scripts/compile_ppb_pivot.py
```

## Layout

| Path | Contents |
|---|---|
| `tables/leaderboard_ppb.csv` | All PPB rows from `results/leaderboard.csv` |
| `tables/paper_table_ppb.csv` | Mean ± std for the paper model set (normalized and original units) |
| `tables/summary_ppb.csv` | Same summary with extra columns |
| `tables/random_vs_scaffold_rf.csv` | RF scaffold vs random split |
| `tables/conformal_ppb.csv` | RF conformal (and ensemble) PICP / MPIW |
| `metrics/` | Per-seed `metrics.json` copies |
| `figures/learning_curves/` | Existing PPB learning-curve plots |
| `figures/chemberta_tuning/` | ChemBERTa LR / epoch search plots |
| `sources/` | PPB-filtered figure-source CSVs and z-score stats |
| `STATUS.json` | Which paper models are present vs still running |

## Paper model set

RF, XGBoost, ChemBERTa, GIN, GCN, GAT, ChemBERTa+GIN, ChemBERTa+GCN, ChemBERTa+GAT.

Fusion schedule is the same for all three graph branches: 20 epochs, ChemBERTa frozen for 2, text LR 1e-5, graph/head LR 1e-3, batch 16. Graph stacks match the standalone GNNs (3 × 128, dropout 0.25, GAT heads = 4).

Seeds: 42, 43, 44. One fixed Bemis–Murcko scaffold split.

Original-unit RMSE uses PPB label std = 16.60849107 (percent bound).

Do not drop tree baselines if they lose. Rank from the table after ChemBERTa+GCN and ChemBERTa+GAT are in.
