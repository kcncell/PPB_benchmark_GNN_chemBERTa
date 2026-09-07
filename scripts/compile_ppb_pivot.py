#!/usr/bin/env python
"""Compile PPB-only tables, metrics, and figures into Project_1/PPB_pivot.

Re-run after new PPB experiments (e.g. ChemBERTa+GCN/GAT fusion) so the
pivot folder stays the single source for the PPB-only manuscript.

Usage:
    conda activate dc
    cd Project_1
    python scripts/compile_ppb_pivot.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIVOT = ROOT / "PPB_pivot"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
LB = RESULTS / "leaderboard.csv"
NORM = DOCS / "normalization_stats.json"

MAIN_MODELS = [
    "RandomForest_ECFP",
    "XGBoost_ECFP",
    "ChemBERTa_zinc_base_v1_lr1e-05_e20",
    "PyG_GIN_h128_e100",
    "PyG_GCN_h128_e100",
    "PyG_GAT_h128_e100",
    "Multimodal_ChemBERTa_GIN_e20",
    "Multimodal_ChemBERTa_GCN_e20",
    "Multimodal_ChemBERTa_GAT_e20",
    "TripleGNN_GCN_GIN_GAT_h128_e100",
    "Multimodal_ChemBERTa_GCN_GIN_GAT_e20",
]
MODEL_LABELS = {
    "RandomForest_ECFP": "RF + ECFP",
    "XGBoost_ECFP": "XGBoost + ECFP",
    "ChemBERTa_zinc_base_v1_lr1e-05_e20": "ChemBERTa",
    "PyG_GIN_h128_e100": "GIN",
    "PyG_GCN_h128_e100": "GCN",
    "PyG_GAT_h128_e100": "GAT",
    "Multimodal_ChemBERTa_GIN_e20": "ChemBERTa+GIN",
    "Multimodal_ChemBERTa_GCN_e20": "ChemBERTa+GCN",
    "Multimodal_ChemBERTa_GAT_e20": "ChemBERTa+GAT",
    "TripleGNN_GCN_GIN_GAT_h128_e100": "GCN+GIN+GAT",
    "Multimodal_ChemBERTa_GCN_GIN_GAT_e20": "ChemBERTa+GCN+GIN+GAT",
    "RandomForest_ECFP_random": "RF + ECFP (random split)",
    "RF_conformal": "RF + conformal",
    "RF_ensemble5_conformal": "RF ensemble + conformal",
}


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _summarize(df: pd.DataFrame, ppb_std: float) -> pd.DataFrame:
    g = (
        df.groupby("model", as_index=False)
        .agg(
            n=("r2", "count"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_norm_mean=("rmse", "mean"),
            rmse_norm_std=("rmse", "std"),
            mae_norm_mean=("mae", "mean"),
            train_r2_mean=("train_r2", "mean"),
            valid_r2_mean=("valid_r2", "mean"),
        )
    )
    g["rmse_orig_mean"] = g["rmse_norm_mean"] * ppb_std
    g["rmse_orig_std"] = g["rmse_norm_std"] * ppb_std
    g["label"] = g["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return g.sort_values("r2_mean", ascending=False)


def main() -> None:
    tables = PIVOT / "tables"
    metrics_out = PIVOT / "metrics"
    figures_out = PIVOT / "figures"
    sources = PIVOT / "sources"
    for d in (tables, metrics_out, figures_out, sources):
        d.mkdir(parents=True, exist_ok=True)

    with NORM.open() as f:
        norm = json.load(f)
    ppb_std = float(norm["ppb"]["std"])
    ppb_mean = float(norm["ppb"]["mean"])
    (sources / "normalization_ppb.json").write_text(
        json.dumps(
            {
                "dataset": "ppb",
                "n": 1614,
                "split": "Bemis-Murcko scaffold 80/10/10 (one fixed cut)",
                "train_valid_test": [1291, 161, 162],
                "mean": ppb_mean,
                "std": ppb_std,
                "unit": norm["ppb"]["unit"],
                "seeds": [42, 43, 44],
                "note": "z-score labels; R2 is scale-invariant; RMSE_orig = RMSE_norm * std",
            },
            indent=2,
        )
        + "\n"
    )

    df = pd.read_csv(LB)
    ppb = df[df["dataset"] == "ppb"].copy()
    ppb.to_csv(tables / "leaderboard_ppb.csv", index=False)

    main_rows = ppb[ppb["model"].isin(MAIN_MODELS)].copy()
    summary = _summarize(main_rows, ppb_std)
    summary.to_csv(tables / "summary_ppb.csv", index=False)

    display = summary[
        [
            "label",
            "model",
            "n",
            "r2_mean",
            "r2_std",
            "rmse_norm_mean",
            "rmse_norm_std",
            "rmse_orig_mean",
            "rmse_orig_std",
            "train_r2_mean",
            "valid_r2_mean",
        ]
    ].copy()
    display.to_csv(tables / "paper_table_ppb.csv", index=False)

    rf_scaf = ppb[ppb["model"] == "RandomForest_ECFP"].copy()
    rf_rand = ppb[ppb["model"] == "RandomForest_ECFP_random"].copy()
    split_cmp = pd.concat(
        [_summarize(rf_scaf, ppb_std), _summarize(rf_rand, ppb_std)],
        ignore_index=True,
    )
    split_cmp.to_csv(tables / "random_vs_scaffold_rf.csv", index=False)

    conf = ppb[ppb["model"].isin(["RF_conformal", "RF_ensemble5_conformal"])].copy()
    conf.to_csv(tables / "conformal_ppb.csv", index=False)

    src_ppb = RESULTS / "ppb"
    if src_ppb.exists():
        for metrics_json in src_ppb.glob("*/*/metrics.json"):
            rel = metrics_json.relative_to(src_ppb)
            _copy_file(metrics_json, metrics_out / rel)

    for src_name in (
        "figure_source_summary.csv",
        "figure_source_summary_adme.csv",
        "figure_source_histories.csv",
        "figure_source_histories_adme.csv",
    ):
        src = FIGURES / src_name
        if not src.exists():
            continue
        hist = pd.read_csv(src)
        if "dataset" in hist.columns:
            hist = hist[hist["dataset"] == "ppb"].copy()
        hist.to_csv(sources / src_name.replace(".csv", "_ppb.csv"), index=False)

    lc_dir = figures_out / "learning_curves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES.glob("learning_curve_ppb_*"):
        _copy_file(fig, lc_dir / fig.name)

    tune_src = ROOT / "datasets" / "ppb" / "hyperparameter_tuning"
    tune_dst = figures_out / "chemberta_tuning"
    if tune_src.exists():
        for fig in list(tune_src.glob("*.png")) + list(tune_src.glob("*.pdf")) + list(
            tune_src.glob("*.csv")
        ):
            _copy_file(fig, tune_dst / fig.name)

    pending = [
        m
        for m in (
            "Multimodal_ChemBERTa_GCN_e20",
            "Multimodal_ChemBERTa_GAT_e20",
            "TripleGNN_GCN_GIN_GAT_h128_e100",
        )
        if m not in set(main_rows["model"].unique())
    ]
    status = {
        "ppb_leaderboard_rows": int(len(ppb)),
        "main_models_present": sorted(main_rows["model"].unique().tolist()),
        "fusion_still_pending": pending,
        "n_metric_jsons": len(list(metrics_out.glob("*/*/metrics.json"))),
    }
    (PIVOT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

    print("PPB_pivot compiled")
    print(f"  rows: {len(ppb)}")
    print(f"  main models: {status['main_models_present']}")
    if pending:
        print(f"  still pending: {pending}")
    print(summary[["label", "n", "r2_mean", "r2_std", "rmse_orig_mean"]].to_string(index=False))
    print(f"Wrote {PIVOT}")


if __name__ == "__main__":
    main()
