#!/usr/bin/env python
"""Summarize Project 1 leaderboard: mean±std by dataset × model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LB = ROOT / "results" / "leaderboard.csv"
OUT = ROOT / "results" / "leaderboard_summary.csv"


def main():
    df = pd.read_csv(LB)
    # Prefer paper-relevant models
    keep_patterns = [
        "RandomForest_ECFP",
        "XGBoost_ECFP",
        "ChemBERTa_zinc_base_v1_lr1e-05_e20",
        "PyG_GIN",
        "PyG_GCN",
        "PyG_GAT",
        "Multimodal_ChemBERTa",
        "RF_ensemble",
        "RF_conformal",
        "RF_MAPIE",
    ]
    mask = df["model"].astype(str).apply(
        lambda m: any(p in m for p in keep_patterns)
    )
    # drop HPPB duplicate rows for summary
    sub = df[mask & (df["dataset"] != "hppb")].copy()
    if sub.empty:
        print("No matching rows yet.")
        return

    g = (
        sub.groupby(["dataset", "model"], as_index=False)
        .agg(
            n_seeds=("r2", "count"),
            test_r2_mean=("r2", "mean"),
            test_r2_std=("r2", "std"),
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            train_r2_mean=("train_r2", "mean"),
            valid_r2_mean=("valid_r2", "mean"),
        )
        .sort_values(["dataset", "test_r2_mean"], ascending=[True, False])
    )
    g.to_csv(OUT, index=False)
    print(g.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
