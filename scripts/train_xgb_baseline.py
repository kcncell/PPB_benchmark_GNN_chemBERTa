#!/usr/bin/env python
"""Multi-seed XGBoost + ECFP baseline for Project 1."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_loaders import list_datasets, load_dataset  # noqa: E402
from shared.metrics import metrics_to_row, regression_metrics  # noqa: E402
from shared.results_io import append_leaderboard, save_metrics  # noqa: E402
from shared.seed import set_seed  # noqa: E402

DEFAULT_DATASETS = ["ppb", "hopv", "clearance"]


def train_one(dataset_name: str, seed: int, n_estimators: int = 300) -> dict:
    set_seed(seed)
    loaded = load_dataset(dataset_name, featurizer="ECFP", splitter=None)
    X_train = np.asarray(loaded.train.X)
    y_train = np.asarray(loaded.train.y, dtype=float)
    multitask = y_train.ndim > 1 and y_train.shape[1] > 1

    t0 = time.time()
    if not multitask:
        y_fit = y_train.ravel()
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_fit)
        preds = {}
        for name, ds in [("train", loaded.train), ("valid", loaded.valid), ("test", loaded.test)]:
            yp = model.predict(np.asarray(ds.X))
            yt = np.asarray(ds.y, dtype=float).ravel()
            preds[name] = regression_metrics(yt, yp)
    else:
        # One model per task for multitask
        n_tasks = y_train.shape[1]
        models = []
        for t in range(n_tasks):
            m = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed + t,
                n_jobs=-1,
                tree_method="hist",
            )
            m.fit(X_train, y_train[:, t])
            models.append(m)

        preds = {}
        for name, ds in [("train", loaded.train), ("valid", loaded.valid), ("test", loaded.test)]:
            X = np.asarray(ds.X)
            yt = np.asarray(ds.y, dtype=float)
            yp = np.column_stack([m.predict(X) for m in models])
            preds[name] = regression_metrics(yt, yp, multitask=True)

    train_time = time.time() - t0
    payload = {
        "dataset": loaded.name,
        "model": "XGBoost_ECFP",
        "seed": seed,
        "splitter": loaded.splitter,
        "n_estimators": n_estimators,
        "train_time_sec": round(train_time, 3),
        "train": preds["train"],
        "valid": preds["valid"],
        "test": preds["test"],
    }
    save_metrics(loaded.name, "XGBoost_ECFP", seed, payload)
    row = metrics_to_row(
        dataset=loaded.name,
        model="XGBoost_ECFP",
        seed=seed,
        split=loaded.splitter,
        split_name="test",
        metrics=preds["test"],
        extra={
            "train_r2": preds["train"]["r2"],
            "valid_r2": preds["valid"]["r2"],
            "train_time_sec": round(train_time, 3),
            "n_estimators": n_estimators,
        },
    )
    append_leaderboard(row)
    print(
        f"[{loaded.name}] XGB seed={seed} test R2={preds['test']['r2']:.4f} "
        f"RMSE={preds['test']['rmse']:.4f} ({train_time:.1f}s)"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset.lower()]
    for ds in datasets:
        for seed in seeds:
            try:
                train_one(ds, seed, n_estimators=args.n_estimators)
            except Exception as e:
                print(f"[ERROR] {ds} seed={seed}: {e}")


if __name__ == "__main__":
    main()
